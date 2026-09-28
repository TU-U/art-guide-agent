import base64
import json
import logging
from typing import Any, Dict, List, Optional

import requests
try:
    import httpx
except Exception:
    httpx = None  # type: ignore
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from config import Config
from services.rule_router import RouteDecision, RuleRouter

logger = logging.getLogger("VLM")


class VLMService:
    """Qwen Omni / DeepSeek VLM service with a lightweight rule router."""

    def __init__(self, model_name: str = Config.VLM_MODEL_NAME):
        self.provider = Config.VLM_PROVIDER
        self.model_name = Config.VLM_MODEL_NAME
        if model_name != self.model_name:
            logger.warning(
                "[INIT] requested model=%s ignored, forcing provider=%s model=%s",
                model_name,
                self.provider,
                self.model_name,
            )
        self.router = RuleRouter()
        self.model_status = "ready"
        self._qwen_client = None
        logger.info(
            "[INIT] init VLM service provider=%s model=%s",
            self.provider,
            self.model_name,
        )

    def _guess_image_mime_type(self, image_data: bytes) -> str:
        if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
            return "image/gif"
        if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    def _image_to_data_url(self, image_data: bytes) -> str:
        mime_type = self._guess_image_mime_type(image_data)
        encoded = base64.b64encode(image_data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _history_text(self, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return ""
        recent = history[-4:]
        return "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in recent
            if item.get("content")
        )

    def _profile_text(self, memory_profile: Optional[Dict[str, Any]]) -> str:
        if not memory_profile:
            return ""
        style_parts = [
            memory_profile.get("category_name", ""),
            memory_profile.get("communication_pref", ""),
            memory_profile.get("aesthetic_pref", ""),
        ]
        return " / ".join(part for part in style_parts if part)

    def _route_system_prompt(self, decision: RouteDecision) -> str:
        base = (
            "你是一个实验室场景下的多模态助手。"
            "回答要简洁、直接、准确。"
            "看不清或无法确认时，要明确说明不确定，不要编造。"
        )

        if decision.route == "text_only":
            return base + " 当前轮次不需要依赖图像，请按纯文本问题作答。"
        if decision.route == "vision_describe":
            return base + " 当前任务是描述图像内容，优先说明主要物体、相对位置和显著特征。"
        if decision.route == "pointed_object_qa":
            return (
                base
                + " 用户很可能在询问画面中的某个单独物体。"
                + " 如果问题里出现“这个”“那个”“它”“我指的”，请优先锁定最显著、最可能被关注的单个物体回答。"
            )
        if decision.route == "vision_qa":
            return base + " 当前任务是基于图像回答视觉问题。"
        return base + " 当前任务需要结合图像和文本一起回答。"

    def _build_user_text(
        self,
        question: str,
        decision: RouteDecision,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        sections: List[str] = []

        route_hint_map = {
            "text_only": "本轮路由：纯文本回答。",
            "vision_describe": "本轮路由：图像描述。",
            "vision_qa": "本轮路由：视觉问答。",
            "pointed_object_qa": "本轮路由：指向物体问答。",
            "vision_contextual": "本轮路由：图像增强回答。",
        }
        sections.append(route_hint_map.get(decision.route, "本轮路由：多模态回答。"))

        history_text = self._history_text(history)
        if history_text:
            sections.append("最近对话：\n" + history_text)

        if memory_context.strip():
            sections.append("记忆补充：\n" + memory_context.strip())

        profile_text = self._profile_text(memory_profile)
        if profile_text:
            sections.append("用户画像参考：\n" + profile_text)

        if decision.route == "pointed_object_qa":
            sections.append("请优先关注图中最显著、最可能被用户指向的单个物体。")

        sections.append("用户问题：\n" + (question.strip() or "请描述当前画面。"))
        return "\n\n".join(section for section in sections if section.strip())

    def _build_messages(
        self,
        question: str,
        decision: RouteDecision,
        image_data: Optional[bytes] = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        system_prompt = self._route_system_prompt(decision)
        user_text = self._build_user_text(
            question=question,
            decision=decision,
            history=history,
            memory_context=memory_context,
            memory_profile=memory_profile,
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        if decision.use_image and image_data:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_to_data_url(image_data)},
                        },
                    ],
                }
            )
            return messages

        messages.append({"role": "user", "content": user_text})
        return messages

    def _extract_content(self, message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif "text" in part:
                        parts.append(part.get("text", ""))
            return "".join(parts)
        return ""

    def _get_qwen_client(self):
        if self._qwen_client is not None:
            return self._qwen_client
        if OpenAI is None:
            raise RuntimeError("openai package unavailable, please install openai")
        if httpx is None:
            raise RuntimeError("httpx package unavailable, please install httpx")
        if not Config.QWEN_OMNI_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用 Qwen Omni API")
        http_client = httpx.Client(timeout=120.0, follow_redirects=True)
        self._qwen_client = OpenAI(
            api_key=Config.QWEN_OMNI_API_KEY,
            base_url=Config.QWEN_OMNI_BASE_URL,
            http_client=http_client,
        )
        return self._qwen_client

    def _call_deepseek(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_image: bool = False,
    ) -> str:
        if not Config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法调用 DeepSeek API")

        url = Config.DEEPSEEK_BASE_URL.rstrip("/") + Config.DEEPSEEK_CHAT_PATH
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek returned invalid payload: {data}")

        message = choices[0].get("message") or {}
        content = self._extract_content(message)
        if not content:
            raise RuntimeError("DeepSeek returned empty content")
        return content

    def _call_qwen_omni(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_image: bool = False,
    ) -> str:
        del use_image
        client = self._get_qwen_client()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
        )
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise RuntimeError(f"Qwen Omni returned invalid payload: {completion}")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("Qwen Omni returned empty message")
        content = getattr(message, "content", "")
        if isinstance(content, str):
            if not content:
                raise RuntimeError("Qwen Omni returned empty content")
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    text = getattr(part, "text", "")
                    if text:
                        parts.append(text)
            merged = "".join(parts)
            if not merged:
                raise RuntimeError("Qwen Omni returned empty content")
            return merged
        raise RuntimeError("Qwen Omni returned unsupported content type")

    def _call_provider(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_image: bool = False,
    ) -> str:
        if self.provider == "qwen_omni":
            return self._call_qwen_omni(messages, use_image=use_image)
        if self.provider == "deepseek":
            return self._call_deepseek(messages, use_image=use_image)
        raise RuntimeError(f"Unsupported VLM provider: {self.provider}")

    def call_text_model(self, messages: List[Dict[str, Any]]) -> str:
        return self._call_provider(messages, use_image=False)

    def _generate_answer(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        image_data: Optional[bytes] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        question = " ".join(prompt.strip() for prompt in prompts if prompt and prompt.strip())
        question = question or "请描述当前画面。"
        decision = self.router.route(question, image_data=image_data, history=history)
        messages = self._build_messages(
            question=question,
            decision=decision,
            image_data=image_data if decision.use_image else None,
            history=history,
            memory_context=memory_context,
            memory_profile=memory_profile,
        )

        self.model_status = "generating"
        try:
            logger.info(
                "[VLM] route=%s use_image=%s reason=%s",
                decision.route,
                decision.use_image,
                decision.reason,
            )
            answer = self._call_provider(messages, use_image=decision.use_image)
            self.model_status = "ready"
            return {
                "answer": answer,
                "question": question,
                "topic_type": decision.route,
                "topic_subject": "",
                "retrieval_query": "",
                "memory_context": memory_context,
                "provider": self.provider,
                "model_name": self.model_name,
                "success": True,
                "route": decision.route,
                "route_reason": decision.reason,
                "used_image": decision.use_image,
            }
        except Exception as exc:
            logger.error("[VLM] generation failed provider=%s: %s", self.provider, exc)
            self.model_status = "ready"
            return {
                "answer": "当前视觉模型服务不可用，请稍后重试。",
                "question": question,
                "topic_type": decision.route,
                "topic_subject": "",
                "retrieval_query": "",
                "memory_context": memory_context,
                "provider": self.provider,
                "model_name": self.model_name,
                "success": False,
                "error": str(exc),
                "route": decision.route,
                "route_reason": decision.reason,
                "used_image": decision.use_image,
            }

    def generate_stream(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        image_data: Optional[bytes] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ):
        result = self._generate_answer(prompts, history, image_data, memory_context, memory_profile)
        for char in result["answer"]:
            yield char

    def generate_sync(
        self,
        prompts: List[str],
        image_data: Optional[bytes] = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._generate_answer(prompts, history, image_data, memory_context, memory_profile)

    def generate_response_stream(
        self,
        image_data: Optional[bytes],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ):
        return self.generate_stream([question], history, image_data, memory_context, memory_profile)

    def generate_response_sync(
        self,
        image_data: Optional[bytes],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.generate_sync([question], image_data, history, memory_context, memory_profile)

    def load_model(self, model_config: Dict[str, Any]):
        try:
            requested_provider = str(model_config.get("provider", self.provider)).strip().lower()
            if requested_provider not in {"qwen_omni", "deepseek"}:
                logger.warning(
                    "[VLM] requested provider=%s ignored, keeping provider=%s",
                    requested_provider,
                    self.provider,
                )
                requested_provider = self.provider

            self.provider = requested_provider
            if self.provider == "qwen_omni":
                self.model_name = Config.QWEN_OMNI_MODEL_NAME
            else:
                self.model_name = Config.DEEPSEEK_MODEL_NAME

            logger.info("[VLM] model updated provider=%s model=%s", self.provider, self.model_name)
            self.model_status = "ready"
            return True
        except Exception as exc:
            logger.error("[VLM] update model config failed: %s", exc)
            self.model_status = "error"
            return False

    def get_model_status(self):
        return {
            "status": self.model_status,
            "provider": self.provider,
            "model_name": self.model_name,
            "resources": {"memory_usage": "N/A", "cpu_usage": "N/A"},
        }

    def update_prompt_template(self, template_name: str, content: str):
        del template_name
        del content
        logger.info("[VLM] prompt templates are handled inline; update ignored")
        return True

    def inject_context(self, context_data: Dict[str, Any]):
        logger.info("[VLM] inject context keys=%s", list(context_data.keys()))
        return True
