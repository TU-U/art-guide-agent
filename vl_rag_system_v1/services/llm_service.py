import json
import logging
import os
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

logger = logging.getLogger("LLM")

try:
    import ollama  # type: ignore
except Exception:
    ollama = None


class LLMService:
    """Text LLM service for RAG, prompt assembly, and chat generation."""

    MAX_ANSWER_CHARS = 100

    def __init__(self, model_name: str = Config.LLM_MODEL_NAME):
        self.provider = Config.LLM_PROVIDER.lower().strip() or "qwen_omni"
        self.model_name = model_name
        self.exhibit_rag_enabled = bool(getattr(Config, "ENABLE_EXHIBIT_RAG", False))
        self.retriever = None
        self.prompt_templates: Dict[str, str] = {}
        self.model_status = "ready"
        self._qwen_client = None
        logger.info(
            "[INIT] init LLM service provider=%s model=%s exhibit_rag=%s",
            self.provider,
            self.model_name,
            self.exhibit_rag_enabled,
        )
        self._load_prompt_templates()

    def _get_retriever(self):
        if self.retriever is None:
            from rag.retriever import MuseumRetriever

            self.retriever = MuseumRetriever()
        return self.retriever

    def _load_prompt_templates(self):
        try:
            for filename in os.listdir(Config.PROMPTS_DIR):
                if filename.endswith(".md"):
                    template_name = filename.replace(".md", "")
                    self.prompt_templates[template_name] = self._load_prompt(filename)
            logger.info("[PROMPT] loaded templates: %s", list(self.prompt_templates.keys()))
        except Exception as exc:
            logger.warning("[PROMPT] load templates failed: %s", exc)

    def _load_prompt(self, filename: str, **kwargs) -> str:
        path = Config.PROMPTS_DIR / filename
        try:
            template = path.read_text(encoding="utf-8")
            return template.format(**kwargs) if kwargs else template
        except Exception as exc:
            logger.warning("[PROMPT] load %s failed: %s", filename, exc)
            if "system" in filename:
                return "你是一个艺术导览机器人。"
            return kwargs.get("question", "请分析这个问题")

    def _limit_answer(self, answer: str) -> str:
        text = (answer or "").strip()
        if len(text) <= self.MAX_ANSWER_CHARS:
            return text
        return text[: self.MAX_ANSWER_CHARS].rstrip("，,；;：:、 ")

    def _build_messages(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        question = " ".join(prompt.strip() for prompt in prompts if prompt and prompt.strip())
        if not question:
            question = "请介绍这个展品。"

        system_prompt = self.prompt_templates.get("system_prompt", "你是一个艺术导览机器人。")
        intent_info = self._analyze_intent(question, history)
        topic_type = intent_info["topic_type"]
        topic_subject = intent_info["topic_subject"]
        retrieval_top_k = intent_info.get("retrieval_top_k", Config.RETRIEVAL_TOP_K)

        retrieval_query = ""
        context = ""
        if self.exhibit_rag_enabled and topic_type in {"knowledge", "knowledge_followup", "knowledge_recommendation"}:
            retrieval_query = self._build_retrieval_query(question, history, intent_info)
            logger.info("[RAG] query: %s", retrieval_query)
            context = self._get_retriever().retrieve(retrieval_query, top_k=retrieval_top_k)
            if topic_type == "knowledge_recommendation":
                prompt_body = self._build_recommendation_prompt(question, context, topic_subject)
            else:
                prompt_body = self._load_prompt(
                    "identify_prompt.md",
                    context=context,
                    question=question,
                )
        elif topic_type in {"knowledge", "knowledge_followup", "knowledge_recommendation"}:
            logger.info("[RAG] exhibit rag disabled, skip retrieval for topic_type=%s", topic_type)
            retrieval_top_k = 0
            prompt_body = self._load_prompt(
                "identify_prompt.md",
                context="",
                question=question,
            )
        else:
            prompt_body = self._build_smalltalk_prompt(question, history, topic_subject)

        full_prompt = question
        if history:
            recent_history = history[-4:]
            history_text = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent_history
            )
            full_prompt = f"对话历史：\n{history_text}\n\n当前问题：{question}"

        memory_block = ""
        if memory_context.strip():
            memory_block += f"记忆补充：\n{memory_context.strip()}\n\n"
        if memory_profile:
            style_parts = [
                memory_profile.get("category_name", ""),
                memory_profile.get("communication_pref", ""),
                memory_profile.get("aesthetic_pref", ""),
            ]
            style_text = " / ".join(part for part in style_parts if part)
            if style_text:
                memory_block += f"用户画像参考：{style_text}\n\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": memory_block + prompt_body + "\n\n" + full_prompt},
        ]
        return {
            "question": question,
            "topic_type": topic_type,
            "topic_subject": topic_subject,
            "retrieval_top_k": retrieval_top_k,
            "retrieval_query": retrieval_query,
            "context": context,
            "memory_context": memory_context,
            "system_prompt": system_prompt,
            "messages": messages,
        }

    def _analyze_intent(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        normalized_question = question.strip()
        previous_user = ""
        previous_assistant = ""
        if history:
            user_msgs = [
                item.get("content", "").strip()
                for item in history[-6:]
                if item.get("role") == "user" and item.get("content", "").strip()
            ]
            assistant_msgs = [
                item.get("content", "").strip()
                for item in history[-6:]
                if item.get("role") == "assistant" and item.get("content", "").strip()
            ]
            previous_user = user_msgs[-1] if user_msgs else ""
            previous_assistant = assistant_msgs[-1] if assistant_msgs else ""

        smalltalk_keywords = [
            "你好", "谢谢", "再见", "哈哈", "陪我", "聊天", "心情", "天气",
            "你是谁", "你叫什么", "你觉得", "你会不会", "辛苦",
        ]
        summary_keywords = ["总结", "回顾", "梳理", "概括", "整理一下"]
        recommendation_keywords = [
            "介绍一个", "推荐一个", "随便介绍", "来一个展品", "挑一个", "选一个", "讲一个展品",
        ]
        knowledge_keywords = [
            "作品", "展品", "作者", "设计", "展区", "灵感", "动机", "技术", "特点",
            "背景", "介绍", "是什么", "为什么", "怎么实现", "材料", "创作", "理念",
        ]
        follow_up_markers = [
            "这个", "这件", "它", "那个", "继续", "再说", "再讲", "详细", "具体",
            "展开", "补充", "还有", "然后", "为什么", "怎么",
        ]

        looks_smalltalk = any(keyword in normalized_question for keyword in smalltalk_keywords)
        looks_summary = any(keyword in normalized_question for keyword in summary_keywords)
        looks_recommendation = any(keyword in normalized_question for keyword in recommendation_keywords)
        looks_knowledge = any(keyword in normalized_question for keyword in knowledge_keywords)
        looks_follow_up = any(marker in normalized_question for marker in follow_up_markers)
        short_question = len(normalized_question) <= 18

        previous_topic_subject = self._extract_topic_subject(previous_user or previous_assistant)
        recommendation_subject = self._extract_recommendation_preference(normalized_question)
        explicit_subject = self._extract_topic_subject(normalized_question)
        if previous_topic_subject and self._is_generic_followup_question(normalized_question):
            explicit_subject = previous_topic_subject

        if looks_summary:
            return {"topic_type": "smalltalk", "topic_subject": "会话总结", "retrieval_top_k": 0}
        if looks_recommendation:
            return {
                "topic_type": "knowledge_recommendation",
                "topic_subject": recommendation_subject or "展品推荐",
                "retrieval_top_k": 5,
            }
        if looks_smalltalk and not looks_knowledge:
            return {
                "topic_type": "smalltalk",
                "topic_subject": previous_topic_subject or "机器人交流",
                "retrieval_top_k": 0,
            }
        if looks_knowledge:
            return {
                "topic_type": "knowledge",
                "topic_subject": explicit_subject or previous_topic_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }
        if explicit_subject and previous_topic_subject and explicit_subject != previous_topic_subject:
            return {
                "topic_type": "knowledge",
                "topic_subject": explicit_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }
        if short_question and looks_follow_up and previous_topic_subject:
            return {
                "topic_type": "knowledge_followup",
                "topic_subject": previous_topic_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }
        if previous_topic_subject and (looks_follow_up or short_question):
            return {
                "topic_type": "knowledge_followup",
                "topic_subject": previous_topic_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }
        return {"topic_type": "smalltalk", "topic_subject": previous_topic_subject or "机器人交流", "retrieval_top_k": 0}

    def _extract_topic_subject(self, text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return ""

        generic_starts = ("这个", "这件", "它", "那个", "那件", "那它")
        if normalized.startswith(generic_starts):
            return ""

        quote_pairs = [("《", "》"), ("“", "”"), ('"', '"')]
        for left, right in quote_pairs:
            if left in normalized and right in normalized:
                start = normalized.find(left)
                end = normalized.find(right, start + 1)
                if start != -1 and end != -1 and end > start + 1:
                    return normalized[start + 1:end].strip()

        alias_markers = ["灵视", "永栖所", "ByteBunny", "萨满鼠", "虚拟偶像"]
        for marker in alias_markers:
            if marker.lower() in normalized.lower():
                return marker

        markers = ["作品", "展品", "作者", "设计", "展区", "技术", "灵感", "动机"]
        if any(marker in normalized for marker in markers):
            return normalized[:24]
        return ""

    def _is_generic_followup_question(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False

        quote_pairs = [("《", "》"), ("“", "”"), ('"', '"')]
        for left, right in quote_pairs:
            if left in normalized and right in normalized:
                return False

        alias_markers = ["灵视", "永栖所", "ByteBunny", "萨满鼠", "虚拟偶像"]
        if any(marker.lower() in normalized.lower() for marker in alias_markers):
            return False

        generic_markers = [
            "这个", "这件", "它", "那个", "那件", "那它",
            "有什么", "是什么", "为什么", "怎么", "背景", "理念", "技术", "作者",
        ]
        return len(normalized) <= 24 and any(marker in normalized for marker in generic_markers)

    def _extract_recommendation_preference(self, text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return ""

        preference_keywords = [
            "实用", "互动", "传统文化", "环保", "科技", "未来", "儿童", "空间",
            "材料", "交互", "可持续", "工业设计", "艺术与科技", "环境设计",
        ]
        matched = [keyword for keyword in preference_keywords if keyword in normalized]
        return "、".join(matched)

    def _build_retrieval_query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        intent_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        normalized_question = question.strip()
        if not history:
            return normalized_question

        recent_history = [item for item in history[-6:] if item.get("content")]
        recent_user_questions = [
            item["content"].strip()
            for item in recent_history
            if item.get("role") == "user" and item.get("content", "").strip()
        ]
        recent_assistant_answers = [
            item["content"].strip()
            for item in recent_history
            if item.get("role") == "assistant" and item.get("content", "").strip()
        ]

        previous_user = recent_user_questions[-1] if recent_user_questions else ""
        previous_assistant = recent_assistant_answers[-1] if recent_assistant_answers else ""
        follow_up_markers = [
            "它", "这个", "这件", "这个作品", "这件作品", "这个展品",
            "那个", "继续", "再说", "再讲", "详细", "具体", "展开", "补充", "还有", "然后", "为什么", "怎么",
        ]
        question_is_short = len(normalized_question) <= 18
        looks_like_follow_up = any(marker in normalized_question for marker in follow_up_markers)
        topic_subject = (intent_info or {}).get("topic_subject", "").strip()

        if (intent_info or {}).get("topic_type") == "knowledge_recommendation":
            recommendation_parts: List[str] = []
            if topic_subject and topic_subject != "展品推荐":
                recommendation_parts.append(f"筛选偏好：{topic_subject}")
            recommendation_parts.append(f"用户需求：{normalized_question}")
            recommendation_parts.append("请优先召回最符合偏好、适合被介绍的一件或几件展品")
            return "\n".join(recommendation_parts)

        if not question_is_short and not looks_like_follow_up and not topic_subject:
            return normalized_question

        topic_parts: List[str] = []
        if topic_subject:
            topic_parts.append(f"当前话题对象：{topic_subject}")
        if previous_user and previous_user != normalized_question:
            topic_parts.append(f"上一轮用户问题：{previous_user}")
        if previous_assistant:
            topic_parts.append(f"上一轮回答摘要：{previous_assistant[:120]}")
        topic_parts.append(f"当前追问：{normalized_question}")
        retrieval_query = "\n".join(topic_parts).strip()
        return retrieval_query or normalized_question

    def _build_recommendation_prompt(self, question: str, context: str, topic_subject: str = "") -> str:
        return self._load_prompt(
            "recommendation_prompt.md",
            topic_subject=topic_subject or "无明确偏好",
            context=context or "无可用参考信息",
            question=question,
        )

    def _build_smalltalk_prompt(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        topic_subject: str = "",
    ) -> str:
        history_hint = ""
        if history:
            recent = history[-4:]
            history_hint = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent
            )
        return self._load_prompt(
            "smalltalk_prompt.md",
            topic_subject=topic_subject or "无",
            history_hint=history_hint or "无",
            question=question,
        )

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

    def _call_deepseek(self, messages: List[Dict[str, Any]]) -> str:
        if not Config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法调用 DeepSeek API")

        url = Config.DEEPSEEK_BASE_URL.rstrip("/") + Config.DEEPSEEK_CHAT_PATH
        payload = {"model": self.model_name, "messages": messages, "stream": False}
        headers = {
            "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
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

    def _call_qwen_omni(self, messages: List[Dict[str, Any]]) -> str:
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

    def _call_ollama(self, messages: List[Dict[str, Any]], system_prompt: str) -> str:
        if ollama is None:
            raise RuntimeError("ollama is not installed or unavailable")

        response = ollama.chat(model=Config.OLLAMA_MODEL_NAME, messages=messages, stream=False)
        message = response.get("message", {})
        content = message.get("content", "")
        if content:
            return content

        prompt = ""
        last_message = messages[-1] if messages else {}
        last_content = last_message.get("content", "")
        if isinstance(last_content, str):
            prompt = last_content
        elif isinstance(last_content, list):
            prompt = self._extract_content({"content": last_content})

        fallback = ollama.generate(
            model=Config.OLLAMA_MODEL_NAME,
            system=system_prompt,
            prompt=prompt,
            stream=False,
        )
        content = fallback.get("response", "")
        if not content:
            raise RuntimeError("Ollama returned empty content")
        return content

    def call_text_model(self, messages: List[Dict[str, Any]]) -> str:
        if self.provider == "qwen_omni":
            return self._call_qwen_omni(messages)
        if self.provider == "deepseek":
            return self._call_deepseek(messages)
        if self.provider == "ollama":
            system_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            return self._call_ollama(messages, str(system_prompt))
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def _generate_answer(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        image_data: Optional[bytes] = None,
        memory_context: str = "",
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del image_data
        prepared = self._build_messages(prompts, history, memory_context, memory_profile)
        messages = [dict(item) for item in prepared["messages"]]

        self.model_status = "generating"
        try:
            if self.provider == "qwen_omni":
                logger.info("[LLM] call Qwen text path")
                answer = self._call_qwen_omni(messages)
            elif self.provider == "deepseek":
                logger.info("[LLM] call DeepSeek text path")
                answer = self._call_deepseek(messages)
            elif self.provider == "ollama":
                logger.info("[LLM] call Ollama")
                answer = self._call_ollama(messages, prepared["system_prompt"])
            else:
                raise RuntimeError(f"Unsupported LLM_PROVIDER: {self.provider}")
            answer = self._limit_answer(answer)

            self.model_status = "ready"
            return {
                "answer": answer,
                "context": prepared["context"],
                "question": prepared["question"],
                "topic_type": prepared["topic_type"],
                "topic_subject": prepared["topic_subject"],
                "retrieval_query": prepared["retrieval_query"],
                "memory_context": prepared["memory_context"],
                "provider": self.provider,
                "model_name": self.model_name,
                "success": True,
                "route": "text_only",
                "route_reason": "llm_service_text_path",
                "used_image": False,
            }
        except Exception as exc:
            logger.error("[LLM] generation failed: %s", exc)
            self.model_status = "error"

            if self.provider not in {"ollama", "qwen_omni"} and ollama is not None:
                try:
                    logger.warning("[LLM] fallback to Ollama after upstream failure")
                    answer = self._call_ollama(messages, prepared["system_prompt"])
                    self.model_status = "ready"
                    return {
                        "answer": answer,
                        "context": prepared["context"],
                        "question": prepared["question"],
                        "topic_type": prepared["topic_type"],
                        "topic_subject": prepared["topic_subject"],
                        "retrieval_query": prepared["retrieval_query"],
                        "memory_context": prepared["memory_context"],
                        "provider": "ollama_fallback",
                        "model_name": Config.OLLAMA_MODEL_NAME,
                        "success": True,
                        "route": "text_only",
                        "route_reason": "llm_service_text_path_fallback",
                        "used_image": False,
                    }
                except Exception as fallback_exc:
                    logger.error("[LLM] Ollama fallback failed: %s", fallback_exc)

            self.model_status = "ready"
            return {
                "answer": "当前模型服务不可用，请稍后重试。",
                "context": prepared["context"],
                "question": prepared["question"],
                "topic_type": prepared["topic_type"],
                "topic_subject": prepared["topic_subject"],
                "retrieval_query": prepared["retrieval_query"],
                "memory_context": prepared["memory_context"],
                "provider": self.provider,
                "model_name": self.model_name,
                "success": False,
                "error": str(exc),
                "route": "text_only",
                "route_reason": "llm_service_text_path_error",
                "used_image": False,
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

    def load_model(self, model_config: Dict[str, Any]):
        try:
            requested_provider = str(model_config.get("provider", self.provider)).strip().lower()
            if requested_provider not in {"qwen_omni", "deepseek", "ollama"}:
                logger.warning(
                    "[LLM] requested provider=%s ignored, keeping provider=%s",
                    requested_provider,
                    self.provider,
                )
                requested_provider = self.provider

            self.provider = requested_provider
            if self.provider == "qwen_omni":
                self.model_name = Config.QWEN_OMNI_MODEL_NAME
            elif self.provider == "deepseek":
                self.model_name = Config.DEEPSEEK_MODEL_NAME
            else:
                self.model_name = Config.OLLAMA_MODEL_NAME

            logger.info("[LLM] model config updated provider=%s model=%s", self.provider, self.model_name)
            self.model_status = "ready"
            return True
        except Exception as exc:
            logger.error("[LLM] update model config failed: %s", exc)
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
        try:
            self.prompt_templates[template_name] = content
            template_path = Config.PROMPTS_DIR / f"{template_name}.md"
            template_path.write_text(content, encoding="utf-8")
            logger.info("[PROMPT] updated template %s", template_name)
            return True
        except Exception as exc:
            logger.error("[PROMPT] update template failed: %s", exc)
            return False

    def inject_context(self, context_data: Dict[str, Any]):
        logger.info("[LLM] inject context keys=%s", list(context_data.keys()))
        return True

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
