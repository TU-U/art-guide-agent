from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class RouteDecision:
    route: str
    use_image: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "use_image": self.use_image,
            "reason": self.reason,
        }


class RuleRouter:
    SMALLTALK_KEYWORDS = (
        "你好", "您好", "谢谢", "再见", "拜拜", "你是谁", "你叫什么", "天气", "聊天",
    )
    POINTER_KEYWORDS = (
        "这个", "那个", "它", "我指的", "前面", "旁边", "手边", "桌上这个", "这里这个",
    )
    VISION_KEYWORDS = (
        "看", "图片", "画面", "图里", "图中", "照片", "识别", "描述", "看到",
        "颜色", "形状", "文字", "标签", "有什么", "是什么", "哪个", "在哪",
    )
    DESCRIBE_KEYWORDS = (
        "描述", "介绍一下图", "看看这个", "帮我看看", "图里有什么", "画面里有什么",
    )

    def route(
        self,
        question: str,
        image_data: Optional[bytes] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RouteDecision:
        del history
        normalized = (question or "").strip()
        has_image = bool(image_data)

        if not has_image:
            return RouteDecision(
                route="text_only",
                use_image=False,
                reason="no_image_available",
            )

        if not normalized:
            return RouteDecision(
                route="vision_describe",
                use_image=True,
                reason="empty_question_with_image",
            )

        if any(keyword in normalized for keyword in self.SMALLTALK_KEYWORDS):
            return RouteDecision(
                route="text_only",
                use_image=False,
                reason="smalltalk_keyword",
            )

        if any(keyword in normalized for keyword in self.POINTER_KEYWORDS):
            return RouteDecision(
                route="pointed_object_qa",
                use_image=True,
                reason="pointer_keyword",
            )

        if any(keyword in normalized for keyword in self.DESCRIBE_KEYWORDS):
            return RouteDecision(
                route="vision_describe",
                use_image=True,
                reason="describe_keyword",
            )

        if any(keyword in normalized for keyword in self.VISION_KEYWORDS):
            return RouteDecision(
                route="vision_qa",
                use_image=True,
                reason="vision_keyword",
            )

        return RouteDecision(
            route="vision_contextual",
            use_image=True,
            reason="image_present_default_to_multimodal",
        )
