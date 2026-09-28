# -*- coding:utf-8 -*-
import json
import logging
from typing import Dict, Any, Optional
from config import Config

logger = logging.getLogger("Resonance")

class ResonanceEngine:
    """
    实现"技心"人设的人格化算法，调节回应的情感质感与美学比重
    """
    def __init__(self):
        logger.info("🎭 [INIT] 正在初始化共鸣引擎...")
        self.persona_protocol = self._load_persona_config()
        self.vibe_history = []
    
    def _load_persona_config(self) -> Dict[str, Any]:
        """加载人设配置"""
        try:
            config_path = Config.PERSONA_CONFIG_PATH
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("⚠️ [PERSONA] 无法加载人设配置，使用默认配置: %s", e)
            return {
                "name": "技心",
                "personality": "友好、专业、富有洞察力",
                "emotion_range": {"min": 0.1, "max": 0.9},
                "response_style": "自然、流畅、有温度",
                "aesthetic_ratio": 0.7
            }
    
    def calculate_vibe(self, text_input: str, user_profile: Optional[Dict[str, Any]] = None) -> float:
        """
        计算情感共鸣分值，结合用户画像调整
        """
        # 简单的情感分析逻辑，实际项目中可以使用更复杂的情感分析模型
        positive_words = ["喜欢", "好", "棒", "赞", "美", "优秀", "精彩"]
        negative_words = ["不喜欢", "差", "糟糕", "讨厌", "丑", "失望"]
        
        positive_count = sum(1 for word in positive_words if word in text_input)
        negative_count = sum(1 for word in negative_words if word in text_input)
        
        base_vibe = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        vibe_score = (base_vibe + 1) / 2  # 归一化到0-1范围
        
        # 结合用户画像调整
        if user_profile:
            user_preference = user_profile.get("emotion_preference", 0.5)
            vibe_score = vibe_score * 0.8 + user_preference * 0.2
        
        # 记录历史
        self.vibe_history.append(vibe_score)
        if len(self.vibe_history) > 100:  # 保持历史记录在合理长度
            self.vibe_history.pop(0)
        
        logger.info("📊 [VIBE] 计算情感共鸣分值: %.2f", vibe_score)
        return vibe_score
    
    def apply_persona_filter(self, raw_response: str, context: Dict[str, Any]) -> str:
        """
        应用人格化滤镜，使响应符合"技心"人设
        """
        # 简单的人格化处理，实际项目中可以使用更复杂的语言模型
        persona_adjustments = [
            ("我不知道", "我目前的知识还在不断学习中"),
            ("你好", "你好！很高兴为你服务"),
            ("谢谢", "不客气，能帮到你我很开心"),
        ]
        
        filtered_response = raw_response
        for original, replacement in persona_adjustments:
            filtered_response = filtered_response.replace(original, replacement)
        
        # 调整情感强度
        vibe_score = context.get("vibe_score", 0.5)
        if vibe_score > 0.7:
            filtered_response += " 😊"
        elif vibe_score < 0.3:
            filtered_response = "我理解你的感受，" + filtered_response
        
        logger.info("🎨 [FILTER] 应用人格化滤镜完成")
        return filtered_response
    
    def get_persona_config(self) -> Dict[str, Any]:
        """
        获取当前人设配置
        """
        return self.persona_protocol
    
    def update_persona_config(self, config: Dict[str, Any]) -> bool:
        """
        更新人设配置参数
        """
        try:
            self.persona_protocol.update(config)
            # 保存到文件
            config_path = Config.PERSONA_CONFIG_PATH
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.persona_protocol, f, ensure_ascii=False, indent=2)
            logger.info("🔧 [PERSONA] 人设配置更新成功")
            return True
        except Exception as e:
            logger.error("❌ [PERSONA] 人设配置更新失败: %s", e)
            return False
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        分析文本情感倾向
        """
        # 简单的情感分析，实际项目中可以使用更复杂的情感分析模型
        positive_words = ["喜欢", "好", "棒", "赞", "美", "优秀", "精彩", "高兴", "开心", "满意"]
        negative_words = ["不喜欢", "差", "糟糕", "讨厌", "丑", "失望", "难过", "伤心", "生气"]
        neutral_words = ["的", "了", "是", "在", "我", "有", "和", "就", "不", "人"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        total_words = len(text.split())
        
        positive_score = positive_count / max(total_words, 1)
        negative_score = negative_count / max(total_words, 1)
        neutral_score = 1 - positive_score - negative_score
        
        return {
            "positive": max(0, min(1, positive_score)),
            "negative": max(0, min(1, negative_score)),
            "neutral": max(0, min(1, neutral_score))
        }
