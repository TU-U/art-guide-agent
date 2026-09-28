"""用户群体画像：JSON 文件持久化 + 关键词匹配。"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import Config

from .models import UserGroupProfile

logger = logging.getLogger("Memory.UserGroups")


DEFAULT_PROFILES: List[UserGroupProfile] = [
    UserGroupProfile(
        group_id="youth_tech",
        category_name="科技青年",
        aesthetic_pref="前卫简约、赛博感、信息密度偏高",
        communication_pref="深度技术讨论，喜欢细节参数与原理",
        typical_tags=["极客", "参数党", "学生", "年轻人", "交互设计", "科技"],
        response_style={"speed": "fast", "detail_level": "high", "tone": "轻盈调侃"},
    ),
    UserGroupProfile(
        group_id="family_kids",
        category_name="亲子家庭",
        aesthetic_pref="温馨柔和、叙事感强",
        communication_pref="浅显易懂、重故事与情感",
        typical_tags=["亲子", "儿童", "家长", "互动体验", "温馨"],
        response_style={"speed": "medium", "detail_level": "medium", "tone": "温柔"},
    ),
    UserGroupProfile(
        group_id="academic_visitor",
        category_name="专业参访",
        aesthetic_pref="严谨克制、学术取向",
        communication_pref="规范术语、有引用和原理说明",
        typical_tags=["老师", "研究", "行业", "评审", "专业"],
        response_style={"speed": "slow", "detail_level": "high", "tone": "沉稳"},
    ),
    UserGroupProfile(
        group_id="general_public",
        category_name="普通观众",
        aesthetic_pref="平和亲切、不晦涩",
        communication_pref="通俗介绍、重亮点与感受",
        typical_tags=["游客", "路人", "散客", "校外"],
        response_style={"speed": "medium", "detail_level": "low", "tone": "温暖"},
    ),
]


class UserGroupProfiles:
    """
    群体画像注册表。支持：
    - 磁盘 JSON 持久化（单文件 `user_groups.json`）
    - 首次初始化时落盘默认群体
    - 基于 typical_tags 的简单关键词匹配
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path or Config.MEMORY_USER_GROUPS_PATH)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._profiles: Dict[str, UserGroupProfile] = {}
        self._load()
        logger.info(
            "👥 [USER_GROUPS] 初始化: path=%s, groups=%d",
            self.storage_path,
            len(self._profiles),
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._profiles = {p.group_id: p for p in DEFAULT_PROFILES}
            self._persist()
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            items = raw.get("groups", [])
            loaded: Dict[str, UserGroupProfile] = {}
            for item in items:
                try:
                    profile = UserGroupProfile(**item)
                    loaded[profile.group_id] = profile
                except Exception as exc:
                    logger.warning("⚠️ [USER_GROUPS] 跳过无效群体: %s", exc)
            if loaded:
                self._profiles = loaded
            else:
                self._profiles = {p.group_id: p for p in DEFAULT_PROFILES}
                self._persist()
        except Exception as exc:
            logger.error("❌ [USER_GROUPS] 加载失败 %s: %s", self.storage_path, exc)
            self._profiles = {p.group_id: p for p in DEFAULT_PROFILES}

    def _persist(self) -> None:
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "groups": [profile.dict() for profile in self._profiles.values()],
        }
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.storage_path)
        except Exception as exc:
            logger.error("❌ [USER_GROUPS] 写入失败: %s", exc)

    def get_group_config(self, group_id: str) -> Optional[UserGroupProfile]:
        with self._lock:
            return self._profiles.get(group_id)

    def save_group_profile(self, profile: UserGroupProfile) -> None:
        with self._lock:
            self._profiles[profile.group_id] = profile
            self._persist()
            logger.info("🔧 [USER_GROUPS] 更新群体: %s", profile.group_id)

    def list_all_groups(self) -> List[str]:
        with self._lock:
            return sorted(self._profiles.keys())

    def list_all_profiles(self) -> List[UserGroupProfile]:
        with self._lock:
            return list(self._profiles.values())

    def match_group(self, user_features: Dict) -> str:
        """
        基于输入特征（tags / keywords / age_group / description 任一字段）做关键词评分。
        若全部不命中，返回 'general_public'。
        """
        tokens: List[str] = []
        for key in ("tags", "keywords", "typical_tags"):
            value = user_features.get(key)
            if isinstance(value, list):
                tokens.extend(str(item) for item in value if item)
        for key in ("description", "age_group", "role"):
            value = user_features.get(key)
            if value:
                tokens.append(str(value))
        joined = " ".join(tokens).lower().strip()

        if not joined:
            return "general_public"

        with self._lock:
            best_id = "general_public"
            best_score = 0
            for group_id, profile in self._profiles.items():
                score = 0
                for tag in profile.typical_tags:
                    if not tag:
                        continue
                    if tag.lower() in joined:
                        score += 2
                if score > best_score:
                    best_score = score
                    best_id = group_id
            return best_id

    def sync_persistence(self) -> None:
        with self._lock:
            self._persist()
