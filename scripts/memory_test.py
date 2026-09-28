"""Memory 模块集成测试脚本。

使用临时目录 + mock embedding / mock llm，覆盖：
- ShortTermMemory
- InsightArchive
- InsightExtractor
- UserGroupProfiles
- MemoryHub

运行方式：
    python scripts/memory_test.py
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.insight_archive import InsightArchive
from memory.insight_extractor import InsightExtractor
from memory.memory_hub import MemoryHub
from memory.models import InsightEntry, UserGroupProfile
from memory.short_term_memory import ShortTermMemory
from memory.user_group_profiles import UserGroupProfiles
from services.llm_service import LLMService


def fake_embedding(texts: List[str]) -> List[List[float]]:
    """稳定、可复现的 mock embedding。"""
    vectors: List[List[float]] = []
    for text in texts:
        text = text or ""
        length = float(len(text))
        code_sum = float(sum(ord(ch) for ch in text))
        keyword_score = float(
            sum(
                1
                for keyword in ("灵视", "作者", "理念", "实用", "科技", "展品", "视障")
                if keyword in text
            )
        )
        vectors.append(
            [
                length,
                code_sum % 97,
                (code_sum // 97) % 97,
                keyword_score,
                float(text.count("灵")),
                float(text.count("视")),
                float(text.count("作")),
                float(text.count("设")),
            ]
        )
    return vectors


def fake_llm_caller(_messages: List[dict]) -> str:
    """mock LLM 输出 insight JSON。"""
    return json.dumps(
        [
            {
                "topic": "展品兴趣",
                "content": "用户持续追问《灵视》的作者与设计理念，对该作品兴趣较高。",
                "key_entities": ["灵视", "作者", "设计理念"],
            },
            {
                "topic": "交流偏好",
                "content": "用户希望回答更短、更像正常人说话。",
                "key_entities": ["回答风格", "简洁"],
            },
        ],
        ensure_ascii=False,
    )


def build_prompt_file(base_dir: Path) -> Path:
    prompt_path = base_dir / "insight_extraction_prompt.md"
    prompt_path.write_text(
        "请从以下对话提炼用户见解，以 JSON 数组返回，字段为 topic/content/key_entities。\n"
        "对话如下：\n{conversation}\n当前话题：{topic_subject}\n",
        encoding="utf-8",
    )
    return prompt_path


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_kv(label: str, value) -> None:
    print(f"[{label}] {value}")


def print_json_block(label: str, value) -> None:
    print(f"[{label}]")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def print_assert(message: str) -> None:
    print(f"[assert] {message}")


def model_to_dict(item):
    if item is None:
        return None
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return item


def test_short_term_memory(base_dir: Path) -> None:
    print_header("ShortTermMemory")
    memory = ShortTermMemory(storage_dir=base_dir / "sessions", max_turns=3)
    session_id = "session-alpha"
    print_kv("session_id", session_id)
    print_kv("max_turns", 3)

    memory.add_chat_history(session_id, "user", "灵视的作者是谁？")
    memory.add_chat_history(session_id, "assistant", "作者是林佳维。")
    memory.add_chat_history(session_id, "user", "它有什么设计理念？")
    memory.add_chat_history(session_id, "assistant", "它聚焦视障辅助与自主生活。")

    history = memory.get_raw_history(session_id)
    turns = memory.get_turns(session_id)
    sessions = memory.list_sessions()
    print_json_block("raw_history", history)
    print_kv("turn_count", len(turns))
    print_kv("sessions", sessions)

    assert_true(len(history) == 3, "ShortTermMemory 应截断到 max_turns")
    print_assert("历史条数正确，已截断为最近 3 条")
    assert_true(history[-1]["role"] == "assistant", "应正确返回 role")
    print_assert("最后一条 role 正确")
    assert_true(turns[0].content == "作者是林佳维。", "截断后应保留最近三条")
    print_assert("截断后首条内容正确")
    assert_true(memory.get_history_count(session_id) == 3, "history count 不正确")
    print_assert("history count 正确")
    assert_true(session_id in sessions, "list_sessions 未返回写入的 session")
    print_assert("list_sessions 返回了 session-alpha")

    memory.sync_persistence()
    persisted = (base_dir / "sessions" / f"{session_id}.json").exists()
    assert_true(persisted, "short term 持久化文件未生成")
    print_assert("短时记忆文件已落盘")

    cleared = memory.clear_chat_history(session_id)
    assert_true(cleared, "clear_chat_history 应返回 True")
    assert_true(memory.get_history_count(session_id) == 0, "clear 后应为空")
    print_assert("clear_chat_history 成功，历史已清空")


def test_user_group_profiles(base_dir: Path) -> None:
    print_header("UserGroupProfiles")
    profiles = UserGroupProfiles(storage_path=base_dir / "user_groups.json")
    groups = profiles.list_all_groups()
    print_kv("default_groups", groups)
    assert_true("general_public" in groups, "默认群体应存在")
    print_assert("默认群体 general_public 存在")

    custom = UserGroupProfile(
        group_id="custom_group",
        category_name="定制观众",
        aesthetic_pref="克制",
        communication_pref="直接",
        typical_tags=["定制", "科技"],
        response_style={"detail_level": "medium"},
    )
    profiles.save_group_profile(custom)
    loaded = profiles.get_group_config("custom_group")
    print_json_block("custom_group", model_to_dict(loaded))
    assert_true(loaded is not None and loaded.category_name == "定制观众", "保存/读取群体失败")
    print_assert("自定义群体可保存并读取")

    matched = profiles.match_group({"tags": ["我是科技学生"]})
    print_kv("matched_group", matched)
    assert_true(matched in {"youth_tech", "custom_group"}, "match_group 未正常命中")
    print_assert("关键词匹配正常命中科技相关群体")
    profiles.sync_persistence()
    assert_true((base_dir / "user_groups.json").exists(), "user_groups.json 未落盘")
    print_assert("用户群体配置已落盘")


def test_insight_archive(base_dir: Path) -> None:
    print_header("InsightArchive")
    archive = InsightArchive(
        db_path=base_dir / "insight_db",
        collection_name="test_insights",
        embedding_fn=fake_embedding,
    )
    entry = InsightEntry(
        user_id="u1",
        session_id="s1",
        topic="展品兴趣",
        content="用户多次提到灵视，希望了解作者与设计理念。",
        key_entities=["灵视", "作者", "设计理念"],
    )

    assert_true(archive.commit_insight_sync(entry), "commit_insight_sync 失败")
    print_assert("同步写入 insight 成功")
    async_result = asyncio.run(
        archive.commit_insight(
            InsightEntry(
                user_id="u1",
                session_id="s1",
                topic="交流偏好",
                content="用户偏好更简短直接的回答。",
                key_entities=["简短", "直接"],
            )
        )
    )
    assert_true(async_result, "commit_insight 异步写入失败")
    print_assert("异步写入 insight 成功")

    by_text = archive.search_by_text("灵视 作者", top_k=3, user_id="u1")
    print_json_block("search_by_text", [model_to_dict(item) for item in by_text])
    assert_true(len(by_text) >= 1, "search_by_text 未召回见解")
    print_assert("文本召回至少命中 1 条")

    vector = fake_embedding(["灵视 设计理念"])[0]
    by_vector = archive.search_insights(vector, top_k=3, user_id="u1", session_id="s1")
    print_json_block("search_insights", [model_to_dict(item) for item in by_vector])
    assert_true(len(by_vector) >= 1, "search_insights 未召回见解")
    print_assert("向量召回至少命中 1 条")

    listed = archive.list_user_insights("u1")
    print_kv("list_user_insights_count", len(listed))
    assert_true(len(listed) >= 2, "list_user_insights 数量异常")
    print_assert("用户见解列表数量正确")

    deleted = archive.delete_insight(entry.insight_id)
    assert_true(deleted, "delete_insight 失败")
    stats = archive.get_stats()
    print_json_block("archive_stats", stats)
    assert_true(stats["total_insights"] >= 1, "get_stats 返回异常")
    print_assert("删除后统计仍然可用")


def test_insight_extractor(base_dir: Path) -> None:
    print_header("InsightExtractor(Mock LLM)")
    prompt_path = build_prompt_file(base_dir)
    extractor = InsightExtractor(
        llm_caller=fake_llm_caller,
        prompt_path=prompt_path,
        max_turns_per_extract=6,
    )
    short_term = ShortTermMemory(storage_dir=base_dir / "extractor_sessions", max_turns=6)
    session_id = "session-extractor"
    short_term.add_chat_history(session_id, "user", "灵视的作者是谁？")
    short_term.add_chat_history(session_id, "assistant", "作者是林佳维。")
    short_term.add_chat_history(session_id, "user", "它的设计理念是什么？")
    short_term.add_chat_history(session_id, "assistant", "重点在视障辅助与自主生活。")

    entries = extractor.extract(
        short_term.get_turns(session_id),
        user_id="u-extractor",
        session_id=session_id,
        topic_subject="灵视",
    )
    print_json_block("extracted_insights", [model_to_dict(item) for item in entries])
    assert_true(len(entries) == 2, "extract 应返回两条 insight")
    assert_true(entries[0].topic == "展品兴趣", "extract topic 解析异常")
    assert_true("灵视" in entries[0].key_entities, "extract key_entities 解析异常")
    print_assert("mock LLM 输出已正确解析为 InsightEntry")


def test_memory_hub(base_dir: Path) -> None:
    print_header("MemoryHub(Mock Flow)")
    prompt_path = build_prompt_file(base_dir)
    short_term = ShortTermMemory(storage_dir=base_dir / "hub_sessions", max_turns=8)
    archive = InsightArchive(
        db_path=base_dir / "hub_insight_db",
        collection_name="hub_insights",
        embedding_fn=fake_embedding,
    )
    groups = UserGroupProfiles(storage_path=base_dir / "hub_user_groups.json")
    extractor = InsightExtractor(llm_caller=fake_llm_caller, prompt_path=prompt_path)
    hub = MemoryHub(
        short_term=short_term,
        insight_archive=archive,
        user_groups=groups,
        extractor=extractor,
    )

    session_id = "session-hub"
    user_id = "u-hub"
    hub.record_turn(session_id, "user", "灵视的作者是谁？")
    hub.record_turn(session_id, "assistant", "作者是林佳维。")
    hub.record_turn(session_id, "user", "它的设计理念是什么？")
    hub.record_turn(session_id, "assistant", "重点在视障辅助与自主生活。")

    recalled_before = hub.recall(
        query="灵视的设计理念",
        user_id=user_id,
        session_id=session_id,
        user_features={"tags": ["科技学生", "交互设计"]},
        top_k=3,
    )
    print_json_block("recall_before", recalled_before.to_dict())
    assert_true(len(recalled_before.raw_history) == 4, "recall 应返回短时历史")
    assert_true(recalled_before.user_group is not None, "recall 应返回用户群体")
    assert_true("近期对话" in recalled_before.combined_context, "combined_context 缺少近期对话")
    print_assert("无 insight 时可正常拼装近期对话上下文")

    committed_sync = hub.reflect_on_conversation_sync(
        session_id=session_id,
        user_id=user_id,
        topic_subject="灵视",
    )
    print_json_block("committed_sync", [model_to_dict(item) for item in committed_sync])
    assert_true(len(committed_sync) == 2, "reflect_on_conversation_sync 应提交两条 insight")
    print_assert("同步反思已写入 insight archive")

    recalled_after = hub.recall(
        query="灵视 作者",
        user_id=user_id,
        session_id=session_id,
        top_k=3,
    )
    print_json_block("recall_after", recalled_after.to_dict())
    assert_true(len(recalled_after.insights) >= 1, "reflect 后 recall 应能取回 insight")
    assert_true("过往见解" in recalled_after.combined_context, "combined_context 缺少过往见解")
    print_assert("reflect 后 recall 能召回过往见解")

    async_session = "session-hub-async"
    hub.record_turn(async_session, "user", "我希望回答再短一点。")
    hub.record_turn(async_session, "assistant", "好，我会更简洁。")
    committed_async = asyncio.run(
        hub.reflect_on_conversation(
            session_id=async_session,
            user_id=user_id,
            topic_subject="回答风格",
        )
    )
    print_json_block("committed_async", [model_to_dict(item) for item in committed_async])
    assert_true(len(committed_async) == 2, "reflect_on_conversation 异步链路失败")
    print_assert("异步反思链路正常")

    stats = hub.get_stats()
    print_json_block("hub_stats", stats)
    assert_true(stats["extractor_bound"] is True, "hub stats 未正确反映 extractor 绑定状态")
    hub.sync_persistence()
    assert_true(hub.clear_session(session_id), "hub.clear_session 失败")
    print_assert("hub stats 正常，session 清理成功")


def build_real_llm_caller() -> Optional[callable]:
    if not os.getenv("DEEPSEEK_API_KEY"):
        return None

    llm_service = LLMService()

    def _caller(messages: List[dict]) -> str:
        if llm_service.provider == "deepseek":
            return llm_service._call_deepseek(messages)
        return llm_service._call_ollama(messages, messages[0]["content"])

    return _caller


def test_memory_hub_real_llm(base_dir: Path) -> None:
    print_header("MemoryHub(Real LLM Insight Flow)")
    llm_caller = build_real_llm_caller()
    if llm_caller is None:
        print("[skip] 未配置 DEEPSEEK_API_KEY，跳过真实 LLM 测试")
        return

    short_term = ShortTermMemory(storage_dir=base_dir / "real_sessions", max_turns=8)
    archive = InsightArchive(
        db_path=base_dir / "real_insight_db",
        collection_name="real_llm_insights",
        embedding_fn=fake_embedding,
    )
    groups = UserGroupProfiles(storage_path=base_dir / "real_user_groups.json")
    extractor = InsightExtractor(llm_caller=llm_caller)
    hub = MemoryHub(
        short_term=short_term,
        insight_archive=archive,
        user_groups=groups,
        extractor=extractor,
    )

    session_id = "session-real"
    user_id = "u-real"
    conversation = [
        ("user", "灵视的作者是谁？"),
        ("assistant", "灵视的作者是林佳维。"),
        ("user", "它的设计理念是什么？尽量说短一点。"),
        ("assistant", "它想用智能眼镜和语音交互帮助视障者更独立地生活。"),
        ("user", "我更关心实用性的作品。"),
    ]
    for role, content in conversation:
        hub.record_turn(session_id, role, content)

    print_json_block("conversation", [{"role": role, "content": content} for role, content in conversation])

    committed = hub.reflect_on_conversation_sync(
        session_id=session_id,
        user_id=user_id,
        topic_subject="灵视",
    )
    print_json_block("real_llm_committed_insights", [model_to_dict(item) for item in committed])
    assert_true(len(committed) >= 1, "真实 LLM 未提取出任何 insight")
    print_assert("真实 LLM 已提取并写入至少 1 条 insight")

    recalled = hub.recall(
        query="用户对什么类型的作品更感兴趣？",
        user_id=user_id,
        session_id=session_id,
        top_k=5,
        user_features={"tags": ["科技学生", "实用主义"]},
    )
    print_json_block("real_llm_recall", recalled.to_dict())
    assert_true(len(recalled.insights) >= 1, "真实 LLM 流程 recall 未召回 insight")
    assert_true("过往见解" in recalled.combined_context, "真实 LLM recall 缺少见解上下文")
    print_assert("真实 LLM -> InsightArchive -> MemoryHub.recall 全链路正常")


def main() -> None:
    base_dir = Path(tempfile.mkdtemp(prefix="memory_test_"))
    try:
        print(f"[memory-test] workspace: {base_dir}")

        test_short_term_memory(base_dir)
        print("[pass] ShortTermMemory")

        test_user_group_profiles(base_dir)
        print("[pass] UserGroupProfiles")

        test_insight_archive(base_dir)
        print("[pass] InsightArchive")

        test_insight_extractor(base_dir)
        print("[pass] InsightExtractor")

        test_memory_hub(base_dir)
        print("[pass] MemoryHub")

        test_memory_hub_real_llm(base_dir)
        print("[pass] MemoryHub(Real LLM)")

        print("[ok] memory 模块测试通过")
    finally:
        gc.collect()
        time.sleep(0.5)
        shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
