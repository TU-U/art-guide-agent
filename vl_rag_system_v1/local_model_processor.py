# local_model_processor.py
#!/usr/bin/env python3
"""
RobotBrain - Qwen Omni 机器人版
目标：保留现有语音交互链路，同时支持纯文本和图文问答。

当前修改与功能：
1. 文本问答默认走 LLMService；若 point/rviz_captured_images/latest.jpg 存在，则在收到文字后与文字一起走 VLMService。
2. 首句门槛低（遇到逗号即可切，最短 6 字），尽量抢开口。
3. TTS 单线程排队合成（解决并发导致的跳说和堆叠）+ 严格按句序播放。
4. 启动时 TTS 预热（消除冷启动延迟）。
5. 保留原有：双重去重、记忆模块、ROS Topic 接口。
6. 新增机器人运行状态写回，供前端和调试接口查看当前会话、模型和最近一轮输出。
"""
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from memory import MemoryHub
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.vlm_service import VLMService
from config import Config

# =========================
# Logger
# =========================
logger = logging.getLogger("RobotBrain")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(_console_handler)
logger.propagate = False


# =========================
# 切句器：流式 token → 句子
# =========================
class StreamSentencer:
    """
    边接收 token 边切句。
    - 第一句门槛低：遇到 。！？；，, 中的任一就切，最短 6 字
    - 后续句严格：只在 。！？；\n 切，最短 4 字
    """
    FIRST_SEPS = set("。！？；，,.!?;\n")
    LATER_SEPS = set("。！？；!?;\n")
    FIRST_MIN_LEN = 6
    LATER_MIN_LEN = 4

    def __init__(self, on_sentence: Callable[[str], None]):
        self.on_sentence = on_sentence
        self.buffer = ""
        self.first_done = False
        self.lock = threading.Lock()

    def feed(self, token: str):
        if not token:
            return
        with self.lock:
            self.buffer += token
            while True:
                seps = self.FIRST_SEPS if not self.first_done else self.LATER_SEPS
                min_len = self.FIRST_MIN_LEN if not self.first_done else self.LATER_MIN_LEN

                # 找最早出现的分隔符位置
                idx = -1
                for i, ch in enumerate(self.buffer):
                    if ch in seps and i + 1 >= min_len:
                        idx = i
                        break
                if idx < 0:
                    break

                sentence = self.buffer[: idx + 1].strip()
                self.buffer = self.buffer[idx + 1 :]
                if sentence:
                    self.first_done = True
                    self.on_sentence(sentence)

    def flush(self):
        with self.lock:
            tail = self.buffer.strip()
            self.buffer = ""
            if len(tail) >= 1:
                self.on_sentence(tail)


# 主节点
# =========================
def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _now_iso() -> str:
    return datetime.now().isoformat()


class StreamingPopProcessor(Node):
    def __init__(self):
        super().__init__("streaming_pop_processor")
        logger.info("🧠 [INIT] 启动 RobotBrain（Qwen Omni 机器人版）...")

        # --- Services ---
        self.text_model = LLMService()
        self.vlm_model = VLMService()
        self.tts = TTSService()
        self.memory_hub = MemoryHub()
        self.robot_system_prompt = self._load_robot_system_prompt()
        self._attach_memory_extractors()

        # --- Paths ---
        self.vlm_image_path = str(Config.POINT_LATEST_IMAGE_PATH.absolute())
        self.latest_image_path = ""
        self.audio_dir = str(Config.AUDIO_OUT_DIR.absolute())
        Config.AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

        # --- Session & 去重状态 ---
        self.session_id = os.getenv("ROBOT_SESSION_ID", f"robot_{uuid4().hex[:12]}")
        self.user_id = os.getenv("ROBOT_USER_ID", "anonymous")
        self.last_input_text = ""
        self.last_input_time = 0.0
        self.last_answer = ""
        self.last_answer_time = 0.0
        self._busy_lock = threading.Lock()   # 防止上一句还没说完又触发新一轮
        self._state_lock = threading.Lock()
        self._runtime_state: Dict[str, Any] = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": "idle",
            "model_provider": self.vlm_model.provider,
            "model_name": self.vlm_model.model_name,
            "last_user_text": "",
            "last_assistant_text": "",
            "input_source": "",
            "latest_image_path": "",
            "vlm_image_path": "",
            "visual_summary": None,
            "visual_summaries": [],
            "robot_events": [],
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._write_runtime_state()
        self._append_runtime_event("RobotBrain started")

        # --- TTS 顺序播放器 ---
        # 核心修改点：将 max_workers 从 3 改为 1，确保严格的串行合成，解决并发导致的跳说和堆叠问题
        self.tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._play_lock = threading.Lock()
        self._next_play_idx = 0
        self._pending: Dict[int, Optional[str]] = {}
        self._enqueue_idx = 0

        # --- 分段计时埋点（每轮重置）---
        self._turn_timing: Optional[Dict[str, float]] = None
        self._first_submit_done = False
        self._first_synth_done = False
        self._first_publish_done = False

        # --- ROS ---
        self.sub = self.create_subscription(String, "/asr/user_text", self.on_input, 10)
        self.tts_pub = self.create_publisher(String, "/xunfei/tts_play", 10)
        logger.info("✅ /asr/user_text 监听已启动")
        logger.info("✅ /xunfei/tts_play 输出已就绪")
        logger.info("✅ Session: %s", self.session_id)
        logger.info("✅ VLM 指向图片: %s", self.vlm_image_path)

        # --- TTS 预热（消除冷启动）---
        threading.Thread(target=self._warmup_tts, daemon=True).start()

    # =========================
    # 预热
    # =========================
    def _warmup_tts(self):
        try:
            warm_path = os.path.join(self.audio_dir, "_warmup.mp3")
            t0 = time.time()
            self.tts.generate_speech("你好。", warm_path)
            logger.info("🔥 TTS 预热完成: %.2fs", time.time() - t0)
            with suppress(Exception):
                os.remove(warm_path)
        except Exception as e:
            logger.warning("TTS 预热失败: %s", e)

    def _load_robot_system_prompt(self) -> str:
        prompt_path = Config.PROMPTS_DIR / "system_prompt.md"
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("加载 system_prompt.md 失败: %s", exc)
            return "你是艺术导览机器人波普。单次回复最多 100 字，不要提自己是 AI 或模型。"

    # =========================
    # Runtime State
    # =========================
    def _write_runtime_state(self, **updates):
        with self._state_lock:
            self._runtime_state.update(updates)
            self._runtime_state["updated_at"] = _now_iso()
            try:
                Config.ROBOT_BRAIN_STATE_PATH.write_text(
                    json.dumps(self._runtime_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("写入机器人运行状态失败: %s", exc)

    def _append_runtime_event(self, message: str, level: str = "info"):
        with self._state_lock:
            events = list(self._runtime_state.get("robot_events", []) or [])
            events.append(
                {
                    "time": _now_iso(),
                    "level": level,
                    "message": message,
                }
            )
            self._runtime_state["robot_events"] = events[-60:]
            self._runtime_state["updated_at"] = _now_iso()
            try:
                Config.ROBOT_BRAIN_STATE_PATH.write_text(
                    json.dumps(self._runtime_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("写入机器人运行事件失败: %s", exc)

    def mark_offline(self):
        self._append_runtime_event("RobotBrain offline")
        self._write_runtime_state(status="offline")

    # =========================
    # Memory Hook
    # =========================
    def _attach_memory_extractors(self):
        def llm_caller(messages: List[Dict[str, str]]) -> str:
            return self.text_model.call_text_model(messages)

        try:
            self.memory_hub.attach_extractor(llm_caller)
        except Exception as exc:
            logger.warning("[MEMORY] attach failed: %s", exc)

    def _load_latest_image_bytes(self, user_input_time: float) -> Optional[bytes]:
        image_path = Config.POINT_LATEST_IMAGE_PATH
        self.latest_image_path = str(image_path) if image_path.exists() else ""
        if not image_path.exists() or not image_path.is_file():
            with suppress(Exception):
                Config.ROBOT_VLM_IMAGE_PATH.unlink()
            self._write_runtime_state(vlm_image_path="", used_image=False)
            self._append_runtime_event(f"未获取到指向图片: {image_path}", level="warning")
            return None
        try:
            image_mtime = image_path.stat().st_mtime
            if image_mtime < user_input_time and user_input_time - image_mtime > 10:
                image_time_text = datetime.fromtimestamp(image_mtime).isoformat()
                user_time_text = datetime.fromtimestamp(user_input_time).isoformat()
                self._write_runtime_state(
                    latest_image_path=str(image_path),
                    vlm_image_path="",
                    used_image=False,
                )
                with suppress(Exception):
                    Config.ROBOT_VLM_IMAGE_PATH.unlink()
                self._append_runtime_event(
                    "指向图片早于用户输入超过 10 秒，本轮不传图给 VLM: "
                    f"image_time={image_time_text} user_time={user_time_text}",
                    level="warning",
                )
                return None

            image_bytes = image_path.read_bytes()
            Config.ROBOT_VLM_IMAGE_PATH.write_bytes(image_bytes)
            os.utime(Config.ROBOT_VLM_IMAGE_PATH, (image_mtime, image_mtime))
            self._write_runtime_state(
                latest_image_path=str(image_path),
                vlm_image_path=str(Config.ROBOT_VLM_IMAGE_PATH),
            )
            self._append_runtime_event(
                "已获取指向图片并生成 VLM 输入快照: "
                f"{image_path} image_time={datetime.fromtimestamp(image_mtime).isoformat()}"
            )
            return image_bytes
        except Exception as exc:
            logger.warning("读取指向图片失败 %s: %s", image_path, exc)
            self.latest_image_path = ""
            with suppress(Exception):
                Config.ROBOT_VLM_IMAGE_PATH.unlink()
            self._append_runtime_event(f"读取指向图片失败: {image_path} ({exc})", level="error")
            return None

    def _remember_visual_summary(self, image_data: bytes, user_text: str, image_path: str):
        try:
            summary = self.vlm_model.generate_visual_summary_sync(
                image_data=image_data,
                user_text=user_text,
                image_path=image_path,
            )
            committed = self.memory_hub.commit_visual_summary(
                session_id=self.session_id,
                user_id=self.user_id,
                summary=summary,
            )
            with self._state_lock:
                summaries = list(self._runtime_state.get("visual_summaries", []) or [])
            summaries.append(summary)
            self._write_runtime_state(
                visual_summary=summary,
                visual_summaries=summaries[-10:],
            )
            if committed is not None:
                self._append_runtime_event(
                    "视觉摘要已写入长期记忆: %s" % (summary.get("main_content") or "无内容")
                )
            else:
                self._append_runtime_event("视觉摘要生成完成，但未写入长期记忆", level="warning")
        except Exception as exc:
            logger.warning("视觉摘要生成失败: %s", exc)
            self._append_runtime_event(f"视觉摘要生成失败: {exc}", level="error")

    # =========================
    # ROS Callback
    # =========================
    def on_input(self, msg: String):
        try:
            user_text = (msg.data or "").strip()
            logger.debug("📩 [TOPIC] 收到输入: %s", user_text)
            self._append_runtime_event(f"收到 ASR 文字: {user_text}")
            self.process_user_text(user_text, source="topic:/asr/user_text")
        except Exception as e:
            logger.exception("❌ on_input failed: %s", e)

    # =========================
    # 主流程
    # =========================
    def process_user_text(self, user_text: str, source: str = "unknown"):
        if not self._busy_lock.acquire(blocking=False):
            logger.info("⏭️ 上一轮还在生成/播放，丢弃: %s", user_text)
            self._write_runtime_state(
                status="busy",
                last_user_text=(user_text or "").strip(),
                input_source=source,
            )
            return
        try:
            user_text = (user_text or "").strip()
            if not user_text:
                return
            if self._should_ignore_input(user_text):
                logger.info("⏭️ 忽略重复/回声: %s", user_text)
                return

            logger.info("🎧 输入来源=%s 内容=%s", source, user_text)
            self._write_runtime_state(
                status="processing",
                last_user_text=user_text,
                input_source=source,
                last_error="",
            )

            # === 分段计时 t0：入口 ===
            timing: Dict[str, float] = {}
            self._turn_timing = timing
            t0 = time.time()
            timing["t0"] = t0

            # 1) 记忆 & 召回
            self.memory_hub.record_turn(self.session_id, "user", user_text)
            learned_facts = self.memory_hub.observe_user_fact(
                self.session_id,
                user_text,
                user_id=self.user_id,
            )
            if learned_facts:
                logger.info("🧠 [MEMORY] 写入实验室事实: %d 条", len(learned_facts))
            t1 = time.time()
            timing["t1"] = t1
            logger.info("⏱️ [t1] ①记忆写入: %.3fs", t1 - t0)

            recall = self.memory_hub.recall(
                query=user_text,
                user_id=self.user_id,
                session_id=self.session_id,
                top_k=3,
                history_tail=8,
            )
            t2 = time.time()
            timing["t2"] = t2
            logger.info("⏱️ [t2] ②记忆召回: %.3fs", t2 - t1)

            # 2) 选择文本 / 图文输入
            image_data = self._load_latest_image_bytes(user_input_time=t0)
            active_model = self.vlm_model if image_data is not None else self.text_model
            if image_data is not None:
                logger.info("🖼️ 使用指向图片进入 VLM: %s", self.latest_image_path)
                self._append_runtime_event(f"文字和图片已传入 VLM: {self.latest_image_path}")
                threading.Thread(
                    target=self._remember_visual_summary,
                    args=(image_data, user_text, self.latest_image_path),
                    daemon=True,
                ).start()
            else:
                logger.info("📝 当前无可用图片，直接走纯文本链路")
                self._append_runtime_event("未获取到图片，本轮走纯文本 LLM")

            # 3) 调用模型 + 切句 + 入 TTS
            self._reset_play_queue()
            sentencer = StreamSentencer(on_sentence=self._submit_tts)

            try:
                if image_data is not None:
                    result = active_model.generate_response_sync(
                        image_data,
                        user_text,
                        history=(getattr(recall, "raw_history", []) or [])[-8:],
                        memory_context=getattr(recall, "combined_context", ""),
                        memory_profile=_model_to_dict(recall.user_group) if getattr(recall, "user_group", None) else None,
                    )
                else:
                    messages = self._build_messages(user_text, recall)
                    answer = self.text_model.call_text_model(messages)
                    result = {
                        "answer": answer,
                        "provider": self.text_model.provider,
                        "model_name": self.text_model.model_name,
                        "route": "text_only_fast",
                        "used_image": False,
                    }
                full_answer = (result.get("answer") or "").strip() or "当前模型服务不可用，请稍后重试。"
                for token in full_answer:
                    sentencer.feed(token)
                t3 = time.time()
                timing["t3"] = t3
                _t0 = timing.get("t0", t3)
                logger.info(
                    "⏱️ [t3] ③模型完成: %.3fs（距入口 %.3fs） provider=%s route=%s used_image=%s",
                    t3 - t2,
                    t3 - _t0,
                    result.get("provider", active_model.provider),
                    result.get("route", ""),
                    result.get("used_image", bool(image_data)),
                )
                self._write_runtime_state(
                    model_provider=result.get("provider", active_model.provider),
                    model_name=result.get("model_name", active_model.model_name),
                    last_route=result.get("route", ""),
                    used_image=result.get("used_image", bool(image_data)),
                    latest_image_path=self.latest_image_path,
                    vlm_image_path=str(Config.ROBOT_VLM_IMAGE_PATH) if image_data is not None else "",
                )
                self._append_runtime_event(
                    "模型完成: provider=%s route=%s used_image=%s"
                    % (
                        result.get("provider", active_model.provider),
                        result.get("route", ""),
                        result.get("used_image", bool(image_data)),
                    )
                )
            except Exception as e:
                logger.exception("Qwen/VLM 生成失败: %s", e)
                full_answer = "系统处理异常，请稍后重试。"
                self._submit_tts(full_answer)

            sentencer.flush()
            self._mark_stream_done()

            # 4) 收尾：记忆 / 状态
            self.memory_hub.record_turn(self.session_id, "assistant", full_answer)
            self.last_answer = full_answer
            self.last_answer_time = datetime.now().timestamp()
            self._write_runtime_state(
                status="idle",
                last_assistant_text=full_answer,
                input_source=source,
                latest_image_path=self.latest_image_path,
                used_image=bool(image_data),
            )

            threading.Thread(
                target=self._run_memory_reflection,
                args=(user_text,),
                daemon=True,
            ).start()

            # === 部分汇总（t0~t3 同步段；t4~t6 在 TTS 线程异步打）===
            t_now = time.time()
            seg_write = timing.get("t1", t0) - t0
            seg_recall = timing.get("t2", timing.get("t1", t0)) - timing.get("t1", t0)
            seg_first_token = timing.get("t3", t_now) - timing.get("t2", t0) if "t3" in timing else -1
            logger.info(
                "⏱️ [同步段汇总] ①写入=%.3fs ②召回=%.3fs ③模型完成=%s | 距入口=%.3fs",
                seg_write, seg_recall,
                f"{seg_first_token:.3f}s" if seg_first_token >= 0 else "N/A",
                t_now - t0,
            )
            logger.info("✅ 回复完成（端到端 %.2fs）: %s", t_now - t0, full_answer)
            print(f"\n🤖 [BRAIN OUTPUT]: {full_answer}", flush=True)
        except Exception as e:
            logger.exception("❌ process_user_text: %s", e)
            self._write_runtime_state(status="error", last_error=str(e))
        finally:
            self._busy_lock.release()

    # =========================
    # 构建 messages
    # =========================
    def _build_messages(self, user_text: str, recall) -> List[Dict[str, str]]:
        system_parts = [
            self.robot_system_prompt,
        ]
        if getattr(recall, "combined_context", ""):
            system_parts.append(f"参考记忆：{recall.combined_context}")
        if getattr(recall, "user_group", None):
            try:
                system_parts.append(f"用户画像：{_model_to_dict(recall.user_group)}")
            except Exception:
                pass

        messages: List[Dict[str, str]] = [{"role": "system", "content": "\n".join(system_parts)}]

        # 历史
        for turn in (getattr(recall, "raw_history", []) or [])[-8:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_text})
        return messages

    # =========================
    # 去重
    # =========================
    def _should_ignore_input(self, text: str) -> bool:
        now = datetime.now().timestamp()
        normalized = text.strip()

        if normalized == self.last_input_text and now - self.last_input_time < 3.0:
            return True

        self.last_input_text = normalized
        self.last_input_time = now

        if self.last_answer and now - self.last_answer_time < 30.0:
            ans = self.last_answer.strip()
            if normalized and (normalized in ans or ans in normalized):
                return True
        return False

    # =========================
    # TTS 顺序播放管线
    # =========================
    def _reset_play_queue(self):
        with self._play_lock:
            self._next_play_idx = 0
            self._enqueue_idx = 0
            self._pending.clear()
            self._stream_done = False
            self._first_submit_done = False
            self._first_synth_done = False
            self._first_publish_done = False

    def _submit_tts(self, sentence: str):
        """切到一句 → 立即丢线程池合成；合成完按 idx 顺序 publish。"""
        if not sentence:
            return
        with self._play_lock:
            idx = self._enqueue_idx
            self._enqueue_idx += 1
            self._pending[idx] = None  # 占位
            # === t4：首句切出 ===
            if not self._first_submit_done:
                self._first_submit_done = True
                t4 = time.time()
                if self._turn_timing is not None:
                    self._turn_timing["t4"] = t4
                    t3 = self._turn_timing.get("t3", self._turn_timing.get("t2", t4))
                    _t0 = self._turn_timing.get("t0", t4)
                    logger.info("⏱️ [t4] ④切首句: %.3fs（距入口 %.3fs）", t4 - t3, t4 - _t0)
        logger.info("🗣️ 切句 #%d: %s", idx, sentence)
        self.tts_pool.submit(self._synth_and_dispatch, idx, sentence)

    def _synth_and_dispatch(self, idx: int, sentence: str):
        audio_path = None
        try:
            ts = datetime.now().strftime("%H%M%S_%f")
            audio_path = os.path.join(self.audio_dir, f"pop_{idx:03d}_{ts}.mp3")
            t0 = time.time()
            ok = self.tts.generate_speech(sentence, audio_path)
            synth_dur = time.time() - t0
            logger.debug("🔊 TTS #%d 合成 %.2fs ok=%s", idx, synth_dur, ok)
            if not ok:
                audio_path = None
        except Exception as e:
            logger.exception("TTS 合成异常 #%d: %s", idx, e)
            audio_path = None

        with self._play_lock:
            self._pending[idx] = audio_path or ""  # 用空字符串标记失败，保证不卡顺序
            # === t5：首句合成完成 ===
            if idx == 0 and not self._first_synth_done:
                self._first_synth_done = True
                t5 = time.time()
                if self._turn_timing is not None:
                    self._turn_timing["t5"] = t5
                    t4 = self._turn_timing.get("t4", t5)
                    logger.info("⏱️ [t5] ⑤TTS首句合成: %.3fs", t5 - t4)
            self._drain_locked()

    def _mark_stream_done(self):
        with self._play_lock:
            self._stream_done = True
            self._drain_locked()

    def _drain_locked(self):
        """在持锁状态下，把已就绪的下一句按序播放。"""
        while self._next_play_idx in self._pending:
            audio_path = self._pending[self._next_play_idx]
            if audio_path is None:
                break  # 还没合成完，等回调
            self._pending.pop(self._next_play_idx)
            self._next_play_idx += 1
            if audio_path:
                self._publish_play(audio_path)

    def _publish_play(self, audio_path: str):
        msg = String()
        msg.data = json.dumps({"cmd": "append", "file": audio_path})
        self.tts_pub.publish(msg)
        logger.info("▶️ publish play: %s", audio_path)

        # === t6：首句 publish → 机器人开口 ===
        if not self._first_publish_done and self._turn_timing is not None:
            self._first_publish_done = True
            t6 = time.time()
            self._turn_timing["t6"] = t6
            self._log_first_audio_summary()

    def _log_first_audio_summary(self):
        """首音到达时，打印完整分段汇总。超标的段标 ⚠️。"""
        t = self._turn_timing
        if t is None or "t6" not in t:
            return
        t0 = t["t0"]
        t1 = t.get("t1", t0)
        t2 = t.get("t2", t1)
        t3 = t.get("t3", t2)
        t4 = t.get("t4", t3)
        t5 = t.get("t5", t4)
        t6 = t["t6"]

        seg1 = t1 - t0
        seg2 = t2 - t1
        seg3 = t3 - t2
        seg4 = t4 - t3
        seg5 = t5 - t4
        total = t6 - t0

        def flag(val, threshold):
            return "  ⚠️ 超标" if val > threshold else ""

        logger.info("════════════════════════════════════════")
        logger.info("⏱️ ===== 首音延迟分段汇总 =====")
        logger.info("⏱️ ①记忆写入  : %.3fs%s", seg1, flag(seg1, 0.1))
        logger.info("⏱️ ②记忆召回  : %.3fs%s", seg2, flag(seg2, 0.3))
        logger.info("⏱️ ③LLM首token: %.3fs%s", seg3, flag(seg3, 1.0))
        logger.info("⏱️ ④切首句    : %.3fs%s", seg4, flag(seg4, 0.3))
        logger.info("⏱️ ⑤TTS首句合成: %.3fs%s", seg5, flag(seg5, 0.8))
        logger.info("⏱️ ─────────────────────────────────")
        logger.info("⏱️ 端到端开口  : %.3fs（目标 < 2.0s）", total)
        logger.info("════════════════════════════════════════")

    # =========================
    # 记忆反思
    # =========================
    def _run_memory_reflection(self, topic_subject: str):
        with suppress(Exception):
            committed = self.memory_hub.reflect_on_conversation_sync(
                session_id=self.session_id,
                user_id=self.user_id,
                topic_subject=topic_subject,
            )
            if committed.get("insights") or committed.get("events"):
                logger.info(
                    "[MEMORY] committed insights=%d events=%d",
                    len(committed.get("insights", [])),
                    len(committed.get("events", [])),
                )


# =========================
# Main
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = StreamingPopProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.warning("🛑 退出")
    except Exception as e:
        logger.exception("❌ Fatal: %s", e)
    finally:
        with suppress(Exception):
            node.mark_offline()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
