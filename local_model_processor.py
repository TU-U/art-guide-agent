# local_model_processor.py
#!/usr/bin/env python3
"""
RobotBrain - DeepSeek 流式版
目标：用户说完 → 机器人开口 < 2 秒

核心改造：
1. LLM 改用 DeepSeek 流式 API（SSE），边收 token 边切句
2. 首句门槛低（遇到逗号即可切，最短 6 字），抢开口
3. TTS 单线程排队合成（解决并发超限导致跳说）+ 严格按句序播放
4. 启动时 TTS 预热（消除冷启动延迟）
5. 保留原有：双重去重、记忆模块、ROS Topic 接口
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

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from memory import MemoryHub
from services.llm_service import LLMService
from services.tts_service import TTSService
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


# =========================
# DeepSeek 流式客户端（仅用于本节点）
# =========================
class DeepSeekStreamClient:
    """直接调 DeepSeek /v1/chat/completions 的 stream=True 模式。"""

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "30"))
        if not self.api_key:
            logger.warning("⚠️ DEEPSEEK_API_KEY 未配置")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        on_token: Callable[[str], None],
        temperature: float = 0.7,
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> str:
        """阻塞调用，但 token 通过 on_token 实时回调。返回完整答案。"""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }

        full_answer = ""
        t0 = time.time()
        first_token_time: Optional[float] = None

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    data = raw_line[6:].strip()
                else:
                    data = raw_line.strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content") or ""
                if not token:
                    continue
                if first_token_time is None:
                    first_token_time = time.time()
                    logger.info("⚡ LLM 首 token 延迟: %.2fs", first_token_time - t0)
                    if on_first_token:
                        try:
                            on_first_token()
                        except Exception:
                            logger.debug("on_first_token 回调异常", exc_info=True)
                full_answer += token
                on_token(token)

        logger.info("⏱️ LLM 总耗时: %.2fs（共 %d 字）", time.time() - t0, len(full_answer))
        return full_answer


# =========================
# 主节点
# =========================
def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class StreamingPopProcessor(Node):
    def __init__(self):
        super().__init__("streaming_pop_processor")
        logger.info("🧠 [INIT] 启动 RobotBrain（DeepSeek 流式版）...")

        # --- Services ---
        self.model = LLMService()            # 仍保留，用于记忆抽取
        self.tts = TTSService()
        self.memory_hub = MemoryHub()
        self._attach_memory_extractors()
        self.ds_client = DeepSeekStreamClient()

        # --- Paths ---
        self.latest_image_path = str(Config.LATEST_IMAGE_PATH)
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

    # =========================
    # Memory Hook
    # =========================
    def _attach_memory_extractors(self):
        def llm_caller(messages: List[Dict[str, str]]) -> str:
            if self.model.provider == "deepseek":
                return self.model._call_deepseek(messages)
            if self.model.provider == "ollama":
                system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")
                return self.model._call_ollama(messages, system_prompt)
            raise RuntimeError("unsupported provider")

        try:
            self.memory_hub.attach_extractor(llm_caller)
        except Exception as exc:
            logger.warning("[MEMORY] attach failed: %s", exc)

    # =========================
    # ROS Callback
    # =========================
    def on_input(self, msg: String):
        try:
            user_text = (msg.data or "").strip()
            logger.debug("📩 [TOPIC] 收到输入: %s", user_text)
            self.process_user_text(user_text, source="topic:/asr/user_text")
        except Exception as e:
            logger.exception("❌ on_input failed: %s", e)

    # =========================
    # 主流程
    # =========================
    def process_user_text(self, user_text: str, source: str = "unknown"):
        if not self._busy_lock.acquire(blocking=False):
            logger.info("⏭️ 上一轮还在生成/播放，丢弃: %s", user_text)
            return
        try:
            user_text = (user_text or "").strip()
            if not user_text:
                return
            if self._should_ignore_input(user_text):
                logger.info("⏭️ 忽略重复/回声: %s", user_text)
                return

            logger.info("🎧 输入来源=%s 内容=%s", source, user_text)

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
                session_id=None,
                top_k=3,
                include_history=False,
                include_long_term=self.memory_hub.should_recall_long_term(user_text),
            )
            recall.raw_history = self.memory_hub.short_term.get_raw_history(self.session_id)[
                -Config.MEMORY_SESSION_CONTEXT_MAX_TURNS:
            ]
            if recall.raw_history and recall.raw_history[-1].get("role") == "user":
                recall.raw_history = recall.raw_history[:-1]
            t2 = time.time()
            timing["t2"] = t2
            logger.info("⏱️ [t2] ②记忆召回: %.3fs", t2 - t1)

            # 2) 组装 messages
            messages = self._build_messages(user_text, recall)

            # 3) 流式调用 + 切句 + 入 TTS
            self._reset_play_queue()
            sentencer = StreamSentencer(on_sentence=self._submit_tts)

            def on_token(tok: str):
                sentencer.feed(tok)

            def on_first_token():
                t3 = time.time()
                timing["t3"] = t3
                _t0 = timing.get("t0", t3)
                logger.info("⏱️ [t3] ③LLM 首 token: %.3fs（距入口 %.3fs）", t3 - t2, t3 - _t0)

            try:
                full_answer = self.ds_client.stream_chat(
                    messages, on_token=on_token, on_first_token=on_first_token
                )
            except requests.exceptions.ReadTimeout:
                logger.warning("DeepSeek API 响应超时。")
                full_answer = "抱歉，我的网络刚刚开了个小差，能再和我说一遍吗？"
                self._submit_tts(full_answer)
            except Exception as e:
                logger.exception("DeepSeek 流式失败: %s", e)
                full_answer = "系统处理异常，请稍后重试。"
                self._submit_tts(full_answer)

            sentencer.flush()  # 把尾巴送出去
            self._mark_stream_done()

            # 4) 收尾：记忆 / 状态
            self.memory_hub.record_turn(self.session_id, "assistant", full_answer)
            self.last_answer = full_answer
            self.last_answer_time = datetime.now().timestamp()

            # === 部分汇总（t0~t3 同步段；t4~t6 在 TTS 线程异步打）===
            t_now = time.time()
            seg_write = timing.get("t1", t0) - t0
            seg_recall = timing.get("t2", timing.get("t1", t0)) - timing.get("t1", t0)
            seg_first_token = timing.get("t3", t_now) - timing.get("t2", t0) if "t3" in timing else -1
            logger.info(
                "⏱️ [同步段汇总] ①写入=%.3fs ②召回=%.3fs ③首token=%s | 距入口=%.3fs",
                seg_write, seg_recall,
                f"{seg_first_token:.3f}s" if seg_first_token >= 0 else "N/A",
                t_now - t0,
            )
            logger.info("✅ 回复完成（端到端 %.2fs）: %s", t_now - t0, full_answer)
            print(f"\n🤖 [BRAIN OUTPUT]: {full_answer}", flush=True)
        except Exception as e:
            logger.exception("❌ process_user_text: %s", e)
        finally:
            self._busy_lock.release()

    # =========================
    # 构建 messages
    # =========================
    def _build_messages(self, user_text: str, recall) -> List[Dict[str, str]]:
        system_parts = [
            "你是一个友好、简洁的对话机器人。",
            # 修改点：将限制改成了 300 字，适应更详尽的回答
            "回答要口语化、自然，控制在 300 字以内，方便语音播报。",
            "不要使用 Markdown 符号、表情符号、列表项。",
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
