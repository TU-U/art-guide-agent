import asyncio
import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config
from memory import MemoryHub, UserRegistry
from memory.models import UserGroupProfile
from services.vlm_service import VLMService
from services.tts_service import TTSService

logger = logging.getLogger("Backend")

app = FastAPI()


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


model_inference: Optional[VLMService] = None
try:
    model_inference = VLMService()
except SystemExit as exc:
    logger.warning("[BOOT] VLMService init failed code=%s", exc.code)
except Exception as exc:
    logger.warning("[BOOT] VLMService init error: %s", exc)


memory_hub = MemoryHub()
user_registry = UserRegistry()
tts_service = TTSService()


def _llm_caller_for_memory(messages: List[Dict[str, str]]) -> str:
    if model_inference is None:
        raise RuntimeError("VLMService not initialized, cannot run memory reflection")
    return model_inference.call_text_model(messages)


def _extract_name_from_input(text: str) -> Optional[str]:
    normalized = (text or "").strip()
    if not normalized:
        return None

    patterns = [
        r"^(?:我叫|我是|名字是|你可以叫我|叫我)\s*([^\s，。！？,.!?]{1,12})$",
        r"^([^\s，。！？,.!?]{1,12})$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            candidate = match.group(1).strip("，。！？,.!? ")
            if candidate:
                return candidate
    return None


def _identity_payload(
    answer: str,
    session_id: str,
    user_id: str = "anonymous",
    user_name: str = "",
    identity_status: str = "identified",
) -> Dict[str, Any]:
    return {
        "status": "success",
        "data": {
            "answer": answer,
            "context": "",
            "topic_type": "identity",
            "topic_subject": user_name or "identity",
            "retrieval_query": "",
            "memory_context": "",
            "memory_group": None,
            "memory_insight_count": 0,
            "memory_event_count": 0,
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user_name,
            "identity_status": identity_status,
            "provider": "",
            "model_name": "",
            "confidence": "high",
        },
        "message": "processed",
        "timestamp": datetime.datetime.now().isoformat(),
    }


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _tts_media_type(path: Path) -> str:
    return "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"


def _build_tts_payload(text: str, voice: Optional[str] = None) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"tts_enabled": False}

    output_path = tts_service.synthesize(text.strip(), voice_type=voice)
    if not output_path:
        return {
            "tts_enabled": True,
            "tts_status": "error",
            "tts_error": "TTS synthesis failed",
        }

    audio_file = Path(output_path)
    return {
        "tts_enabled": True,
        "tts_status": "ok",
        "audio_url": f"/api/tts/audio/{audio_file.name}",
        "audio_filename": audio_file.name,
        "audio_media_type": _tts_media_type(audio_file),
    }


def _attach_tts_payload(response: Dict[str, Any], enabled: bool, voice: Optional[str] = None) -> Dict[str, Any]:
    if not enabled:
        return response
    data = response.setdefault("data", {})
    data.update(_build_tts_payload(data.get("answer", ""), voice=voice))
    return response


_session_idle_tokens: Dict[str, str] = {}


async def _schedule_idle_session_end(session_id: str, user_id: str, topic_subject: str) -> None:
    """Consolidate only if this session remains quiet for the configured time."""
    token = uuid4().hex
    _session_idle_tokens[session_id] = token
    await asyncio.sleep(Config.MEMORY_SESSION_IDLE_SECONDS)
    if _session_idle_tokens.get(session_id) != token:
        return
    try:
        committed = await memory_hub.end_session(session_id, user_id, topic_subject)
        if not committed.get("skipped"):
            logger.info("[MEMORY] idle session consolidated session=%s", session_id)
    except Exception as exc:
        logger.warning("[MEMORY] idle session consolidation failed session=%s error=%s", session_id, exc)


if model_inference is not None:
    try:
        memory_hub.attach_extractor(_llm_caller_for_memory)
    except Exception as exc:
        logger.warning("[MEMORY] attach extractor failed: %s", exc)
else:
    logger.info("[MEMORY] extractor unavailable because VLMService is not ready")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = Config.BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
    )


@app.get("/")
def root():
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "backend is running", "status": "ok"}


@app.get("/memory-test")
def memory_test_page():
    page = _frontend_dir / "memory_test.html"
    if page.exists():
        return FileResponse(str(page))
    raise HTTPException(status_code=404, detail="memory_test.html not found")


@app.get("/health")
def health_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}


@app.get("/status")
def status_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}


@app.post("/chat")
async def chat_endpoint(
    image: UploadFile = File(None),
    question: str = Form(None),
    history: str = Form(None),
    session_id: str = Form(None),
    user_id: str = Form("anonymous"),
    user_features: str = Form(None),
    tts_enabled: str = Form("false"),
    tts_voice: str = Form(None),
):
    logger.info("=== incoming /chat request ===")

    image_data = None
    current_question = question if question else "请描述当前画面中的主要物体和场景。"
    current_session_id = (session_id or "").strip() or f"session_{uuid4().hex[:12]}"
    current_user_id = (user_id or "anonymous").strip() or "anonymous"
    parsed_history = None
    parsed_user_features = None
    should_synthesize_tts = _is_truthy(tts_enabled)

    if image:
        try:
            image_data = await image.read()
            logger.info("image loaded: %d bytes", len(image_data))
        except Exception as exc:
            logger.error("read image failed: %s", exc)

    if history:
        try:
            loaded_history = json.loads(history)
            if isinstance(loaded_history, list):
                parsed_history = [
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content", ""),
                    }
                    for item in loaded_history
                    if isinstance(item, dict) and item.get("content")
                ]
                logger.info("parsed request history: %d turns", len(parsed_history))
        except Exception as exc:
            logger.warning("parse history failed, ignore request history: %s", exc)

    if user_features:
        try:
            loaded_features = json.loads(user_features)
            if isinstance(loaded_features, dict):
                parsed_user_features = loaded_features
        except Exception as exc:
            logger.warning("parse user_features failed, ignore request user_features: %s", exc)

    if model_inference is None:
        logger.error("[CHAT] VLMService not initialized")
        return {
            "status": "error",
            "message": "后端 VLMService 尚未初始化。",
        }

    try:
        resolved_user = user_registry.resolve_user(current_user_id, current_session_id)
        if resolved_user is None:
            if image_data:
                current_user_id = current_user_id if current_user_id != "anonymous" else "anonymous"
                current_user_name = ""
            elif user_registry.is_waiting_name(current_session_id):
                candidate_name = _extract_name_from_input(current_question)
                if candidate_name:
                    resolved_user = user_registry.register_name(candidate_name, current_session_id)
                    return _attach_tts_payload(_identity_payload(
                        answer=f"记住了，你叫{resolved_user.name}。现在可以继续问我画面里的物体、位置和特征这些内容。",
                        session_id=current_session_id,
                        user_id=resolved_user.user_id,
                        user_name=resolved_user.name,
                        identity_status="name_registered",
                    ), should_synthesize_tts, tts_voice)
                return _attach_tts_payload(_identity_payload(
                    answer="我还没记住你的名字。先告诉我你叫什么，我再继续陪你聊。",
                    session_id=current_session_id,
                    identity_status="needs_name",
                ), should_synthesize_tts, tts_voice)
            else:
                user_registry.mark_waiting_name(current_session_id)
                return _attach_tts_payload(_identity_payload(
                    answer="第一次见面，先告诉我你叫什么名字吧。我记住之后，后面就能按你的记录继续聊。",
                    session_id=current_session_id,
                    identity_status="needs_name",
                ), should_synthesize_tts, tts_voice)
        else:
            current_user_id = resolved_user.user_id
            current_user_name = resolved_user.name

        memory_hub.record_turn(current_session_id, "user", current_question)
        # Explicitly taught facts are cheap and deterministic; preserve them
        # immediately instead of waiting for the end-of-session LLM pass.
        memory_hub.observe_user_fact(current_session_id, current_question, current_user_id)
        use_long_term_memory = memory_hub.should_recall_long_term(current_question)
        recall_result = memory_hub.recall(
            query=current_question,
            user_id=current_user_id,
            session_id=None,
            top_k=3,
            user_features=parsed_user_features,
            include_history=False,
            include_long_term=use_long_term_memory,
        )
        session_history = memory_hub.short_term.get_raw_history(current_session_id)[
            -Config.MEMORY_SESSION_CONTEXT_MAX_TURNS:
        ]
        # The current user message is passed separately as ``question`` below;
        # do not inject it once in history and once again as the new turn.
        if session_history and session_history[-1].get("role") == "user":
            session_history = session_history[:-1]
        merged_history = session_history or parsed_history

        result = model_inference.generate_response_sync(
            image_data,
            current_question,
            history=merged_history,
            memory_context=recall_result.combined_context,
            memory_profile=_model_to_dict(recall_result.user_group) if recall_result.user_group else None,
        )

        memory_hub.record_turn(current_session_id, "assistant", result["answer"])
        asyncio.create_task(
            _schedule_idle_session_end(
                session_id=current_session_id,
                user_id=current_user_id,
                topic_subject=result.get("topic_subject", ""),
            )
        )

        logger.info("chat completed, answer length=%d", len(result["answer"]))
        response_payload = {
            "status": "success",
            "data": {
                "answer": result["answer"],
                "context": result.get("context", ""),
                "topic_type": result.get("topic_type", ""),
                "topic_subject": result.get("topic_subject", ""),
                "retrieval_query": result.get("retrieval_query", ""),
                "memory_context": result.get("memory_context", ""),
                "memory_group": _model_to_dict(recall_result.user_group) if recall_result.user_group else None,
                "memory_insight_count": len(recall_result.insights),
                "memory_event_count": len(recall_result.events),
                "session_id": current_session_id,
                "user_id": current_user_id,
                "user_name": current_user_name,
                "identity_status": "identified",
                "provider": result.get("provider", ""),
                "model_name": result.get("model_name", ""),
                "route": result.get("route", ""),
                "route_reason": result.get("route_reason", ""),
                "used_image": result.get("used_image", False),
                "confidence": "high",
            },
            "message": "processed",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        return _attach_tts_payload(response_payload, should_synthesize_tts, tts_voice)
    except Exception as exc:
        logger.exception("chat failed: %s", exc)
        return {"status": "error", "message": f"模型处理异常: {str(exc)}"}


class TurnIn(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    role: str = Field("user", description="user / assistant / system")
    content: str


class RecallIn(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    top_k: int = 3
    user_features: Optional[Dict[str, Any]] = None
    history_tail: int = 6


class ReflectIn(BaseModel):
    session_id: str
    user_id: str = "anonymous"
    topic_subject: str = ""


class EndSessionIn(BaseModel):
    session_id: str
    user_id: str = "anonymous"
    topic_subject: str = ""


class GroupMatchIn(BaseModel):
    user_features: Dict[str, Any]


class GroupProfileIn(BaseModel):
    group_id: str
    category_name: str
    aesthetic_pref: str = ""
    communication_pref: str = ""
    typical_tags: List[str] = Field(default_factory=list)
    response_style: Dict[str, str] = Field(default_factory=dict)


class UserNameIn(BaseModel):
    session_id: str
    name: str


class TTSSynthesizeIn(BaseModel):
    text: str
    voice: Optional[str] = None


class TTSConfigIn(BaseModel):
    voice: Optional[str] = None
    speed: Optional[float] = None
    volume: Optional[float] = None
    pitch: Optional[float] = None


@app.get("/memory/stats")
def memory_stats():
    return {"status": "ok", "data": memory_hub.get_stats()}


@app.get("/memory/users")
def memory_list_users():
    users = user_registry.list_users()
    return {"status": "ok", "count": len(users), "data": [_model_to_dict(user) for user in users]}


@app.get("/memory/user/{user_id}")
def memory_get_user(user_id: str):
    user = user_registry.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"user not found: {user_id}")
    return {"status": "ok", "data": _model_to_dict(user)}


@app.post("/memory/user/name")
def memory_register_user_name(payload: UserNameIn):
    record = user_registry.register_name(payload.name.strip(), payload.session_id)
    return {"status": "ok", "data": _model_to_dict(record)}


@app.post("/memory/turn")
def memory_add_turn(payload: TurnIn):
    memory_hub.record_turn(payload.session_id, payload.role, payload.content)
    return {
        "status": "ok",
        "count": memory_hub.short_term.get_history_count(payload.session_id),
    }


@app.get("/memory/sessions")
def memory_list_sessions():
    return {"status": "ok", "data": memory_hub.short_term.list_sessions()}


@app.get("/memory/history/{session_id}")
def memory_history(session_id: str):
    return {
        "status": "ok",
        "session_id": session_id,
        "history": memory_hub.short_term.get_raw_history(session_id),
        "count": memory_hub.short_term.get_history_count(session_id),
    }


@app.delete("/memory/session/{session_id}")
def memory_clear_session(session_id: str):
    ok = memory_hub.clear_session(session_id)
    return {"status": "ok" if ok else "error", "session_id": session_id}


@app.post("/memory/session/end")
async def memory_end_session(payload: EndSessionIn):
    """Consolidate an ended session once, then retain its raw transcript."""
    if memory_hub.extractor is None and memory_hub.event_extractor is None:
        raise HTTPException(status_code=503, detail="memory extractors unavailable")
    try:
        committed = await memory_hub.end_session(
            session_id=payload.session_id,
            user_id=payload.user_id,
            topic_subject=payload.topic_subject,
        )
    except Exception as exc:
        logger.exception("[MEMORY] end session failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    summary = committed.get("summary")
    return {
        "status": "ok",
        "session_id": payload.session_id,
        "skipped": committed.get("skipped", False),
        "summary": _model_to_dict(summary) if summary else None,
        "insights": [_model_to_dict(entry) for entry in committed.get("insights", [])],
        "events": [_model_to_dict(entry) for entry in committed.get("events", [])],
    }


@app.post("/memory/recall")
def memory_recall(payload: RecallIn):
    result = memory_hub.recall(
        query=payload.query,
        user_id=payload.user_id,
        session_id=payload.session_id,
        top_k=payload.top_k,
        user_features=payload.user_features,
        history_tail=payload.history_tail,
    )
    return {"status": "ok", "data": result.to_dict()}


@app.post("/memory/reflect")
async def memory_reflect(payload: ReflectIn):
    if memory_hub.extractor is None and memory_hub.event_extractor is None:
        raise HTTPException(status_code=503, detail="memory extractors unavailable")
    try:
        committed = await memory_hub.reflect_on_conversation(
            session_id=payload.session_id,
            user_id=payload.user_id,
            topic_subject=payload.topic_subject,
        )
    except Exception as exc:
        logger.exception("[MEMORY] reflect failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "status": "ok",
        "insights": [_model_to_dict(entry) for entry in committed.get("insights", [])],
        "events": [_model_to_dict(entry) for entry in committed.get("events", [])],
        "insight_count": len(committed.get("insights", [])),
        "event_count": len(committed.get("events", [])),
    }


@app.get("/memory/insights/{user_id}")
def memory_list_insights(user_id: str, limit: int = 50):
    entries = memory_hub.insights.list_user_insights(user_id, limit=limit)
    return {
        "status": "ok",
        "user_id": user_id,
        "count": len(entries),
        "data": [_model_to_dict(entry) for entry in entries],
    }


@app.delete("/memory/insight/{insight_id}")
def memory_delete_insight(insight_id: str):
    ok = memory_hub.insights.delete_insight(insight_id)
    return {"status": "ok" if ok else "error", "insight_id": insight_id}


@app.get("/memory/events/{user_id}")
def memory_list_events(user_id: str, limit: int = 50):
    entries = memory_hub.events.list_user_events(user_id, limit=limit)
    return {
        "status": "ok",
        "user_id": user_id,
        "count": len(entries),
        "data": [_model_to_dict(entry) for entry in entries],
    }


@app.get("/memory/groups")
def memory_list_groups():
    return {
        "status": "ok",
        "data": [_model_to_dict(p) for p in memory_hub.user_groups.list_all_profiles()],
    }


@app.get("/memory/group/{group_id}")
def memory_get_group(group_id: str):
    profile = memory_hub.user_groups.get_group_config(group_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"group not found: {group_id}")
    return {"status": "ok", "data": _model_to_dict(profile)}


@app.post("/memory/group/match")
def memory_match_group(payload: GroupMatchIn):
    group_id = memory_hub.user_groups.match_group(payload.user_features)
    profile = memory_hub.user_groups.get_group_config(group_id)
    return {
        "status": "ok",
        "group_id": group_id,
        "data": _model_to_dict(profile) if profile else None,
    }


@app.post("/memory/group")
def memory_save_group(payload: GroupProfileIn):
    profile = UserGroupProfile(**_model_to_dict(payload))
    memory_hub.user_groups.save_group_profile(profile)
    return {"status": "ok", "data": _model_to_dict(profile)}


@app.get("/api/tts/status")
def tts_status():
    return {"status": "ok", "data": tts_service.get_status()}


@app.post("/api/tts/config")
def tts_config(payload: TTSConfigIn):
    config = {k: v for k, v in _model_to_dict(payload).items() if v is not None}
    tts_service.set_config(config)
    return {"status": "ok", "data": tts_service.get_status()}


@app.post("/api/tts/synthesize")
def tts_synthesize(payload: TTSSynthesizeIn):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    output_path = tts_service.synthesize(payload.text.strip(), voice_type=payload.voice)
    if not output_path:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

    audio_file = Path(output_path)
    media_type = "audio/wav" if audio_file.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(path=output_path, media_type=media_type, filename=audio_file.name)


@app.get("/api/tts/audio/{filename}")
def tts_audio(filename: str):
    safe_name = Path(filename).name
    audio_file = Config.TTS_OUTPUT_DIR / safe_name
    if not audio_file.exists() or not audio_file.is_file():
        raise HTTPException(status_code=404, detail="audio file not found")
    return FileResponse(path=str(audio_file), media_type=_tts_media_type(audio_file), filename=safe_name)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        log_level="info",
        reload=Config.BACKEND_RELOAD,
        reload_dirs=[
            str(Config.BASE_DIR / "services"),
            str(Config.BASE_DIR / "rag"),
            str(Config.BASE_DIR / "memory"),
            str(Config.BASE_DIR / "agents"),
            str(Config.BASE_DIR / "prompts"),
            str(Config.BASE_DIR),
        ],
        reload_excludes=["data/*", "*.log", "rviz_captured_images/*"],
    )
