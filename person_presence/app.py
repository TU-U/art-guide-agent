"""Standalone FastAPI app for the laptop-webcam prototype."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .config import PresenceConfig
from .event_store import EventStore
from .service import WebcamPresenceService
from .social import SocialAgent


def _create_social_agent(config: PresenceConfig, store: EventStore) -> SocialAgent | None:
    if not config.social_enabled:
        return None
    # Reuse the project's provider selection and TTS implementation only when
    # social speech was explicitly enabled by environment configuration.
    from services.tts_service import TTSService
    from services.vlm_service import VLMService

    llm = VLMService()
    tts = TTSService() if config.social_tts_enabled else None
    return SocialAgent(
        store=store,
        text_caller=llm.call_text_model,
        cooldown_seconds=config.social_cooldown_seconds,
        tts_caller=tts.synthesize if tts else None,
    )


def create_app(config: PresenceConfig | None = None) -> FastAPI:
    config = config or PresenceConfig()
    store = EventStore(config.database_path)
    social_agent = _create_social_agent(config, store)
    service = WebcamPresenceService(config, store, social_agent=social_agent)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.start()
        if social_agent:
            social_agent.start()
        yield
        service.stop()
        if social_agent:
            social_agent.stop()
        store.close()

    app = FastAPI(title="Person Presence Prototype", lifespan=lifespan)
    app.state.presence_service = service
    app.state.presence_store = store

    @app.get("/health")
    def health():
        return {
            "status": "error" if service.error else "ok",
            "camera_index": config.camera_index,
            "error": service.error,
        }

    @app.get("/world-model")
    def world_model():
        return service.snapshot().to_dict()

    @app.get("/events")
    def events(limit: int = 100):
        return {"events": store.recent(limit=limit), "today_entry_count": store.today_entry_count()}

    @app.get("/social/messages")
    def social_messages(limit: int = 50):
        return {"enabled": social_agent is not None, "messages": store.recent_social_messages(limit=limit)}

    @app.get("/video.mjpg")
    async def video_feed():
        async def generate():
            while True:
                frame = service.jpeg_frame()
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return """<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>访客感知原型</title><style>body{font:16px sans-serif;max-width:1000px;margin:24px auto}img{max-width:100%;border:1px solid #ccc}pre{background:#f6f6f6;padding:12px;overflow:auto}</style>
<h1>访客感知原型</h1><p>绿色框是追踪到的人；橙色框是进入区域。该原型只保存事件，不做人脸身份识别。</p>
<img src='/video.mjpg'><h2>当前 World Model</h2><pre id='world'>loading…</pre><h2>最近事件</h2><pre id='events'>loading…</pre><h2>机器人主动说的话</h2><pre id='social'>loading…</pre>
<script>async function refresh(){const [w,e,s]=await Promise.all([fetch('/world-model').then(r=>r.json()),fetch('/events').then(r=>r.json()),fetch('/social/messages').then(r=>r.json())]);document.querySelector('#world').textContent=JSON.stringify(w,null,2);document.querySelector('#events').textContent=JSON.stringify(e,null,2);document.querySelector('#social').textContent=JSON.stringify(s,null,2)}refresh();setInterval(refresh,1000)</script></html>"""

    return app


app = create_app()
