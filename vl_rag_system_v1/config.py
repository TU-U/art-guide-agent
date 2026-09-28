import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


class Config:
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    PROMPTS_DIR = BASE_DIR / "prompts"
    LOG_FILE = BASE_DIR / "service.log"

    PERSONA_DIR = DATA_DIR / "persona_setting_profile"
    PERSONA_JSON_PATH = PERSONA_DIR / "persona.json"
    PERSONA_CONFIG_PATH = PERSONA_JSON_PATH

    CHROMA_PATH = DATA_DIR / "chroma_db_local_model"
    CHROMA_ANONYMIZED_TELEMETRY = os.getenv("CHROMA_ANONYMIZED_TELEMETRY", "false").lower() == "true"

    MEMORY_DIR = DATA_DIR / "memory"
    MEMORY_SESSIONS_DIR = MEMORY_DIR / "sessions"
    MEMORY_INSIGHT_DB = MEMORY_DIR / "insight_db"
    MEMORY_EVENT_DB = MEMORY_DIR / "event_db"
    MEMORY_LAB_FACT_DB = MEMORY_DIR / "lab_fact_db"
    MEMORY_USER_GROUPS_PATH = MEMORY_DIR / "user_groups.json"
    MEMORY_USER_REGISTRY_PATH = MEMORY_DIR / "user_registry.json"
    MEMORY_SHORT_TERM_MAX_TURNS = 40
    MEMORY_INSIGHT_COLLECTION = "insight_archive"
    MEMORY_EVENT_COLLECTION = "event_archive"
    MEMORY_LAB_FACT_COLLECTION = "lab_fact_memory"

    VLM_PROVIDER = os.getenv("VLM_PROVIDER", "qwen_omni").strip().lower()
    QWEN_OMNI_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_OMNI_BASE_URL = os.getenv("QWEN_OMNI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    QWEN_OMNI_MODEL_NAME = os.getenv("QWEN_OMNI_MODEL_NAME", "qwen3-omni-flash")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")
    VLM_MODEL_NAME = QWEN_OMNI_MODEL_NAME if VLM_PROVIDER == "qwen_omni" else DEEPSEEK_MODEL_NAME
    LLM_PROVIDER = VLM_PROVIDER
    LLM_MODEL_NAME = VLM_MODEL_NAME
    EMBEDDING_MODEL_PATH = MODELS_DIR / "bge-small-zh-v1.5"
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_CHAT_PATH = os.getenv("DEEPSEEK_CHAT_PATH", "/chat/completions")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")

    BAIDU_ASR_APP_ID = os.getenv("BAIDU_ASR_APP_ID", "")
    BAIDU_ASR_API_KEY = os.getenv("BAIDU_ASR_API_KEY", "")
    BAIDU_ASR_SECRET_KEY = os.getenv("BAIDU_ASR_SECRET_KEY", "")

    TENCENT_ASR_APP_ID = os.getenv("TENCENT_ASR_APP_ID", "")
    TENCENT_ASR_SECRET_ID = os.getenv("TENCENT_ASR_SECRET_ID", "")
    TENCENT_ASR_SECRET_KEY = os.getenv("TENCENT_ASR_SECRET_KEY", "")

    BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8765"))
    BACKEND_RELOAD = os.getenv("BACKEND_RELOAD", "false").lower() == "true"

    ENABLE_EXHIBIT_RAG = os.getenv("ENABLE_EXHIBIT_RAG", "false").lower() == "true"
    RETRIEVAL_TOP_K = 3

    VISION_SAVE_DIR = BASE_DIR / "rviz_captured_images"
    LATEST_IMAGE_PATH = VISION_SAVE_DIR / "latest.jpg"
    VLM_IMAGE_DIR = DATA_DIR / "vlm_image"
    POINT_DIR = BASE_DIR / "point"
    POINT_IMAGE_DIR = POINT_DIR / "rviz_captured_images"
    POINT_LATEST_IMAGE_PATH = POINT_IMAGE_DIR / "latest.jpg"

    ASR_SAVE_DIR = DATA_DIR / "voice_to_text"
    AUDIO_OUT_DIR = DATA_DIR / "audio_out"
    RUNTIME_DIR = DATA_DIR / "runtime"
    ROBOT_BRAIN_STATE_PATH = RUNTIME_DIR / "robot_brain_state.json"
    ROBOT_VLM_IMAGE_PATH = RUNTIME_DIR / "last_vlm_image.jpg"

    XF_APPID = os.getenv("XF_APPID", "")
    XF_API_KEY = os.getenv("XF_API_KEY", "")
    XF_API_SECRET = os.getenv("XF_API_SECRET", "")

    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "xf").lower()
    TTS_OUTPUT_DIR = AUDIO_OUT_DIR
    TTS_DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "")
    TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")

    QWEN_TTS_MODEL_NAME = os.getenv("QWEN_TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    QWEN_TTS_MODEL_PATH = os.getenv("QWEN_TTS_MODEL_PATH", QWEN_TTS_MODEL_NAME)
    QWEN_TTS_VOICE = os.getenv("QWEN_TTS_VOICE", "Vivian")
    QWEN_TTS_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "zh")
    QWEN_TTS_STYLE_PROMPT = os.getenv("QWEN_TTS_STYLE_PROMPT", "")
    QWEN_TTS_DTYPE = os.getenv("QWEN_TTS_DTYPE", "bfloat16").lower()
    QWEN_TTS_SAMPLE_RATE = int(os.getenv("QWEN_TTS_SAMPLE_RATE", "24000"))
    QWEN_TTS_OUTPUT_FORMAT = os.getenv("QWEN_TTS_OUTPUT_FORMAT", "wav").lower()

    SERVICES_DIR = BASE_DIR / "services"
    ASR_SERVICE_PATH = SERVICES_DIR / "asr_service.py"
    VISION_SERVICE_PATH = SERVICES_DIR / "vision_service.py"
    LLM_SERVICE_PATH = SERVICES_DIR / "llm_service.py"
    TTS_SERVICE_PATH = SERVICES_DIR / "tts_service.py"

    _LOGGING_INITIALIZED = False

    @classmethod
    def setup_logging(cls):
        if cls._LOGGING_INITIALIZED:
            return

        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.LOG_FILE.write_text("", encoding="utf-8")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            force=True,
            handlers=[
                logging.FileHandler(cls.LOG_FILE, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("ollama").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
        logging.getLogger("posthog").setLevel(logging.ERROR)
        logging.getLogger("backoff").setLevel(logging.ERROR)

        cls._LOGGING_INITIALIZED = True
        logging.info("[SYSTEM] 统一日志系统已启动，日志记录于: %s", cls.LOG_FILE)

    @classmethod
    def ensure_dirs(cls):
        for attr in [
            "DATA_DIR",
            "PROMPTS_DIR",
            "AUDIO_OUT_DIR",
            "RUNTIME_DIR",
            "TTS_OUTPUT_DIR",
            "VISION_SAVE_DIR",
            "VLM_IMAGE_DIR",
            "POINT_IMAGE_DIR",
            "ASR_SAVE_DIR",
            "PERSONA_DIR",
            "MEMORY_DIR",
            "MEMORY_SESSIONS_DIR",
            "MEMORY_INSIGHT_DB",
            "MEMORY_EVENT_DB",
            "MEMORY_LAB_FACT_DB",
        ]:
            path = getattr(cls, attr)
            path.mkdir(parents=True, exist_ok=True)


Config.ensure_dirs()
Config.setup_logging()
