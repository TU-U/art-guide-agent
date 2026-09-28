# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project in one sentence

VL-RAG-System is a museum-guide robot stack that combines ChromaDB + embedding RAG, an LLM (DeepSeek cloud or Ollama local) and a ROS 2 voice/vision loop. The same `LLMService` is driven by either a FastAPI web backend (`main.py`) or a ROS 2 brain node (`local_model_processor.py`).

## Two run modes

The codebase intentionally has two independent entrypoints sharing the same service layer.

**Mode A — PC / Web debug (no ROS required):**
```bash
python main.py                # serves FastAPI on :8000, POST /chat
powershell ./run_dev.ps1      # Windows dev equivalent with --reload
```

**Mode B — Robot, requires ROS 2 Humble + Linux:**
```bash
./start_all.sh                # launches asr_service.py, vision_service.py, local_model_processor.py in separate terminals
tail -f service.log           # unified log sink
```

Mode A bypasses `asr_service.py`, `vision_service.py`, `tts_service.py`, `local_model_processor.py`, and `agents/*`. Only `main.py -> services/llm_service.py -> rag/retriever.py` is exercised. Treat those ROS-only files as optional: they fail to import on Windows (rclpy/cv_bridge/pyaudio).

## Building the knowledge base

The retriever refuses to start if `data/chroma_db_local_model/` is missing. Build with either:
```bash
python rag/ingest.py                 # builds collection "museum_local" from data/raw_docs/艺术与科技展览数据.xlsx using BGE
python rag/ingest_descriptions.py    # separate "industrial_design_assets" collection from industrial_design.txt
```
`MuseumRetriever` on first run auto-rebuilds a parallel `museum_qwen3_embedding` collection by re-encoding every document with `models/Qwen3-Embedding-0.6B` (if present), otherwise it stays on keyword fallback against the BGE collection. Deleting `data/chroma_db_local_model/` forces a full rebuild on next `ingest.py`.

## Configuration

All env-driven config lives in `config.py` (loaded from `.env` in the project root). Notable keys that must be set for Mode A to actually answer:
- `LLM_PROVIDER` (`deepseek` default, `ollama` fallback)
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_CHAT_PATH`
- `OLLAMA_MODEL_NAME` (used when provider=ollama or as auto-fallback)
- Xunfei (`XF_*`) keys are required for `tts_service.py`; Baidu/Tencent keys for `asr_service.py`.

`Config.ensure_dirs()` and `Config.setup_logging()` run at import time — every module that imports `config` writes to `service.log`.

## Architecture summary

Full detail is in `architecture_design.md`. Short version of what actually runs today:

- **Entry layer** — `main.py` (FastAPI) or `local_model_processor.py` (ROS node). Both hold a single `LLMService` instance and pass user text + optional image bytes + chat history.
- **Service layer** — `services/llm_service.py` is the real brain. It:
  1. runs heuristic intent classification over keywords (`smalltalk` / `knowledge` / `knowledge_followup` / `knowledge_recommendation`) in `_analyze_intent`,
  2. builds a RAG-aware retrieval query in `_build_retrieval_query` (rewrites pronouns/follow-ups by prepending the previous topic),
  3. calls `rag.retriever.MuseumRetriever.retrieve()` only for knowledge intents,
  4. loads a template from `prompts/*.md` by intent (`identify_prompt` / `recommendation_prompt` / `smalltalk_prompt`) and composes `[system, user]` messages,
  5. dispatches to DeepSeek via `requests`, or Ollama via the `ollama` SDK, with automatic Ollama fallback on DeepSeek failure.
- **RAG layer** — `rag/retriever.py::MuseumRetriever` is a singleton-ish wrapper that opens ChromaDB, lazily loads `Qwen3-Embedding-0.6B` via `GlobalQwenEmbeddingModel`, mirrors the BGE collection into a Qwen collection on first use, and exposes `retrieve(query, top_k) -> str` (formatted, not list). A keyword `_fallback_search` activates when sentence-transformers / Qwen model aren't available.
- **ROS brain** — `local_model_processor.py` subscribes `/asr/user_input`, calls `LLMService.generate_response_stream`, splits the streamed text on Chinese sentence terminators (`。！？；\n! ? ...`), then per sentence spawns a thread that calls `TTSService.generate_speech` and publishes a JSON `{cmd:"append", file:...}` to `/xunfei/tts_play`. History is capped at `max_history=10`.
- **`agents/` is mostly stubs.** `agents/intro_agent.py`, `chat_agent.py`, `smalltalk_agent.py` inherit `BaseAgent` and return placeholder strings. `services/agent_manager.py` imports `SceneAnalyzerAgent`, `DialogueAgent`, `ActionAgent` that don't exist in the repo — the try/except swallows the ImportError and the manager silently runs with zero agents. **Do not assume AgentManager is wired into the live request path; it isn't.** Same for `services/resonance_engine.py` — defined but not called by `LLMService` or `main.py`.
- **`frontend/index.html`** — single Vue 3 CDN page that POSTs multipart form to `/chat`.

## Gotchas

- **`generate_stream` is fake.** It calls `_generate_answer` synchronously and yields characters from the completed string. The ROS brain's "streaming TTS" is therefore latency-bound by the full LLM round-trip before the first sentence is spoken.
- **Two different embedding models coexist.** `ingest.py` writes with BGE-small-zh-v1.5; `retriever.py` reads with Qwen3-Embedding-0.6B in a separately rebuilt collection. Keep them in sync by letting `_prepare_target_collection` auto-rebuild — don't manually edit `museum_qwen3_embedding`.
- **`tts_service.py` references `Config.TTS_OUTPUT_DIR`** which is not defined in `config.py` (only `AUDIO_OUT_DIR` exists). `TTSService.speak/synthesize/play_audio` will raise `AttributeError`; only `generate_speech(text, output_path)` (the path used by `local_model_processor.py`) is safe.
- **`docs/architecture_design.md` is aspirational.** It describes a five-layer design with Memory/Resonance/multi-Provider RAG that is not implemented in code. `docs/项目理解与开发底稿.md` is the accurate "what's really built" doc. When planning changes, trust the code first.
- **Paths are absolute in `rag/build_vector_db_new.py`** (`D:/OpenResource/...`) and will fail outside the author's machine. This script is not part of the main RAG path (it builds a separate image-feature DB).
- **No test suite, no linter config.** `rag/test_image_analysis.py` is a manual script, not pytest. There is no `pyproject.toml`, `setup.py`, `Makefile`, or CI config.

## Common commands

```bash
# Install (Python 3.9+)
pip install -r requirements.txt

# Interactive RAG query against the knowledge base
python rag/retriever.py --query "永栖所的设计作者是谁" --top_k 3
python rag/retriever.py --stats

# Rebuild BGE collection from Excel source
python rag/ingest.py

# Start web backend
python main.py
```

## When editing

- `prompts/*.md` templates are loaded on `LLMService.__init__` and cached in `self.prompt_templates`. Changes require a process restart (or use `run_dev.ps1` which passes `--reload`).
- Intent keywords and follow-up markers are hard-coded lists inside `_analyze_intent` / `_build_retrieval_query`. New topic types → extend those lists and `prompts/`.
- When adding new Chroma collections, follow the `source_collection -> target_collection` pattern in `MuseumRetriever` — don't bypass the rebuild guard.
