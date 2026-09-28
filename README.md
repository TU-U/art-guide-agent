# Art Guide Agent

> 让机器人不只“看图回答”，而是能够把语音、视觉、记忆与现场事件组织成可解释、可演进的交互闭环。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20compatible-22314E?logo=ros&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20%26%20debugging-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20prototype-F59E0B)

**Art Guide Agent** 是一个面向展厅、实验室和桌面机器人场景的多模态交互系统。它将语音输入、相机画面、视觉指向、领域知识与会话记忆接入同一条机器人交互链路；同时为后续的 World Model、任务规划和可验证操控预留明确边界。VL-RAG 是其多模态检索与交互技术路线。

| Problem | Solution | Value |
| --- | --- | --- |
| 传统机器人问答无法理解“这个”“刚才那个”，每轮交互彼此割裂；把视频流逐帧丢给 LLM 又慢且不可控。 | 用**事件驱动感知**沉淀结构化事实，以“当前会话上下文 + 按需长期记忆 + 图文问答”组织一次交互。 | 机器人能在视觉异常时降级为语音对话，在多轮交流中保持上下文，并为后续具身任务提供可观测的数据入口。 |

```text
          ┌─────────────── Input ────────────────┐
          │  ASR · Camera · RGB-D · Pointing     │
          └──────────────────┬───────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Robot Interaction  │
                  │  routing · context  │
                  └───────┬───────┬─────┘
                          │       │
          ┌───────────────▼──┐ ┌──▼────────────────┐
          │ VLM / LLM / RAG  │ │ Memory             │
          │ multimodal reply │ │ session · facts    │
          └───────────────┬──┘ └──┬────────────────┘
                          │       │
                    ┌─────▼───────▼─────┐
                    │ TTS · Web Debug   │
                    └───────────────────┘

  Evolution path: perception events → World Model → BT / ROS Actions → MoveIt
```

## Why this project

Many “robot + LLM” demos stop at a serial pipeline: **detect something → turn it into text → ask a model → speak**. This is useful for a one-off explanation, but it fails as soon as interaction becomes continuous:

- a user asks “what is this?” while pointing at a real object;
- a camera frame is stale or unavailable;
- a user corrects the robot and expects it to remember;
- a person enters the scene and the robot should react once, not once per video frame;
- the system later needs to turn “put the apple in the basket” into observable, verifiable steps.

Art Guide Agent treats language models as **semantic reasoning and interaction components**, not as the source of all real-time physical truth. Vision is converted into snapshots or events; conversation is kept within a bounded session; durable information is written deliberately; and the path toward structured world state is explicit.

## What works today

| Capability | What it does | Main implementation |
| --- | --- | --- |
| Multimodal dialogue | Selects text or image+text input according to fresh visual input and returns a grounded reply. | `main.py`, `local_model_processor.py`, `services/vlm_service.py` |
| Robot voice loop | Receives ASR text through ROS, filters echoes/repeats, streams replies, and queues TTS to avoid audio overlap. | `local_model_processor.py`, `services/` |
| Session-aware memory | Keeps bounded in-session history; writes explicitly taught facts immediately; recalls long-term memory only for historical intent; consolidates sessions on idle/end. | `memory/` |
| RAG as an optional tool | Retrieves stable domain knowledge instead of treating it as real-time environmental state. | `rag/` |
| Pointing prototype | Uses RGB-D, hand landmarks, YOLO boxes, and a 3D ray to identify the likely pointed object and emit a debug snapshot/JSON result. | `point/` |
| Person-presence prototype | Uses webcam person tracking to convert continuous video into enter/exit events, SQLite history, and an optional cooldown-protected greeting. | `person_presence/` |
| Debug surfaces | Exposes chat, memory, TTS and perception state through FastAPI pages/APIs for onsite debugging. | `main.py`, `frontend/`, `person_presence/app.py` |

## Architecture

### Runtime interaction path

```text
User speech
  → ASR / HTTP input
  → Robot Brain
      ├─ remove empty, duplicate and TTS-echo inputs
      ├─ record current turn and deterministic user facts
      ├─ load bounded session context
      ├─ recall long-term memory only when intent requires it
      └─ decide whether the latest visual frame is fresh
  → VLM / LLM (+ optional RAG context)
  → sentence streaming and single-worker TTS queue
  → robot audio + Web debugging state
```

### Perception and future embodiment path

```text
Camera / RGB-D
  → detection · tracking · hand/pointing geometry
  → structured observations and events
  → World Model (planned shared state)
  → Behavior Tree / ROS Action (planned task execution)
  → MoveIt / gripper / verification (planned physical control)
```

The repository **does not yet implement** a shared World Model, Behavior Tree executor, MoveIt integration, or autonomous pick-and-place. The existing `point/` and `person_presence/` modules are deliberately isolated prototypes and should become observation adapters when the World Model contract is introduced. See [the robot-agent research draft](docs/robot_agent_research_draft.md) for the target design and milestones.

## Repository map

```text
.
├── main.py                    # FastAPI chat, memory and TTS debugging APIs
├── local_model_processor.py   # ROS robot brain / streaming interaction loop
├── services/                  # ASR, visual input, VLM/LLM and TTS adapters
├── memory/                    # short-term session, facts, insights, events, summaries
├── rag/                       # optional domain knowledge retrieval
├── point/                     # RGB-D hand-pointing and 3D ray prototype
├── person_presence/           # webcam person event and social-greeting prototype
├── prompts/                   # interaction and persona prompts
├── frontend/                  # local debugging UI
├── docs/                      # architecture decisions and research notes
├── tests/                     # prototype unit tests
└── vl_rag_system_v1/          # historical/parallel implementation kept for reference
```

## Quick start

### 1. Prepare a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the web/memory path without ROS or hand-pointing dependencies, use:

```bash
pip install -r requirements-main.txt
```

ROS packages such as `rclpy`, `cv_bridge`, `sensor_msgs` and `message_filters` are provided by your ROS 2 installation, not by pip.

### 2. Configure local credentials

Create a local `.env` file. It is ignored by Git and must never be committed.

```dotenv
# Select qwen_omni or deepseek
VLM_PROVIDER=qwen_omni
DASHSCOPE_API_KEY=replace_with_your_key

# Optional, required for Xunfei TTS
XF_APPID=
XF_API_KEY=
XF_API_SECRET=
```

See [`config.py`](config.py) for all supported environment variables. Keep model weights under `models/` and runtime data under `data/`; both paths are intentionally excluded from version control.

### 3. Run the Web debug backend

```bash
python3 main.py
```

Open `http://127.0.0.1:8765/` (or the configured `BACKEND_PORT`). The backend provides `/chat`, `/memory/*`, and `/api/tts/*` endpoints.

### 4. Run the ROS interaction loop (optional)

After sourcing your ROS 2 environment and bringing up compatible ASR/camera topics:

```bash
./start_all.sh
```

The root startup script launches ASR, visual input and the robot brain. Hardware-specific topics, camera drivers and playback configuration must match the target robot deployment.

### 5. Run the person-presence prototype (optional)

```bash
python3 -m person_presence
```

Open `http://127.0.0.1:8090`. The first YOLO run may download a model weight. This prototype requires a camera and the vision dependencies from `requirements.txt`.

## Development and verification

```bash
# Unit tests for presence state/event logic
python3 -m unittest discover -s tests -v

# Syntax check for core modules
python3 -m compileall -q memory person_presence point main.py local_model_processor.py
```

Before a hardware experiment, validate in this order:

1. Web chat and memory behavior with no camera;
2. fresh/stale image fallback with a recorded image source;
3. perception event stability with replayed frames;
4. ROS topics, TTS playback and timing on the target machine.

## Design principles

- **Freshness before fluency** — a stale frame should not silently answer a current-world question.
- **Events over raw video for reasoning** — language models receive concise, auditable facts rather than a continuous camera stream.
- **Memory is not world state** — memory stores past interactions and learned facts; physical state needs time, confidence and evidence.
- **Graceful degradation** — failure of camera or detection must not break ordinary voice interaction.
- **Verification before autonomy** — physical execution should be confirmed by new observations, not assumed from a generated response.
- **Privacy by design** — person presence tracks temporary camera IDs, not names or face embeddings; identity requires informed consent and a separate design.

## Roadmap

- [x] Voice, visual question answering, optional RAG, memory and TTS interaction loop
- [x] Webcam person-enter/exit event prototype
- [x] RGB-D pointing prototype
- [ ] Shared World Model contract for people, objects, relations and observations
- [ ] RGB-D object tracking, stable object IDs and 3D pose fusion
- [ ] ROS Action interfaces: `Observe`, `FindObject`, `Pick`, `Place`, `Verify`
- [ ] Behavior Tree task executor and simulation baseline
- [ ] Fixed-table “apple into basket” verification MVP with MoveIt

## Documentation

- [Robot Agent architecture discussion](docs/2026-08-10_robot_agent_discussion.md)
- [Robot Agent research and implementation draft](docs/robot_agent_research_draft.md)
- [Person-presence prototype guide](docs/person_presence_prototype.md)

## Contributing

This repository is in an experimental stage. Issues and pull requests are welcome, especially for reproducible perception benchmarks, ROS integration, replayable datasets, World Model contracts, and test coverage. Please avoid committing API credentials, model weights, camera recordings, or generated runtime data.

## License

The source repository currently does not include a license file. Add an explicit `LICENSE` before distributing or accepting third-party contributions under defined terms.
