# Architecture Design — VL-RAG-System (实装版)

> 本文档基于对仓库内全部源代码的扫描结果编写，描述的是“当前代码真正实现的架构”，并在末尾与 `docs/architecture_design.md` 中描述的“目标架构”做对比。两者差距较大，阅读代码时请以本文档为准。

---

## 1. 顶层视图

系统有两套完全对等的调用入口，复用同一套服务层：

```
                           +-------------------+
                           |   prompts/*.md    |  <- 提示词模板 (系统/识别/推荐/闲聊)
                           +---------+---------+
                                     |
  ┌────────── Mode A: Web (跨平台) ──┼────────── Mode B: Robot (ROS 2 / Linux) ──────────┐
  │                                  │                                                  │
  │  frontend/index.html             │   services/asr_service.py  (ROS Node)            │
  │        │ multipart POST /chat    │        │ publishes /asr/user_input                │
  │        v                         │        v                                         │
  │  main.py (FastAPI)               │   local_model_processor.py (ROS Node)            │
  │        │                         │        │ subscribes /asr/user_input              │
  │        │                         │        │ reads data/.../latest.jpg               │
  │        └───────────┐             │        └───────┐                                 │
  │                    v             │                v                                 │
  │             services/llm_service.LLMService   (共享实例,业务核心)                    │
  │                    │                                                                 │
  │                    │── services/llm_service._analyze_intent  (关键词路由)            │
  │                    │── services/llm_service._build_retrieval_query                   │
  │                    │── rag/retriever.MuseumRetriever.retrieve()                      │
  │                    │       │                                                         │
  │                    │       ├── GlobalQwenEmbeddingModel (Qwen3-Embedding-0.6B)       │
  │                    │       └── ChromaDB persistent client (data/chroma_db_local_model)│
  │                    │── _call_deepseek  (requests -> DeepSeek Chat)                   │
  │                    └── _call_ollama    (ollama SDK,回退)                              │
  │                    │                                                                 │
  │   HTTP JSON 回包   │   文本流式切句 -> services/tts_service.TTSService.generate_speech│
  │                    │            -> 发布 /xunfei/tts_play (JSON cmd)                  │
  └──────────────────────────────────────────────────────────────────────────────────────┘
```

关键事实：

- 同一进程内只有 **一个 `LLMService` 实例**（`main.py` 或 `local_model_processor.py` 各自构造）。
- `agents/`、`services/agent_manager.py`、`services/resonance_engine.py` **没有被任一入口调用**。目前属于占位代码。
- 配置加载在 `config.py` 模块初始化时副作用完成（`ensure_dirs() + setup_logging()`）。

---

## 2. 模块分层

按“实际运行起来的调用关系”分层，而不是按设想的五层架构：

| 层 | 实装文件 | 是否实际被主链路调用 |
|---|---|---|
| 入口/API | `main.py`, `local_model_processor.py` | 是（二选一） |
| 服务 | `services/llm_service.py`, `services/tts_service.py` | 是 |
| ROS 节点 | `services/asr_service.py`, `services/vision_service.py` | Mode B 下是 |
| RAG | `rag/retriever.py`, `rag/ingest.py`, `rag/ingest_descriptions.py` | 是（只有 retriever；ingest 是离线脚本） |
| 提示词 | `prompts/*.md` | 是（由 LLMService 懒加载） |
| **记忆** | `memory/*.py`（见 §12） | **已实装但未挂主链路**：API 稳定、独立可用、需外部代码显式调用 |
| 占位/未启用 | `agents/*.py`, `services/agent_manager.py`, `services/resonance_engine.py`, `rag/build_vector_db_new.py`, `rag/test_image_analysis.py`, `docs/local_model_processor2.py`, `docs/qwen_vl.py` | 否 |
| 前端 | `frontend/index.html` | 是（单文件 Vue3 CDN） |
| 配置 | `config.py` + `.env` | 是 |

---

## 3. 入口层

### 3.1 `main.py` (FastAPI Web 入口)

- 单文件创建 `FastAPI()`，开放全站 CORS。
- 路由：`GET /`、`GET /health`、`GET /status` 均返回静态 JSON；`POST /chat` 是唯一业务端点。
- `POST /chat` 参数（multipart/form-data）：`image: UploadFile?`、`question: str?`、`history: str?`（JSON 字符串）。
- 调用 `LLMService.generate_response_sync(image_data, question, history)`，把返回值包成 `{status, data:{answer, context, topic_type, topic_subject, retrieval_query, provider, model_name, confidence}, message, timestamp}`。
- 当前实现中 **`image_data` 进入 `LLMService` 后被直接 `del`**（保留接口兼容，不做视觉推理），所以 Mode A 的图片字段实际被忽略。

### 3.2 `local_model_processor.py` (ROS 2 大脑节点)

- `StreamingPopProcessor(Node)`：订阅 `std_msgs/String` 话题 `/asr/user_input`，发布 `/xunfei/tts_play`。
- 图像通道是文件共享：每次收到 ASR 文本，读取 `Config.LATEST_IMAGE_PATH`（由 `vision_service.py` 定时写入）作为图片输入。
- 调用 `LLMService.generate_response_stream` —— 尽管名字叫 stream，实现上是先同步拿到完整回答再逐字符 yield，见 §4.3。
- 流式输出时按中文/英文标点断句，每断一句起一个 `threading.Thread` 调 `TTSService.generate_speech` 写 MP3，然后 publish `{cmd:"append", file:<path>}` 给机器人底层播放器。
- 短时记忆：`self.chat_history` 内存列表，最多保留 10 条（`max_history=10`）。**没有持久化，进程重启即丢失。**

### 3.3 `services/asr_service.py`

- 两套并存的实现：
  1. `ASRService`：基于 `pyaudio` 直采 + 伪百度/腾讯云识别（当前返回硬编码字符串 `"这是云API识别的结果"`，未接入真实 SDK）。
  2. `ASRMonitor(Node)`：订阅讯飞 AIUI 话题 `/xunfei/aiui_msg`，解析 JSON 中的用户文本，转发到 `/asr/user_input` 给大脑节点。
- **实际跑通的是 `ASRMonitor`**，即语音识别走讯飞设备端，本服务只做消息格式转换。
- `SAVE_DIR` (`data/voice_to_text/`) 会保留最多 50 份 wav/txt，由 `cleanup()` 轮转。

### 3.4 `services/vision_service.py`

- 订阅 `sensor_msgs/Image` 话题 `virtual_camera/image_raw`，通过 `cv_bridge` 转成 BGR 并以 95% JPEG 质量写入 `VISION_SAVE_DIR/latest.jpg`。
- 频率由上游摄像头话题决定，本服务是纯 sink，没有去重/采样。
- 仅 Mode B 使用；Windows 下 `rclpy`/`cv_bridge` 不可安装，不要尝试启动它。

---

## 4. 服务层 `services/llm_service.py`

整个业务逻辑几乎都压在 `LLMService` 一个类里。下面按方法拆解。

### 4.1 初始化

- `__init__`：读取 `Config.LLM_PROVIDER / LLM_MODEL_NAME`，构造 `MuseumRetriever()`（这一步就会加载 Chroma 和 Qwen Embedding）。
- `_load_prompt_templates()`：枚举 `Config.PROMPTS_DIR` 下所有 `.md`，读文件字符串存 `self.prompt_templates`。加载的是原文，不做 `format`。

### 4.2 意图分析 `_analyze_intent`

纯关键词规则，返回 `{topic_type, topic_subject, retrieval_top_k}`，四种 topic_type：

1. `smalltalk` —— 问候、感谢、情绪、机器人自我介绍；或会话总结请求。`retrieval_top_k=0`，跳过 RAG。
2. `knowledge` —— 含“作品 / 展品 / 作者 / 技术 / 背景 / 介绍 …”等关键词。`retrieval_top_k = Config.RETRIEVAL_TOP_K`（默认 3）。
3. `knowledge_followup` —— 问句短（≤18 字）且含“它 / 这个 / 那件 / 继续 / 为什么 …”等追问标记，且历史里提取到过 `topic_subject`。检索会拼接上一轮主题。
4. `knowledge_recommendation` —— 含“推荐 / 挑一个 / 随便介绍 …”。`retrieval_top_k=5`，走独立推荐 prompt。

主题抽取 `_extract_topic_subject`：优先从《…》“…”引号里取；否则匹配内置展品别名（`灵视 / 永栖所 / bytebunny / 萨曼鼓 / 萨满鼓 / 虚拟偶像`）；最后兜底截断前 24 字。

**这是一套硬编码的“伪意图识别”**，没有任何模型参与。新增主题必须修改本文件里的列表。

### 4.3 消息组装 `_build_messages`

- 检索问题 `_build_retrieval_query`：如果是短句+追问标记，把 `当前话题对象 / 上一轮问题 / 上一轮回答摘要` 拼成一个多行 query；否则直接用原问题。推荐类会显式写 `筛选偏好：xxx` 头部。
- 知识/推荐分支：调 `MuseumRetriever.retrieve(query, top_k)` 拿到 **已格式化为字符串** 的上下文（不是 chunk 列表），代入 `identify_prompt.md` 或 `recommendation_prompt.md`。
- 闲聊分支：走 `smalltalk_prompt.md`，用最近 4 条历史作为 `history_hint`。
- 最终消息结构：

```python
[
  {"role":"system", "content": prompts/system_prompt.md},
  {"role":"user",   "content": prompt_body + "\n\n" + full_prompt_with_history},
]
```

注意这里 **没有按 OpenAI 风格把历史拆成多条 message**，而是把最近 4 轮历史当成一段文本拼进 user 里。

### 4.4 模型调用

- `_call_deepseek`：HTTP POST 到 `Config.DEEPSEEK_BASE_URL + DEEPSEEK_CHAT_PATH`，`stream=False`，超时 120s，`Authorization: Bearer <DEEPSEEK_API_KEY>`。失败抛异常。
- `_call_ollama`：先 `ollama.chat(model=Config.OLLAMA_MODEL_NAME, messages=...)`，失败回退到 `ollama.generate(system=..., prompt=last_user_content)`。需要本机装 `ollama` 包和拉过模型。
- `_generate_answer` 的失败路径：若 provider 是 DeepSeek 且失败，自动二次尝试 Ollama（`provider="ollama_fallback"`）。两路都挂则返回固定错误语并保持进程不崩。

### 4.5 “流式” API

`generate_stream` / `generate_response_stream` **不是真流式**：它们内部调 `_generate_answer` 拿到完整回答后再逐字符 yield。`local_model_processor.py` 依赖这个 API 做分句 TTS，但首句延迟 ≈ 全回答生成时间。若要真流式必须改造 `_call_deepseek` 走 `stream=true` + SSE 解析。

---

## 5. RAG 子系统 `rag/`

### 5.1 构建阶段

| 脚本 | 输入 | Embedding | 集合名 | 目标库 |
|---|---|---|---|---|
| `rag/ingest.py` | `data/raw_docs/艺术与科技展览数据.xlsx`（多 sheet） | `models/bge-small-zh-v1.5`（CPU） | `museum_local` | `data/chroma_db_local_model/` |
| `rag/ingest_descriptions.py` | `data/raw_docs/industrial_design.txt` | BGE | `industrial_design_assets` | `data/image_analysis_db/` |
| `rag/build_vector_db_new.py` | `D:/OpenResource/.../原始图片/` | OpenCV ORB+HSV+LBP（312 维手工特征） | `art_exhibits` | `D:/OpenResource/.../exhibit_vector_db/` |

`ingest.py` 会把每行 Excel 拼成多字段文本（`作品名称 / 设计作者 / 指导老师 / 类别标签 / 呈现形式 / 作品描述 / 创作时间 / 设计动机 / 灵感来源 / 设计目的 / 设计理念 / 视觉形式语言 / 技术特点 / 预期效果 / 创作历程 / 面临的困难 / 所属展区`），然后批量 `collection.add`。

`build_vector_db_new.py` 路径写死 `D:/OpenResource/...`，与主链路无关，当前视作离线实验脚本。

### 5.2 检索阶段 `rag/retriever.py`

- `GlobalQwenEmbeddingModel`：单例包装 `SentenceTransformer(Qwen3-Embedding-0.6B)`，`encode_queries` 用 `prompt_name="query"`，`encode_documents` 不加 prompt，全部 L2 归一化。
- `MuseumRetriever.__init__`：
  1. 打开 `data/chroma_db_local_model/`；缺失则直接 `sys.exit(1)` 提示跑 ingest。
  2. 载入源集合 `museum_local`（BGE 生成）。
  3. 若本地有 `models/Qwen3-Embedding-0.6B` 就加载，否则切 `fallback_mode=True`。
  4. `_prepare_target_collection`：检查目标集合 `museum_qwen3_embedding` 记录数是否等于源集合，不等则整个 drop 重建，逐批 16 条用 Qwen 重新编码写入。
- 查询路径：
  - 语义：`_semantic_search` → Qwen encode → `collection.query(n_results=top_k)`。
  - 关键词 fallback：`_fallback_search` 走 `_tokenize`（含简单同义词表 / 停用词 / 2~4 字滑窗）→ 在拼接字段上打分。
- `retrieve(query, top_k) -> str`：**最终只返回格式化好的大段字符串**（包含 `作品名称 / 设计作者 / … / 所属展区` 字段），调用方无法拿到结构化 chunk。若上游要做 rerank/融合，需要改 `_format_results` 先把 list 返回再外部格式化。

### 5.3 两个嵌入模型共存

- 写入集合 `museum_local` 使用 BGE；读取集合 `museum_qwen3_embedding` 使用 Qwen3。两者通过 `_rebuild_qwen_collection` 强制同步 —— **不要**手工插入/修改其中一个而不动另一个。
- 如需完全切换到 Qwen：直接删 `museum_local` 后改 `ingest.py` 的 `model_path`，但目前代码要求 `museum_local` 必须存在（作为 ground truth）。

---

## 6. Prompt 层 `prompts/`

四个模板，均为简体中文 Markdown：

| 文件 | 用途 | 消费者 |
|---|---|---|
| `system_prompt.md` | 机器人人设 “波普 / 技心”，风格、字数约束、禁用 Markdown 等 | 所有消息的 system role |
| `identify_prompt.md` | 知识问答主干（拼 `{context}` / `{question}`） | `topic_type in {knowledge, knowledge_followup}` |
| `recommendation_prompt.md` | 展品推荐主干（拼 `{topic_subject}` / `{context}` / `{question}`） | `knowledge_recommendation` |
| `smalltalk_prompt.md` | 闲聊主干（拼 `{topic_subject}` / `{history_hint}` / `{question}`） | `smalltalk` |

模板在进程启动时一次性读入，运行期修改需要重启；`run_dev.ps1` 用 `uvicorn --reload` 自动捡。

---

## 7. 数据与状态

| 类别 | 位置 | 生产者 | 消费者 | 生命周期 |
|---|---|---|---|---|
| 向量库 | `data/chroma_db_local_model/` | `rag/ingest.py` + `MuseumRetriever` | `MuseumRetriever` | 手动构建，服务进程共享 |
| 现场画面 | `rviz_captured_images/latest.jpg` | `vision_service.py` | `local_model_processor.py` | 覆盖写 |
| ASR 音频缓存 | `data/voice_to_text/*.wav/*.txt` | `asr_service.py` | 日志/调试 | LRU 50 份 |
| TTS 输出 | `data/audio_out/pop_*.mp3` | `local_model_processor.py`+`TTSService` | 机器人播放器（通过 ROS 消息指向文件） | 未清理 |
| 人设配置 | `data/persona_setting_profile/persona.json` | `ResonanceEngine.update_persona_config` | `ResonanceEngine`（未接入主链路） | 手动 |
| 日志 | `service.log` | 所有模块（`Config.setup_logging`） | 人工查看 | 追加 |
| 会话历史 | `StreamingPopProcessor.chat_history` 内存 / `main.py` 由前端带入 | 对应入口 | LLMService | 进程内或单次请求 |
| 短时记忆 (按 session) | `data/memory/sessions/{session_id}.json` | `memory.ShortTermMemory` | `MemoryHub.recall` / 调用方 | 持久化，按 `MEMORY_SHORT_TERM_MAX_TURNS` 截断 |
| 长期见解 | `data/memory/insight_db/`（Chroma collection `insight_archive`） | `memory.InsightArchive.commit_insight` | `MemoryHub.recall` / `search_insights` | 持久化，按 `insight_id` upsert |
| 群体画像 | `data/memory/user_groups.json` | `memory.UserGroupProfiles.save_group_profile` | `MemoryHub.recall` / `match_group` | 持久化，首次启动落盘 4 个默认群体 |

没有数据库、没有 Redis、没有消息队列。唯一的跨进程通道是 ROS 话题与文件系统。

---

## 8. 配置 `config.py`

- 所有路径通过 `Path(__file__).parent` 解析，天然跨 Windows/Linux。
- 所有外部凭证走 `.env`（`python-dotenv` 加载），包括：
  - `LLM_PROVIDER`（`deepseek` 默认）、`LLM_MODEL_NAME`、`DEEPSEEK_*`、`OLLAMA_MODEL_NAME`
  - `XF_APPID` / `XF_API_KEY` / `XF_API_SECRET`（讯飞 TTS）
  - `BAIDU_ASR_*` / `TENCENT_ASR_*`（ASR 服务当前未真正使用）
- `RETRIEVAL_TOP_K = 3` 是 `knowledge` / `knowledge_followup` 的默认；`knowledge_recommendation` 在意图层硬写 5。
- **记忆系统路径**：`MEMORY_DIR`、`MEMORY_SESSIONS_DIR`、`MEMORY_INSIGHT_DB`、`MEMORY_USER_GROUPS_PATH`、`MEMORY_SHORT_TERM_MAX_TURNS`（默认 40）、`MEMORY_INSIGHT_COLLECTION`（默认 `insight_archive`）。`ensure_dirs()` 会在 import 时创建前三者。
- **已知缺陷**：`tts_service.py` 里多处使用 `Config.TTS_OUTPUT_DIR`，但 `config.py` 只声明了 `AUDIO_OUT_DIR`。`TTSService.speak/synthesize/play_audio` 三个方法会 AttributeError；只有 `generate_speech(text, output_path)` 能正常工作（`local_model_processor` 走的就是这条）。

---

## 9. 与目标架构 (`docs/architecture_design.md`) 的差距

`docs/architecture_design.md` 是 v2 规划文稿，描述了：

- 五层架构（感官 / 认知 / 代理 / 调度 / 反馈）
- `Memory System`（短时 + 长期 + 用户画像）
- `RetrievalOrchestrator` + 多 `Provider`（Static / Insight / UserGroup）
- `Async Reflection` 内省链
- 多 Agent (`SceneAnalyzer / Dialogue / Action / Intro / Chat / SmallTalk`) 协作
- 完整 REST API 分组 (`/api/rag/*`, `/api/memory/*`, `/api/agent/*` …)

**当前代码里有的：**

- 单一向量库 + 单一检索器。
- 单 LLM 调用 + 关键词路由，无 Agent 协调。
- 极简 `POST /chat` 一个端点，其它 `/api/*` 端点均未实现。
- **记忆系统 `memory/` 已按设计文档实装**：短时记忆（多 session JSON）、长期见解库（ChromaDB）、群体画像、统一 Memory Hub、LLM 异步内省抽取。目前作为独立模块存在，未接入主链路（详见 §12）。
- `ResonanceEngine` 存在但无人调用；`AgentManager` 的默认 Agent 注册因 ImportError 全部静默失败。

**所以实际开发时：**

- 要改行为首先看 `services/llm_service.py` 和 `prompts/`。
- 要改召回首先看 `rag/retriever.py` 和 `rag/ingest.py`。
- 不要去 `agents/` 里改业务逻辑，它还没接上。
- `docs/项目理解与开发底稿.md` 是与本文档一致的实装说明，可作为交叉参考。

---

## 10. 变更建议的落点（编辑速查）

| 想做的改动 | 改这里 |
|---|---|
| 增加新的 topic_type / 意图分支 | `services/llm_service.py::_analyze_intent` + 新建 `prompts/xxx.md` + `_build_messages` 分支 |
| 换检索模型 | 放到 `models/` 并改 `rag/retriever.py::GlobalQwenEmbeddingModel.load` 的模型名；首次启动会自动重建 target collection |
| 增加新字段到知识库 | `rag/ingest.py::fields` 列表 + `rag/retriever.py::_format_results` 的展示模板 |
| 支持真正的流式输出 | `_call_deepseek` 改 `stream=True` + `iter_lines`；相应修改 `generate_stream` 不走 `_generate_answer` |
| 让图片参与推理 | `_generate_answer` 目前 `del image_data`；需要接入多模态模型（DeepSeek 暂不支持，考虑 Qwen-VL / 本地多模态），并修改消息格式 |
| 接入 Agent | 恢复 `services/agent_manager.py` 里 `SceneAnalyzerAgent/DialogueAgent/ActionAgent` 的实现，然后在 `main.py` 或 `local_model_processor.py` 里替换对 `LLMService.generate_response_*` 的直接调用 |
| 修 TTS 的非法路径 | 在 `config.py` 补 `TTS_OUTPUT_DIR = AUDIO_OUT_DIR`，或把 `tts_service.py` 里的 `Config.TTS_OUTPUT_DIR` 全部改成 `Config.AUDIO_OUT_DIR` |
| 把记忆系统挂进主链路 | 在 `main.py` 构造 `MemoryHub`（并通过 `hub.attach_extractor(lambda m: llm._call_deepseek(m))` 绑定抽取器），`/chat` 入口先 `hub.recall(...)` 拿 `combined_context` 拼进 prompt，回答完成后 `hub.record_turn(...)` 两条并 `asyncio.create_task(hub.reflect_on_conversation(...))` 触发内省 |

---

## 11. 部署与运维

- **依赖**：`requirements.txt` 覆盖 Web/RAG/TTS 所需；ROS 2 依赖 (`rclpy`, `cv_bridge`, `sensor_msgs`, `std_msgs`) 需要系统级安装，不走 pip。
- **目录准备**：`Config.ensure_dirs()` 会在 import 时创建 `DATA_DIR / PROMPTS_DIR / AUDIO_OUT_DIR / VISION_SAVE_DIR / ASR_SAVE_DIR / PERSONA_DIR`。
- **启动顺序**（Mode B）：ASR → Vision → Brain（`start_all.sh` 按这个顺序）。Brain 启动时 `MuseumRetriever` 会阻塞式加载 Qwen3 模型（几百 MB），冷启动约 10–30 秒。
- **日志**：统一落 `service.log`，同时转发 stdout。`httpx / ollama / urllib3` 的 DEBUG 已被静音。
- **跨平台注意**：Windows 上只能跑 Mode A；Mode B 依赖 `pyaudio / rclpy / cv_bridge / pygame.mixer`，需要原生 Linux/WSL2 带音频。

---

以上为当前仓库的真实架构。后续若对服务层、RAG 链路或 Agent 做非平凡改造，建议同步更新本文档与 `docs/项目理解与开发底稿.md`，以保持“代码即事实，文档即索引”的一致性。

---

## 12. 记忆系统 `memory/`

实现位置：项目根目录 `memory/`。已按 `docs/architecture_design.md` §3.2.2 的接口契约完成，作为**独立可运行的模块**存在；默认不影响 `main.py` / `local_model_processor.py` 的既有行为，需要外部代码显式构造 `MemoryHub` 并调用。

### 12.1 目录与文件

```
memory/
├── __init__.py              # 包导出
├── models.py                # ChatTurn / InsightEntry / UserGroupProfile (Pydantic)
├── short_term_memory.py     # 短时记忆：多 session JSON 持久化
├── insight_archive.py       # 长期见解库：ChromaDB 持久化
├── insight_extractor.py     # 对话 -> 见解 的 LLM 异步抽取器
├── user_group_profiles.py   # 群体画像 JSON 注册表 + 关键词匹配
└── memory_hub.py            # 对外唯一入口 MemoryHub
prompts/
└── insight_extraction.md    # 内省抽取提示词模板
```

落盘位置（见 `config.py`）：
- 短时记忆：`data/memory/sessions/{session_id}.json`
- 长期见解：`data/memory/insight_db/`（Chroma PersistentClient）
- 群体画像：`data/memory/user_groups.json`

### 12.2 数据模型 `memory/models.py`

| 模型 | 字段 | 说明 |
|---|---|---|
| `ChatTurn` | `role: str`, `content: str`, `timestamp: datetime` | 短时记忆的一条轮次；`to_message()` 输出 OpenAI 风格 dict |
| `UserGroupProfile` | `group_id`, `category_name`, `aesthetic_pref`, `communication_pref`, `typical_tags: List[str]`, `response_style: Dict` | 与架构文档 §3.2.2 一致 |
| `InsightEntry` | `insight_id`(uuid 默认), `user_id`, `session_id?`, `topic`, `content`, `key_entities: List[str]`, `timestamp`, `embedding?: List[float]` | 可 `to_chroma_metadata()` / `from_chroma(doc, meta)` 往返 |

Chroma metadata 仅存标量（字符串），`key_entities` 以 `,` 拼接后存、读回时再拆开。

### 12.3 短时记忆 `ShortTermMemory`

- **多 session 隔离**：每个 `session_id` 一个 JSON 文件（`data/memory/sessions/{session_id}.json`），运行期内有内存缓存 `self._cache[session_id]` 避免重复读盘。
- **写入策略**：追加轮次 → 超过 `max_turns`（默认 `Config.MEMORY_SHORT_TERM_MAX_TURNS=40`）裁前段 → 立即原子落盘（`tmp + os.replace`），防止断电丢数据。
- **并发**：`threading.Lock` 全局一把（I/O 占比低，收益递减后再换细粒度）。
- **对外接口**（与架构文档 §3.2.2 对齐，全部签名显式带 `session_id`）：
  - `add_chat_history(session_id, role, content) -> None`
  - `get_raw_history(session_id) -> List[{role, content}]`
  - `get_turns(session_id) -> List[ChatTurn]`（供 extractor 用）
  - `clear_chat_history(session_id) -> bool`
  - `get_history_count(session_id) -> int`
  - `list_sessions() -> List[str]`
  - `sync_persistence() -> None`（强制刷新所有 dirty session）

### 12.4 长期见解库 `InsightArchive`

- **存储**：`chromadb.PersistentClient(data/memory/insight_db/)` + collection `insight_archive`（collection 名来自 `Config.MEMORY_INSIGHT_COLLECTION`）。
- **embedding**：构造时接受 `embedding_fn: Callable[[List[str]], List[List[float]]]`；默认调 `_default_embedding_fn()`，尝试复用 `rag.retriever.GlobalQwenEmbeddingModel`（Qwen3-Embedding-0.6B）。若本机没有 Qwen3 模型，自动进入 `fallback_mode`：写入时不带 embedding，查询时走关键词 `_keyword_fallback`。
- **写入**：`commit_insight(entry)` 是 `async`（按架构文档要求），内部用 `asyncio.to_thread` 把 Chroma 的 `upsert` 推到线程池。同时暴露 `commit_insight_sync(entry)` 给非 async 代码使用。
- **检索（三种入口）**：
  - `search_insights(query_vector, top_k, user_id=None, session_id=None)` — 架构文档规定的底层接口，接收预计算向量。
  - `search_by_text(query, top_k, user_id=None, session_id=None)` — 便捷接口，内部自动编码后走 `search_insights`；fallback 模式下走关键词扫描。
  - `list_user_insights(user_id, limit=100)` — 列出该用户全部见解，调试/展示用。
- **过滤条件**：`user_id` / `session_id` 组合成 Chroma `where` 字典（`{"$and":[...]}`），底层由 Chroma 做索引过滤，不做二次扫描。
- **删除**：`delete_insight(insight_id)`。

### 12.5 见解抽取 `InsightExtractor`

- **解耦契约**：只依赖一个 `llm_caller: Callable[[List[{role, content}]], str]`。把 LLM 细节留给调用方，避免 memory 模块硬连接 DeepSeek/Ollama。
- **提示词模板**：`prompts/insight_extraction.md`。模板里只有 `{conversation}` 和 `{topic_subject}` 两个占位符。模板要求 LLM **严格输出 JSON 数组**（形如 `[{"topic": "...", "content": "...", "key_entities": [...]}]`），允许返回空数组表示“这段对话不值得长期保留”。
- **解析**：三层兜底——先尝试直接 JSON 解析 → 识别 ` ```json ... ``` ` code fence → 用正则抓第一个 `[...]` 子串再解析。失败时记 warn 并返回空列表，不抛异常。
- **限幅**：单次只看最近 `max_turns_per_extract=10` 轮，防止 prompt 过长。
- **入口**：`extract(turns, user_id, session_id, topic_subject) -> List[InsightEntry]`。产出的 `InsightEntry` **不带 embedding**，embedding 由 `InsightArchive._ensure_embedding` 在写入时计算。

### 12.6 群体画像 `UserGroupProfiles`

- **存储**：单个 JSON 文件 `data/memory/user_groups.json`。首次启动时落盘四个默认群体：
  - `youth_tech`（科技青年）
  - `family_kids`（亲子家庭）
  - `academic_visitor`（专业参访）
  - `general_public`（普通观众 / 兜底）
- **匹配 `match_group(user_features)`**：把 `user_features` 里的 `tags / keywords / typical_tags / description / age_group / role` 字段拼成一个小写文本，对每个群体的 `typical_tags` 做逐项 `in` 匹配计分（命中 +2），取得分最高者；全部为 0 时落回 `general_public`。**不走 embedding**，因为群体维度很少、描述字段短，关键词就足够。
- **CRUD**：`get_group_config` / `save_group_profile` / `list_all_groups` / `list_all_profiles`，所有写操作立即原子落盘。

### 12.7 统一入口 `MemoryHub`

`MemoryHub` 是对外“唯一柜台”，持有上述三个子模块与可选的 `InsightExtractor`。

核心方法：

```python
hub = MemoryHub()                                        # 默认走 Config 路径
hub.attach_extractor(lambda msgs: llm._call_deepseek(msgs))  # 可选：绑定抽取器

# 写入
hub.record_turn(session_id, "user", "《永栖所》的作者是谁？")
hub.record_turn(session_id, "assistant", "作者是王同学。")

# 读取（多路聚合）
result = hub.recall(
    query="永栖所",
    user_id="u1",
    session_id="s1",
    top_k=3,
    user_features={"tags": ["极客", "参数党"]},
    history_tail=6,
)
# result: RecallResult(raw_history, insights, user_group, combined_context)

# 内省（异步）
committed = await hub.reflect_on_conversation(session_id="s1", user_id="u1", topic_subject="永栖所")
hub.reflect_on_conversation_sync(...)                    # 同步版本

# 持久化 / 统计
hub.sync_persistence()
hub.get_stats()
```

**`recall` 的聚合语义**：
- 短时记忆：按 `session_id` 拿最近 `history_tail` 条。
- 见解：调 `InsightArchive.search_by_text(query, top_k, user_id=...)`；若 `user_id == "anonymous"`，不做用户过滤（跨用户通用记忆）。
- 群体画像：有 `user_features` 就 `match_group`；否则落 `general_public`。
- `combined_context` 把三路结果拼成 prompt-ready 的一段纯文本（块标签 `[用户群体] / [过往见解] / [近期对话]`），调用方可以直接贴到 system / user 消息里。

### 12.8 与主链路的集成契约（未接入但稳定）

按当前设计，接入主链路只需要在 `main.py` 的 `/chat` 入口做以下替换：

```python
from memory import MemoryHub

hub = MemoryHub()
hub.attach_extractor(lambda msgs: model_inference._call_deepseek(msgs))

@app.post("/chat")
async def chat_endpoint(image=..., question=..., history=..., session_id: str = Form("default"), user_id: str = Form("anonymous")):
    recall = hub.recall(question, user_id=user_id, session_id=session_id)
    result = model_inference.generate_response_sync(
        image_data, question,
        history=recall.raw_history,        # 替代前端带入的 history
        extra_context=recall.combined_context,  # 需要 LLMService 开一个 kwarg
    )
    hub.record_turn(session_id, "user", question)
    hub.record_turn(session_id, "assistant", result["answer"])
    asyncio.create_task(hub.reflect_on_conversation(session_id, user_id, topic_subject=result.get("topic_subject", "")))
    return ...
```

主链路侧唯一需要的改动是 `LLMService._build_messages` 接受额外上下文并拼到 system prompt 之前（或作为第二个 user turn）。该改动本文件未实施，保留给后续决策。

### 12.9 关键设计取舍

| 取舍 | 选择 | 原因 |
|---|---|---|
| Embedding 模型 | 复用 `rag.retriever.GlobalQwenEmbeddingModel` | 避免在同一进程里加载两份 Qwen3（数百 MB）；同时允许注入任意 embedding_fn 方便替换/测试 |
| 群体画像是否向量化 | 否，用关键词计分 | 群体总量很小且 `typical_tags` 本身就是关键词；上 embedding 会过拟且增加一次模型调用 |
| 内省是否默认开启 | 否 | `attach_extractor` 显式绑定；避免构造 `MemoryHub` 时产生副作用的 LLM 调用 |
| 见解 embedding 何时计算 | 在 `InsightArchive.commit_insight` 内部懒算 | 让 extractor 保持纯文本输出；单元测试不需要真 embedding |
| session 粒度 | 每个 `session_id` 一个 JSON 文件 | 便于手动查看/清理；并发写入锁足够；单会话 >40 轮再考虑拆分 |
| `commit_insight` 为什么 async | 对齐架构文档 §3.2.2 接口定义；内部 `asyncio.to_thread` 实际走同步 Chroma | 不强制调用方 async；同时提供 `commit_insight_sync` 逃生舱 |
| Chroma metadata 类型限制 | `key_entities` 以 `","` 拼接存储 | Chroma metadata 不接受 list 值；`from_chroma` 解析时复原 |
| 并发控制 | 模块级 `threading.Lock` | I/O 占比低，简单可验证；若后续 QPS 上升再按 session 拆锁 |

### 12.10 测试

`memory/` 通过了 6 项 smoke 用例（临时目录 + 伪 embedding + 伪 LLM）：

1. 多 session 写入/读取隔离
2. `InsightArchive.upsert` + 按 `user_id` 过滤检索
3. `UserGroupProfiles.match_group` 三档（命中 youth_tech / family_kids / 无特征落 general_public）
4. `MemoryHub.recall` 三路聚合 + `combined_context` 包含期望关键词
5. `attach_extractor` + `reflect_on_conversation` 异步落地
6. `sync_persistence` 后 `user_groups.json` 落盘存在

当前仓库没有 pytest 工程，smoke 脚本以一次性命令形式存在于 git 历史中（见实装 commit）。后续引入测试框架时，建议迁到 `tests/memory/`。

### 12.11 已知边界

- **Qwen3 首次加载阻塞**：若 `rag.retriever` 尚未在本进程加载过 Qwen3，`InsightArchive.__init__` 会触发一次冷加载（约 10–30 秒）。在已有 `LLMService`/`MuseumRetriever` 的进程中构造 `MemoryHub`，第二次加载命中单例缓存，无感知。
- **Chroma `where` 语法**：不同版本 Chroma 对单条件 vs 多条件的 where 表达要求不同（单条件直接 `{"user_id": "u1"}`，多条件要 `{"$and":[...]}`）。`InsightArchive._build_where` 已做兼容处理，升级 Chroma 时如遇 `InvalidCollectionException` 先看这里。
- **提示词改动不热加载**：`InsightExtractor` 的模板首次 `_load_template` 后缓存到实例；如果改了 `prompts/insight_extraction.md` 要新建 extractor 或手动清空 `_template_cache`。
- **fallback 关键词检索质量有限**：当 Qwen3 不可用时，`search_by_text` 只做大小写无关的空格切词 + `in` 子串匹配。生产环境建议始终保证嵌入模型可用。

