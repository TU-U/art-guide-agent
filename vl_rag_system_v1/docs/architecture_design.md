# VL-RAG-System 架构设计文档

本文档定义了“技心”多模态交互机器人的整体技术方案、模块规范及数据流转逻辑。

---

## 1. 核心架构图 (Core Architecture Diagrams)

### 1.1 系统全局架构
![系统架构图](架构图.png)

### 1.2 逻辑调用流程
![模块调用流程图](模块调用流程图.png)
![内部模块调用细节](内部模块调用细节图.png)

---

## 2. 架构分层 (Architectural Layers)

系统采用高度解耦的五层架构设计，通过同步响应链 (Sync Chain) 与异步内省链 (Async Reflection) 实现智慧闭环：

1.  **感官层 (Perception Layer)**: `ASR` (听觉) 与 `Vision` (视觉) 服务，负责原始环境信号的语义抽象。
2.  **认知层 (Cognitive Layer)**: `LLM` (语言大脑)、**`Memory System` (记忆系统)** 与 **`Retrieval Engine` (检索引擎)**。
3.  **代理层 (Agent Layer)**: **核心业务大脑**，执行认知循环并调用记忆系统获取与存储所有信息。
4.  **调度层 (Orchestration Layer)**: `System Orchestrator` (总控)，负责跨模块的状态机维护、路由管理及任务编排。
5.  **反馈层 (Execution Layer)**: `TTS`、`Body Controller` (肢体控制) 与前端展示组件，完成交互闭环。

---

## 3. 详细模块设计 (Detailed Module Design)

---

### 3.1 系统总控与 API 层 (Orchestration & API)
- **文件**: `local_model_processor.py`, `api/routers/`
- **职责**: 作为“中枢神经”，协调感知输入、智能决策与执行层调度。维护全局状态机，负责会话启动与分发。
- **持有的子系统**: Language, Retrieval, Memory, Resonance, Vision, Hearing, TTS.
- **关键接口**:
    - `start_orchestration()`: 系统全局启动与节点健康检查。
    - `dispatch(query: str, image: bytes) -> Stream`: 分发原始输入并返回流式响应。
    - `identify_intent(query: str) -> Intent`: 语义意图识别，决定后续路由。
    - `process_input(input_data)`: 原始输入预处理与多模态对齐。
    - `reset_session()`: 重置当前会话上下文与 Agent 状态。
    - `handle_async_reflection(chat_history: list) -> None`: 触发后台内省任务。

### 3.2 认知引擎模块 (Cognitive Engines)
#### 3.2.1 检索引擎 (Retrieval Engine)
- **定位**: **检索为记忆服务**。它不直接对接代理层，仅作为记忆系统的基础设施。
- **职责**: 负责知识库的构建（离线）与查询（在线），实现从数据入库到多路 RAG 检索的完整流程。

**1. 数据构建层 (Data Ingestion)**:
- **`rag/ingest.py`**: 从结构化数据构建文本向量数据库（ChromaDB）。
- **`rag/build_vector_db_new.py`**: 构建图像特征向量数据库（多模态对齐）。

**2. 检索引擎层 (Search Infrastructure)**:
- **设计模式**: **提供者模式 (Provider Pattern)** —— 并行调度多仓库检索并输出原子结果。
- **引擎组件 (Engine Components)**:
    - **`StaticRAGProvider`**: 调用由 `ingest.py` 构建的 80+ 展品专业库。
- **数据结构定义 (Static Entity)**:
```python
class Exhibit(BaseModel):
    """静态展品实体模型（根据需求补充调整）"""
    exhibit_id: str               # 展品 ID (如: "robot_001")
    name: str                     # 展品名称 (如: "双子星机器人")
    location: str                 # 物理位置 (如: "一楼 A 厅 102 号")
    era_period: Optional[str]     # 历史年代或所属技术时代
    category: str                 # 类别 (如: "服务类机器人")
    description: str              # 核心背景介绍文本 (用于生成检索内容)
    technical_params: Dict        # 技术参数 (如: {"height": "170cm", "weight": "60kg"})
    related_tags: List[str]       # 语义标签 (如: ["双足", "空间导览"])
```
    - **`InsightArchiveProvider`**: (见解检索) **仅检索 LLM 抽取的结论/见解**（向量库 B）。它不搜原话，而是搜机器人对用户的“认知总结”。
    - **`UserGroupProvider`**: 匹配并提取预设的群体配置（风格库）。
- **核心接口 (针对内部调用)**:
    - **`vector_search(query: str, collection: str)`**: 针对特定集合的原子召回。
    - **`rerank(query, candidates)`**: (关键环节) 对多路回传的候选结果进行深度语义重排序。
    - **`fuse_knowledge(chunks: list) -> str`**: 将多路信息打碎、去重并融合成一段文字。
    - **`get_stats()`**: 统计知识库中有多少展品、多少条见解等数据。
    - **`rebuild_index()`**: 索引增量维护。

- **【通俗易懂】架构是怎么升级的？**:
    - **从“手动脚本”变成“自动流水线”**: 以前我们需要手动跑 `ingest.py` 来存数据。现在这些脚本被我们变成了系统的“数据加工厂”（离线部分）。而机器人聊天时，我们会用一个“智能调度员”（`RetrievalOrchestrator`）来自动并发调用所有数据。
    - **从“一个库”变成“三路并发”**: 系统不再只搜展品库，而是同时开辟了三条检索专线：
        1.  **静态展品路 (Static)**: 调用原来的 RAG，搜展品背景。
        2.  **对话见解路 (Insight)**: 调用见解库，搜 **LLM 提炼出的结论性见解**（比如“用户喜欢喝浓茶”），而不是搜聊天的原话。
        3.  **用户群体路 (Profiles)**: 调用画像库，匹配该群体的聊天风格。
    - **每个库都有自己的“搜索专员”**: 检索引擎为每一类记忆都量身定制了 `search` 接口和处理类（Provider）。这意味着不管是查历史、找配置还是翻资料，都有专门的通道，解决了以前 RAG 逻辑大杂烩的问题。

- **接口定义示例 (Python)**:
```python
class MemoryProvider:
    """记忆提供者基类"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        raise NotImplementedError

class StaticRAGProvider(MemoryProvider):
    """静态展品知识检索 (ChromaDB Collection A)"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        # 实现专业展品背景的向量检索
        pass

class InsightArchiveProvider(MemoryProvider):
    """历史对话见解检索 (ChromaDB Collection B)"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        # 实现对该用户过往“觉察/见解”的语义检索
        pass

class UserGroupProvider(MemoryProvider):
    """用户群体偏好检索 (PostgreSQL/JSON)"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        # 实现对群体审美偏好、交流风格的提取与 Prompt 化转换
        pass

class RetrievalOrchestrator:
    """多路检索编排器 (指挥官)"""
    def __init__(self, providers: List[MemoryProvider]):
        self.providers = providers

    async def multi_path_retrieve(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        import asyncio
        # 1. 并发执行所有检索 (通过 asyncio 提升多库查询性能)
        tasks = [provider.search(query) for provider in self.providers]
        results = await asyncio.gather(*tasks)
        
        # 2. 扁平化结果列表并进行全局得分排序
        flat_results = [chunk for sublist in results for chunk in sublist]
        return sorted(flat_results, key=lambda x: x.score, reverse=True)
```

#### 3.2.2 记忆系统 (Memory System)
- **目录位置**: `memory/` (根文件夹)
- **职责**: 管理跨时空的对话上下文、长期洞察积淀与用户信息。
- **核心数据结构**:
```python
class UserGroupProfile(BaseModel):
    """用户群体画像模型 (User Group Profiles)"""
    group_id: str                 # 群体唯一标识 (如: "youth_tech", "elderly_family")
    category_name: str            # 类别名 (如: "科技青年", "亲子家庭")
    aesthetic_pref: str           # 审美偏好 (如: "前卫简约", "传统温馨")
    communication_pref: str       # 交流偏好 (如: "深度技术讨论", "浅显易懂介绍")
    typical_tags: List[str]       # 该群体典型标签: ["极客", "参数党"]
    response_style: Dict          # 响应风格定制: {"speed": "fast", "detail_level": "high"}

class InsightEntry(BaseModel):
    """对话见解模型 (Insight Archive)"""
    insight_id: str
    topic: str                    # 提取的主题: "设备操作疑问"
    content: str                  # 深度摘要: "用户曾表达过对 XX 展品手势交互的困惑"
    key_entities: List[str]       # 关联实体: ["双子机器人", "空间交互"]
    timestamp: datetime
    embedding: List[float]        # 见解内容的语义向量
```

**核心子模块与接口 (记忆 Hub 接口层)**:
- **定位**: **记忆对外提供服务**。作为代理层获取与存储信息的唯一官方入口。
**核心子模块与接口清单**:

#### **1. 短期记忆层 (Short-term Memory)**
- **文件**: `memory/short_term_memory.py`
- **职责**: 负责当前会话原始对话的实时存取与生命周期管理（JSON 缓存）。
- **核心接口**:
    - **`add_chat_history(role: str, content: str) -> None`**: 向缓存中追加一条原始对话记录。
    - **`get_raw_history(session_id: str) -> List[dict]`**: 获取当前会话的所有原始对话列表。
    - **`clear_chat_history(session_id: str) -> bool`**: 清空当前会话缓存（如切换用户或结束交谈时）。
    - **`get_history_count() -> int`**: 获取当前已缓存的对话轮数。

#### **2. 长期见解层 (Insight Archive)**
- **文件**: `memory/insight_archive.py`
- **职责**: 存储由 LLM 异步提炼的用户个性化见解与结论（向量库 B）。
- **核心接口**:
    - **`commit_insight(entry: InsightEntry) -> bool`**: 异步持久化一条新的见解条目。
    - **`search_insights(query_vector: list, top_k: int) -> List`**: (底层的向量召回) 供内部检索使用。
    - **`delete_insight(insight_id: str)`**: 删除错误的或过时的见解。

#### **3. 长期画像层 (User Group Profiles)**
- **文件**: `memory/user_group_profiles.py`
- **职责**: 维护用户群体的审美偏好、交流风格及类别配置。
- **核心接口**:
    - **`get_group_config(group_id: str) -> UserGroupProfile`**: 根据 ID 获取完整的群体配置。
    - **`match_group(user_features: dict) -> str`**: 根据感知层提供的特征（如年龄段、打扮）匹配最接近的群体 ID。
    - **`save_group_profile(profile: UserGroupProfile) -> None`**: 录入或更新某个群体的全局配置参数。
    - **`list_all_groups() -> List[str]`**: 列出当前系统中支持的所有群体类别。

#### **4. 记忆总线与对外服务 (Memory Hub)**
- **职责**: 记忆系统的对外“唯一柜台”，屏蔽底层存储细节。
- **核心接口**:
    - **`recall(query: str, user_id: str) -> List`**: (业务读) 统一触发“多路回想”逻辑（联动手感检索引擎）。
    - **`sync_persistence()`**: 定时或强制触发所有内存数据落盘。

#### 3.2.3 共鸣引擎 (Resonance Engine)
- **文件**: `services/resonance_engine.py`
- **职责**: 实现“技心”人设的人格化算法，调节回应的情感质感与美学比重。
- **核心子模块**:
    1.  **人格化滤镜**: 基于“技心”人设协议（Prompts），对 LLM 生成的原始文本进行倾向性调节。
    2.  **情感共鸣计算**: 评估用户输入的情感分值 (Vibe)，动态调整机器人回复的情绪强度。
    3.  **美学比重控制**: 在“技术准确度”与“情感表现力”之间寻找动态平衡点。
- **核心接口**:
    - `calculate_vibe(text_input, user_profile)`: 评估情感共鸣分值。
    - `apply_persona_filter(raw_response, context)`: 通过人设协议进行滤镜化处理。
    - `get_persona_config()`: 获取当前活跃的人设参数。
    - `update_persona_config(config)`: 动态调整情感系数与回复风格。

### 3.3 代理层 (Agent Layer)
- **职责**: 代理层作为整个系统的业务大脑，负责任务分解、策略路由与认知循环执行。
- **内部核心 Agent 单元 (Functional Roles)**:
    1.  **SceneAnalyzerAgent**: 环境与展品视觉分析，对应 **Perceive** 阶段，识别场景中的关键特征。
    2.  **DialogueAgent**: 对话策略与人设维护，对应 **Plan** 阶段，确保语义流畅。
    3.  **ActionAgent**: 机器人动作与播放规划，对应 **Execute/Context** 阶段，协调物理肢体行为。

- **交互场景分类 (Scene Agents)**:
    - **Exhibit Intro Agent**: 提供专业化、权威的展品背景讲解。
    - **Deep Chat Agent**: 跨轮次的深度语义探讨。
    - **Small Talk Agent**: 负责身份认同、日常寒暄及情感抚慰。

- **核心接口**:
    - `register_agent(name, agent)` ⮕ 注册新的 Agent 实例。
    - `execute_agent(name, task, context)` ⮕ 发起单次 Agent 执行任务。
    - `coordinate_agents(task, context)` ⮕ 跨 Agent 协作完成复杂任务。

### 3.4 感知与执行模块 (Perception & Execution)
#### 3.4.1 视觉系统 (Vision System)
- **文件**: `services/vision_service.py`
- **职责**: 负责实时摄像头流的捕捉、快照分析及展品特征提取。
- **核心接口**: `capture_snapshot()`, `get_latest_frame()`

#### 3.4.2 听觉系统 (Hearing System)
- **文件**: `services/asr_service.py`
- **职责**: 负责音频采集、断句及文字转化 (ASR)，将语音输入转化为可处理文本。
- **核心子模块**:
    1.  **音频流处理**: 实现低延迟的音频捕获与信号增益。
    2.  **云端 API 集成**: 支持百度语音、腾讯云等高精度 ASR 服务。
    3.  **本地备用模式**: 在网络异常时切换至轻量级本地识别模型。
- **核心接口**:
    - `start_listening()`: 开启麦克风监听。
    - `stop_listening()`: 停止监听并返回文本。
    - `set_api_config(config)`: 配置云端识别参数。
    - `switch_recognition_mode(mode)`: 在云端与本地模式间切换。

#### 3.4.3 语言系统 (Language System)
- **文件**: `services/llm_service.py`
- **职责**: 负责大模型的底层调用、流式输出管理及提示词注入。
- **核心子模块**:
    1.  **多模态处理**: 支持图像 + 文本的混合推理输入。
    2.  **流式输出管理**: 实现实时打字机响应效果。
    3.  **提示词工程**: 自动化管理 Agent 各阶段的 Prompts。
- **核心接口**:
    - `generate_stream(prompts, history, image_data)`: 发起模型预测。
    - `on_generation_complete(history)`: 触发异步内省逻辑。
    - `load_model(model_config)`: 加载特定参数的大模型。
    - `get_model_status()`: 监控显存与资源占用。

### 3.5 执行控制层 (Execution Layer)
#### 3.5.1 行为控制器 (Behavior Controller)
- **职责**: 将代理决策转化为底层的硬件执行指令。
- **数据结构**:
    - **`AgentAction`**: `{reply_text, motion_id, tts_config: dict, expression_id, light_effect}`
- **核心接口**: `sync_execute(action: AgentAction) -> None`

---

## 4. 模块交互与数据流 (Interaction & Data Flow)

### 4.1 交互时序图
![时序图](时序图.png)

### 4.2 数据流描述
1.  **语音唤醒**: 听觉系统解析语音 ⮕ 转化为文本 ⮕ 发布到 ROS 话题。
2.  **视觉捕捉**: 总控监听话题 ⮕ 触发视觉系统拍摄最新帧图像。
3.  **认知分析**: 认知层提取展品专业背景并召回历史相关记忆。
4.  **代理决策**: 代理层根据当前 Context (展品、历史、闲聊) 选取合适的 Agent (介绍/深聊/闲聊) 生成原始文本。
5.  **智慧生成**: 语言系统整合 Agent 策略、人设与历史 ⮕ 产生流式响应内容。
6.  **反馈输出**: 响应内容推送到前端显示，并触发 TTS 进行语音播报。

---

## 5. 目录结构 (Directory Structure)

```text
vl-rag-system/
├── agents/                  # 🤖 代理层 (任务分解与执行大脑)
│   ├── core/                # ⚙️ 核心原子代理 (功能逻辑实现)
│   │   ├── scene_analyzer.py # 👁️ 环境分析与视觉特征抽象
│   │   ├── dialogue_manager.py # 💬 语义对齐与对话策略控制
│   │   └── action_planner.py # 🦾 动作/表情序列规划排程
│   ├── base_agent.py        # 🆔 代理通用基类
│   ├── intro_agent.py       # 🏺 展品讲解专有代理
│   ├── chat_agent.py        # 💬 深度聊天专有代理
│   └── smalltalk_agent.py   # 🌸 闲聊与情感专有代理
├── api/                     # 🌐 Web 接口层 (FastAPI 分层实现)
│   ├── routers/             # 🚦 路由映射与状态码定义
│   └── controllers/         # 🛡️ 参数校验与服务调用中转
├── services/                # 🧱 核心服务层 (技术原子能力封装)
│   ├── agent_manager.py     # 🧭 场景分发与 Agent 状态调度
│   ├── resonance_engine.py  # 🎭 人设滤镜与情感共鸣计算
│   ├── llm_service.py       # 🧠 背景注入与流式输出管理
│   ├── tts_service.py       # 🔊 语音合成输出
│   ├── asr_service.py       # 🎙️ 听觉识别服务 (含云端/本地)
│   └── vision_service.py    # 📸 视觉捕捉服务 (多模态输入)
├── memory/                  # 🧠 记忆系统 (对外接口与长短时存储)
│   ├── short_term_memory.py # 🗒️ 短期 JSON 缓存 (原话历史)
│   ├── insight_archive.py   # 📁 长期见解库 (提炼后的结论/特征)
│   └── user_group_profiles.py # 👥 用户群体画像 (群体策略库)
├── rag/                     # 📚 检索引擎层 (底层检索原子能力)
│   ├── ingest.py            # 🏗️ 文本知识入库脚本 (ChromaDB)
│   ├── build_vector_db_new.py # 🖼️ 图像特征对齐入库脚本
│   └── retriever.py         # 🔍 统一检索调度器 (Provider 模式实现)
├── local_model_processor.py # 🤖 系统总控调度器 (Orchestrator)
├── main.py                  # 🏁 Web 服务入口与启动配置
├── config.py                # ⚙️ 全局配置中心
├── rag/                     # 📚 RAG 检索逻辑与知识管理
├── prompts/                 # 📝 提示词模板 (包含各 Agent 人设协议)
├── docs/                    # 📂 架构图、时序图与设计文档
├── data/                    # 💾 数据库与临时缓存
└── models/                  # 🤖 本地模型存放 (Embedding 等)
```

---

## 6. 技术选型 (Technology Stack)

| 核心维度 | 技术选型 | 作用说明 |
| :--- | :--- | :--- |
| **机器人框架** | ROS 2 Humble | 组件化异步通信与硬件节点管理 |
| **后端框架** | Python + FastAPI | 高性能、异步化的业务逻辑支撑 |
| **多模态核心** | DeepSeek / Qwen | 核心的语义理解、推理与视觉对齐 |
| **推理引擎** | Ollama / API | 驱动大语言模型的高效运行 |
| **向量数据库** | ChromaDB | 实时的展品专业知识向量检索 |
| **嵌入模型** | BGE-Small-ZH | 本地化的中文语义向量化 |
| **前端展现** | HTML + Vue 3 | 现代、组件化的交互式仪表盘与视觉反馈 |
| **语音识别** | 百度语音 API | 高精度的云端语音识别服务 |
| **语音合成** | TTS 服务 | 自然的流式语音输出 |
