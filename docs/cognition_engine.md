# 认知引擎 (Cognition Engine) 设计文档

本文档详细定义了 “技心” 机器人的核心智慧循环（Cognitive Loop）。该循环由六大核心模块组成，指导 Agent 如何从感知环境到执行动作，并最终实现自我反思。

---

## 1. 核心循环概览 (The 6-Step Loop)

整个认知过程遵循 **Sense-Think-Act-Reflect** 的闭环逻辑：

1.  **Perceive (感知抽象)**: 原始信号 ⮕ 语义理解
2.  **Retrieve (知识调取)**: RAG 检索 ⮕ 领域知识增强
3.  **Plan (逻辑规划)**: 逻辑推理 ⮕ 决策输出
4.  **Execute (动作分发)**: 语言生成 ⮕ 肢体控制 (ROS Action)
5.  **Context (场景锚点)**: 意图锁定 ⮕ 交互状态维护
6.  **Reflect (深度自省)**: 经验沉淀 ⮕ 长期记忆积累

---

## 2. 模块职责详解 (Detailed Module Responsibilities)

### 2.1 Perceive (感知抽象)
- **输入**: 来自 `asr_service` 的文本片段 + `vision_service` 的图像特征/识别标签。
- **职责**: 将物理信号转化为 **“语义化场景描述”**。
- **示例**: ASR 听到“那个”，Vision 识别到手指指向 A01 展品。Perceive 模块输出为：`{focus: "Artifact_A01", query: "detail_request"}`。

### 2.2 Retrieve (知识调取)
- **输入**: Perceive 提取的关键词与实体。
- **职责**: 从本地向量知识库 (`ChromaDB`) 与静态常识库 (`static_rag.py`) 中召回相关背景。
- **输出**: 展品历史、制作工艺等专业背景文本。

### 2.3 Plan (逻辑规划)
- **输入**: 上文 Context + 检索到的知识 + 当前场景描述。
- **职责**: 充当 **“大脑”**，决定下步行动。采用 Chain of Thought (CoT) 推理：
    - **Step 1**: 确定用户真实意图（是想听讲解还是开玩笑？）。
    - **Step 2**: 制定回复策略（专业叙事 vs. 温暖回应）。
    - **Step 3**: 规划非语言动作（是否需要点头、转向用户？）。

### 2.4 Execute (动作分发)
- **职责**: 将 Plan 的决策转化为实际的反馈序列。
- **输出**:
    - **语音流**: 发送文本至 `tts_service`。
    - **控制流**: 发送运动指令（如：`point_to_artifact`, `nod_head`）至 ROS 控制话题。

### 2.5 Context (场景锚点)
- **职责**: 维护对话的 **“瞬时状态”** 与 **“交互深度”**。
- **核心数据**: `user_intent_lock` (当前对话聚焦的主题), `interaction_level` (当前亲密/正式程度)。

### 2.6 Reflect (深度自省)
- **职责**: 对话完成后（或异步进行）的 **“自我升级”**。
- **逻辑**: 
    - 评估对话成败：用户是否得到了满意的答案？
    - 提取新知识：如果用户分享了新信息，将其整合进 `Insight_Archive`。
    - 调整人设：根据用户反馈微调共鸣引擎的参数。

---

## 3. 架构映射 (Architectural Mapping)

认知引擎的 6 模块与系统 5 层架构的映射关系如下：

| 认知引擎模块 | 对应的层级 (Layer) | 关键组件 / 服务 |
| :--- | :--- | :--- |
| **Perceive** | 感官层 (Perception) | `ASR`, `Vision` |
| **Retrieve** | 认知层 (Cognitive) | `RAG Retriever`, `Memory System` |
| **Plan** | 代理层 (Agent) | `Agent Brain` (Logic in `agents/*.py`) |
| **Execute** | 反馈层 (Execution) | `TTS`, `Body Controllers` |
| **Context** | 调度层 (Orchestration) | `System Orchestrator` |
| **Reflect** | 代理层 & 认知层 | `Memory/Insight_Archive` |

---

## 4. 数据流示例 (Example Flow)

1.  **用户**: “这个看起来好老啊。”
2.  **Perceive**: 识别到用户情感为“好奇”，目标为“当前视线内的陶瓷器”。
3.  **Retrieve**: 调取“龙泉窑瓷器”的年份背景（南宋）。
4.  **Plan**: 决定先幽默回应（“确实，它已经 800 多岁了”），再引入历史。
5.  **Execute**: TTS 播报 + 机器人微微向前倾（倾听/亲近状）。
6.  **Context**: 标记当前为“宋代文物专题讲解”阶段。
7.  **Reflect**: 记录用户对“年份”话题感兴趣，存入用户画像。
