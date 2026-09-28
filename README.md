# Art Guide Agent｜多模态机器人导览智能体

> 让机器人不只“看图回答”，而是能将语音、视觉、记忆与现场事件组织为可解释、可演进的交互闭环。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20兼容-22314E?logo=ros&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-接口与调试-009688?logo=fastapi&logoColor=white)
![状态](https://img.shields.io/badge/状态-研究原型-F59E0B)

**Art Guide Agent** 是一个面向展厅、实验室与桌面机器人场景的多模态交互系统。它将语音输入、相机画面、用户指向、领域知识与会话记忆接入同一条机器人交互链路，并为后续的世界模型（World Model）、任务规划与可验证操控预留清晰边界。项目采用视觉语言检索增强（VL-RAG）作为多模态理解与知识问答路线。

| 核心问题 | 项目方案 | 产生价值 |
| --- | --- | --- |
| 传统机器人问答无法理解“这个”“刚才那个”，每轮交互彼此割裂；把视频流逐帧交给大模型又会带来高延迟和失控风险。 | 以**事件驱动感知**沉淀结构化事实，并使用“当前会话上下文 + 按需长期记忆 + 图文问答”组织单轮交互。 | 机器人能够在视觉异常时稳定退回语音对话，在多轮交流中保持上下文，并为后续具身任务提供可观测的数据入口。 |

```text
          ┌────────────── 输入层 ───────────────┐
          │  ASR · Camera · RGB-D · Pointing    │
          └─────────────────┬───────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  机器人交互中枢     │
                  │  路由 · 上下文管理  │
                  └──────┬───────┬─────┘
                         │       │
          ┌──────────────▼──┐ ┌──▼────────────────┐
          │ VLM / LLM / RAG │ │ 记忆系统            │
          │ 多模态理解与回复 │ │ 会话 · 事实 · 事件  │
          └──────────────┬──┘ └──┬────────────────┘
                         │       │
                    ┌────▼───────▼─────┐
                    │ TTS · Web 调试页 │
                    └──────────────────┘

  演进路线：感知事件 → World Model → 行为树 / ROS Action → MoveIt
```

## 项目要解决什么问题

不少“机器人 + 大模型”演示仍是串行链路：**检测物体 → 转成文字 → 询问模型 → 语音播报**。这足以完成一次性讲解，但连续真实交互会很快暴露局限：

- 用户指着实物问“这是什么”，系统不知道“这”指向哪里；
- 相机画面已经过期，模型却仍把它当作当前现场回答；
- 用户纠正机器人后，希望它在后续对话中记住；
- 有人进入画面时，机器人应该只响应一次，而非每一帧都触发；
- 后续若要完成“把苹果放进篮子”，系统必须能把语言拆成可观察、可验证的步骤。

Art Guide Agent 将大模型定位为**语义理解与交互组件**，而不是实时物理事实的唯一来源：视觉被转为快照或事件；对话被限制在有边界的会话内；持久信息被有条件地写入；而迈向结构化世界状态的路径被明确保留。

## 当前已具备的能力

| 能力 | 作用 | 主要实现 |
| --- | --- | --- |
| 多模态对话 | 根据视觉输入是否新鲜，自动选择纯文本或图文输入，并生成基于画面的回复。 | `main.py`、`local_model_processor.py`、`services/vlm_service.py` |
| 机器人语音链路 | 通过 ROS 接收 ASR 文本，过滤回声和重复输入，流式切句并以单队列 TTS 避免抢播。 | `local_model_processor.py`、`services/` |
| 会话记忆 | 保留有边界的会话历史；立即写入明确教学事实；仅在具有历史意图时召回长期记忆；会话结束或空闲时统一沉淀。 | `memory/` |
| 可选 RAG 知识检索 | 用于检索稳定的领域知识，而不是把它误当作实时环境状态。 | `rag/` |
| 用户指向识别原型 | 结合 RGB-D、手部关键点、YOLO 检测框与三维射线，识别用户可能指向的目标，输出调试快照与 JSON。 | `point/` |
| 人员在场感知原型 | 用摄像头人员追踪把连续视频转为进入/离开事件，记录 SQLite 历史，并提供带冷却机制的可选问候。 | `person_presence/` |
| 调试界面与接口 | 通过 FastAPI 暴露对话、记忆、TTS 与感知状态，便于现场排障。 | `main.py`、`frontend/`、`person_presence/app.py` |

## 系统架构

### 运行时交互链路

```text
用户语音
  → ASR / HTTP 输入
  → 机器人中枢
      ├─ 过滤空输入、重复输入与 TTS 回声
      ├─ 记录本轮对话与可确定的用户教学事实
      ├─ 读取有长度限制的当前会话上下文
      ├─ 仅在意图需要时召回长期记忆
      └─ 判断最新视觉帧是否仍属于本轮交互
  → VLM / LLM（可选附加 RAG 上下文）
  → 流式切句与单工作线程 TTS 队列
  → 机器人音频输出 + Web 调试状态
```

### 感知与具身能力演进链路

```text
Camera / RGB-D
  → 检测 · 跟踪 · 手势/指向几何计算
  → 结构化观测与事件
  → World Model（规划中：共享世界状态）
  → 行为树 / ROS Action（规划中：任务执行）
  → MoveIt / 夹爪 / 结果验证（规划中：物理控制）
```

目前仓库**尚未实现**共享 World Model、行为树执行器、MoveIt 集成或自主抓取放置。现有 `point/` 与 `person_presence/` 都是刻意隔离的感知原型；后续引入 World Model 契约后，它们应成为统一的观测适配器，而不是直接承担任务决策。目标设计与分期计划见[机器人 Agent 调研与实施初稿](docs/robot_agent_research_draft.md)。

## 目录说明

```text
.
├── main.py                    # FastAPI：对话、记忆与 TTS 调试接口
├── local_model_processor.py   # ROS 机器人中枢与流式交互循环
├── services/                  # ASR、视觉输入、VLM/LLM、TTS 适配层
├── memory/                    # 会话、事实、洞察、事件与摘要记忆
├── rag/                       # 可选领域知识检索
├── point/                     # RGB-D 手势指向与三维射线原型
├── person_presence/           # 摄像头人员事件与主动问候原型
├── prompts/                   # 交互与人设提示词
├── frontend/                  # 本地调试页面
├── docs/                      # 架构决策与调研文档
├── tests/                     # 原型单元测试
└── vl_rag_system_v1/          # 保留的历史/并行版本，供迁移参考
```

## 快速开始

### 1. 准备 Python 环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

如果只需要 Web、记忆和模型调用链路，不需要 ROS 或手势指向依赖，可使用：

```bash
pip install -r requirements-main.txt
```

`rclpy`、`cv_bridge`、`sensor_msgs`、`message_filters` 等 ROS 包由 ROS 2 安装环境提供，而不是通过 pip 安装。

### 2. 配置本地凭据

在根目录创建 `.env`。该文件已被 Git 忽略，**不得提交**。

```dotenv
# 可选 qwen_omni 或 deepseek
VLM_PROVIDER=qwen_omni
DASHSCOPE_API_KEY=替换为你的密钥

# 可选：启用讯飞 TTS 时填写
XF_APPID=
XF_API_KEY=
XF_API_SECRET=
```

其他可配置项见 [`config.py`](config.py)。模型权重请放在 `models/`，运行数据放在 `data/`；二者均被排除在版本控制之外。

### 3. 启动 Web 调试后端

```bash
python3 main.py
```

访问 `http://127.0.0.1:8765/`（或 `BACKEND_PORT` 指定的端口）。后端提供 `/chat`、`/memory/*` 与 `/api/tts/*` 等接口。

### 4. 启动 ROS 交互链路（可选）

加载 ROS 2 环境，并确保 ASR、相机话题与目标硬件匹配后执行：

```bash
./start_all.sh
```

根目录脚本会启动 ASR、视觉输入和机器人中枢。相机驱动、话题名及音频播放配置需按实际机器人部署环境调整。

### 5. 启动人员在场感知原型（可选）

```bash
python3 -m person_presence
```

访问 `http://127.0.0.1:8090`。首次运行 YOLO 可能下载模型权重；该原型需要摄像头及 `requirements.txt` 中的视觉依赖。

## 开发与验证

```bash
# 人员在场状态机、SQLite 与冷却逻辑测试
python3 -m unittest discover -s tests -v

# 核心模块语法检查
python3 -m compileall -q memory person_presence point main.py local_model_processor.py
```

建议按以下顺序完成硬件测试：

1. 不接相机时验证 Web 对话和记忆行为；
2. 用录制图片源验证新鲜/过期视觉帧的降级逻辑；
3. 用回放帧验证感知事件的稳定性；
4. 在目标机器上联调 ROS 话题、TTS 播放与端到端时延。

## 设计原则

- **先确认新鲜度，再追求表达流畅**：过期画面不能被悄悄当成当前现场回答。
- **让事件进入推理，而不是让原始视频流进入推理**：大模型接收简洁、可审计的事实，而不是连续视频帧。
- **记忆不等于世界状态**：记忆保存过去交互与经验；物理状态必须带有时间、置信度与观测证据。
- **稳定降级**：相机或检测服务异常时，普通语音交互不能随之中断。
- **验证优先于自主性**：物理动作完成与否应由新观测确认，而不是由模型生成文本推断。
- **隐私优先**：人员感知只使用临时相机轨迹 ID，不保存姓名或人脸特征；身份关联必须有用户知情同意和独立方案。

## 路线图

- [x] 语音、图文问答、可选 RAG、记忆和 TTS 交互闭环
- [x] 摄像头人员进入/离开事件原型
- [x] RGB-D 用户指向识别原型
- [ ] 定义人员、物体、关系与观测证据的共享 World Model 契约
- [ ] RGB-D 物体跟踪、稳定对象 ID 与三维位姿融合
- [ ] `Observe`、`FindObject`、`Pick`、`Place`、`Verify` 等 ROS Action
- [ ] 行为树任务执行器与仿真基线
- [ ] 基于 MoveIt 的固定桌面“苹果入篮”验证型 MVP

## 相关文档

- [机器人 Agent 架构讨论纪要](docs/2026-08-10_robot_agent_discussion.md)
- [机器人 Agent 调研与实施初稿](docs/robot_agent_research_draft.md)
- [人员在场感知原型说明](docs/person_presence_prototype.md)

## 参与贡献

项目仍处于研究原型阶段，欢迎提交 Issue 和 Pull Request，尤其欢迎：可复现实验数据、感知基准、ROS 集成、回放数据集、World Model 契约以及测试覆盖改进。请勿提交 API 密钥、模型权重、摄像头录制内容或运行时生成数据。

## 开源许可

当前仓库尚未包含 `LICENSE` 文件。在公开分发或接受第三方贡献前，请补充明确的开源许可协议。
