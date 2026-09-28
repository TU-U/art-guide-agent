# 2026-08-10：机器人 Agent 架构讨论纪要

## 讨论目标

将当前以“视觉问答 + RAG + 对话”为主的系统，逐步重构为可持续理解物理环境、执行任务并验证结果的机器人 Agent。目标示例是：机器人看到苹果和篮子后，能够完成“拿起苹果并放入篮子”。

## 已达成的关键判断

### 1. 不继续堆叠串行模型链路

当前 `YOLO → VLM/LLM → 回复` 适合导览问答，但不适合作为机器人实时认知与控制主链路。问题包括：

- 每帧都等待语言模型会造成高延迟；
- 视觉结果被转成文字后，位置、置信度、时间关系容易丢失；
- 系统缺少稳定对象 ID，无法知道“刚才的苹果”和“现在的苹果”是否同一实例；
- 无法可靠判断动作是否成功或在失败后恢复。

推荐改为高频感知与低频语言推理解耦：

```text
RGB-D / Camera
  -> 检测、追踪、分割、深度融合（高频）
  -> World Model（实时状态）
  -> 事件（对象出现、丢失、移动、人员进入）
  -> LLM / VLM（仅按需解释、规划或对话）
  -> Behavior Tree / MoveIt / 技能执行
  -> 新观测回写 World Model
```

### 2. World Model 优先于 VLA

第一阶段的 World Model 不做生成式预测世界模型，也不需要复现完整 3D Scene Graph 论文系统。它首先是带时间和置信度的“当前世界事实表”，至少维护：

- 对象实例：`apple_001`、`basket_001`、`person_001`；
- 状态：`visible`、`occluded`、`attached`、`placed`、`lost`；
- 位置：电脑摄像头阶段先二维；Orbbec 接入后升级为三维位姿；
- 关系：`on`、`near`、`inside`、`attached_to`；
- 证据：最近观测时间、检测置信度、关联帧/深度数据。

VLA 是后续可插拔的技能策略，不是系统大脑。初期由 Behavior Tree 编排 `Find → Pick → Verify → Place → Verify`，由 MoveIt 负责无碰撞轨迹。未来可在 `PickPolicy` 等技能接口内评估 VLA。

### 3. Memory 采用“会话内上下文 + 会话结束沉淀”

不应每轮都做长期记忆检索和 LLM 反思。新的策略是：

| 时机 | 策略 |
|---|---|
| 会话进行中 | 当前会话历史直接注入一次；不对普通对话检索长期记忆 |
| 出现“上次、之前、还记得、历史、我的偏好”等语义 | 按需检索长期记忆 |
| 用户明确教学 | 用规则即时保存稳定事实 |
| 会话显式结束或空闲超时 | 统一生成摘要、洞察和事件 |

“全量注入”指每次 Prompt 都放入全部历史对话、所有记忆和所有资料。当前系统不适合这样做：上下文会无限增长、旧信息会干扰当前问题、成本和延迟会上升。短会话可以全量注入**当前会话历史**，但长期记忆应结构化沉淀并按需召回。

### 4. 人员感知采用事件驱动，不把视频流直接交给大脑

电脑摄像头原型的链路为：

```text
Webcam
  -> YOLO person detection
  -> ByteTrack 短期 track_id
  -> 进入/离开状态机
  -> PersonEntered / PersonExited
  -> World Model 当前在场人员 + SQLite 事件历史
  -> Social Agent 按事件决定是否调用 LLM
```

当前阶段只保证同一次连续出现在镜头中的稳定 `track_id`。离开后再回来是否为同一人属于 ReID 第二阶段；具体姓名需要用户确认或经授权的身份方案，不能仅按外观猜测。

### 5. 主动社交由结构化事件驱动

Social Agent 不接收逐帧视频，也不做 OCR。本阶段传给 LLM 的是人员事件转换出的事实文字，例如：

```text
检测到一位访客进入摄像头区域
当前在场人数：1
视觉追踪置信度：0.89
身份状态：未确认
```

LLM 只能生成简短、友好、轻微俏皮的招呼。必须有全局冷却、去重和约束：不能猜姓名、性别、年龄、外貌、情绪或敏感身份；默认不自动调用 LLM/TTS，只有显式配置后才启用。

## 本次代码变更（根目录版本）

### 访客感知原型

- 新增 `person_presence/`：摄像头输入、YOLO person 检测、ByteTrack 追踪、进入/离开状态机、SQLite、FastAPI 页面；
- 新增 `person_presence/social.py`：事件驱动的 LLM 社交层；
- 新增 `docs/person_presence_prototype.md`：启动、配置和验收说明；
- 新增测试 `tests/test_person_presence.py`。

### Memory 重构

- 当前会话历史不再与 `combined_context` 重复注入；
- 普通轮次不再自动运行记忆反思；
- 新增按语义触发的长期记忆召回；
- 新增 `memory/session_summary.py`，在会话结束时持久化摘要；
- 新增 `POST /memory/session/end` 显式结束会话接口；
- Web 主链路在配置的静默时间后自动尝试会话沉淀；
- 未修改 `vl_rag_system_v1/`。

## 后续讨论顺序

1. 定义 World Model v0 的对象、状态、关系与事件契约；
2. 让人员感知和物体感知都写入同一个 World Model；
3. 接入 Orbbec 后完成 RGB-D 到三维对象位置的融合；
4. 定义 `Observe`、`FindObject`、`Pick`、`Place`、`Verify` 等技能接口；
5. 用 Behavior Tree 和 MoveIt 完成固定桌面“苹果入篮”MVP；
6. 最后评估 VLA 作为局部操控技能，而非替代世界状态和安全执行。
