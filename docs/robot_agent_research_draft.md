# VLM-RAG 机器人 Agent 重构调研与实施初稿

> 状态：讨论初稿，面向初学者。  
> 目标：将当前以“图像 + 对话 + RAG”为主的导览系统，逐步重构为能理解真实场景、执行并验证物理任务的机器人 Agent。  
> 本文先确定工程路线与验证方法，不在此阶段追求新的端到端模型或复杂多 Agent。

## 1. 一句话结论

应建设的不是“一个会看图回答问题的大模型串联流程”，而是一个**闭环具身系统**：

```text
用户说任务
  -> 任务规划（决定做哪些技能）
  -> 行为执行器（检查、重试、恢复）
  -> 机器人技能（观察、找物、抓取、放置、导航）
  -> ROS 2 / MoveIt / 硬件
  -> 新观测回写 World Model
  -> 判断任务是否真的完成
```

其中，World Model（世界模型）记录“机器人当前相信世界是什么样”；Memory（记忆）记录“过去发生过什么、学到什么”。二者不能混为一个向量库。

## 2. 目标场景与边界

### 2.1 最终愿景

用户说：“把苹果放进篮子里。”机器人能够：

1. 找到苹果和篮子；
2. 利用 RGB-D 得到苹果、篮子相对于机器人的三维位置；
3. 判断苹果是否可抓取、篮子是否可放置；
4. 规划并执行抓取、抬升、放置；
5. 通过视觉、夹爪状态等证据验证“苹果真的在篮子里”；
6. 失败时重新观察、换抓取姿态或向用户求助。

### 2.2 第一阶段 MVP（必须刻意缩小）

先只验证**固定桌面上的苹果入篮**，不要同时做移动导航、任意物体、复杂对话和长期学习。

| 项目 | MVP 约束 |
|---|---|
| 物体 | 一个苹果、一个篮子、一张桌子 |
| 场景 | 室内固定光照；相机固定或已完成外参标定 |
| 动作 | 找到、抓取、抬升、放入篮内、验证 |
| 输入 | RGB、Depth、CameraInfo、TF、夹爪状态 |
| 成功条件 | 苹果在篮中、夹爪已松开、视觉再确认 |
| 暂不做 | 任意物体泛化、移动底盘、端到端 VLA 控制、多人协作 |

### 2.3 非目标

- 不把大语言模型直接用于输出关节轨迹；
- 不让一段 VLM Caption 直接覆盖物理事实；
- 不以“换一个更大模型”替代数据、标定、追踪和执行验证；
- 不先建设相互聊天的多 Agent 系统。

## 3. 当前仓库与目标系统的差距

当前仓库是导览/交互系统：ASR、图片快照、VLM/LLM、展品 RAG、TTS，以及面向对话的 Memory。它很适合保留为“交互与知识能力”，但尚不具备可控操纵能力。

| 已有能力 | 可以保留 | 需要新增或重构 |
|---|---|---|
| ASR / TTS | 用户输入和任务反馈 | 将语音转成结构化任务，而非直接拼 Prompt |
| RAG | 展品知识、规则、说明书问答 | 不作为实时物理状态来源 |
| VLM / YOLO | 语义理解、候选检测 | 多帧追踪、分割、深度融合、3D 位姿 |
| 对话 Memory | 用户偏好、对话事件 | 新增空间、情景、程序、视觉记忆 |
| ROS 图像节点 | 相机接入基础 | 标准 ROS Action、TF、MoveIt、生命周期管理 |

当前 `point/` 中是 `YOLOv8s` 通用权重，以 `imgsz=320`、约每秒一次的方式检测二维框。它可作为原型输入，但不能直接承担抓取目标的稳定三维定位。

## 4. 推荐的工程架构

```text
┌──────────────────── 交互层 ────────────────────┐
│ ASR / Web / TTS / 对话 / 文档 RAG              │
└───────────────────────┬────────────────────────┘
                        │ TaskSpec（结构化任务）
┌───────────────────────▼────────────────────────┐
│ Task Agent：受约束的任务规划                    │
│ 例：Find(apple) -> Pick(apple) -> Place(basket) │
└───────────────────────┬────────────────────────┘
                        │ BT 黑板 + ROS Actions
┌───────────────────────▼────────────────────────┐
│ Task Executive：Behavior Tree                   │
│ 前置条件 / 超时 / 取消 / 重试 / 恢复 / 验证     │
└──────────────┬───────────────────┬──────────────┘
               │                   │
┌──────────────▼──────┐  ┌─────────▼─────────────┐
│ World Model          │  │ Skills                │
│ 当前物理事实          │  │ Observe / Find / Pick │
│ 3D 对象、关系、置信度 │  │ Place / Verify        │
└──────────────┬──────┘  └─────────┬─────────────┘
               │                   │
┌──────────────▼───────────────────▼─────────────┐
│ ROS 2：RGB-D、TF、机器人状态、MoveIt、夹爪等    │
└────────────────────────────────────────────────┘
```

### 4.1 为什么采用 Behavior Tree（行为树）

语言模型擅长把意图变为高层计划，但不会可靠地处理抓取失败、目标被遮挡或动作超时。行为树适合把每个动作写成可观察状态：`RUNNING / SUCCESS / FAILURE`，并明确失败后的重试或重新观察。Nav2 已将 ROS Action 和行为树用于导航与自治执行，是可借鉴的工程模式。

### 4.2 World Model：第一版应是什么

这里的 World Model 指**结构化、随观测更新的世界状态估计器**，不是需要训练预测未来视频的生成式世界模型。

最低限度应包含：

```text
WorldState
├── robot_state：base/arm/gripper 的位姿、运动状态
├── objects：当前对象实例表
├── relations：对象间空间关系
├── spatial_map：桌面、占据或局部点云
└── observations：支撑当前判断的原始观测证据
```

`ObjectInstance` 建议字段：

```text
object_id: apple_001                 # 稳定实例 ID，而非每帧序号
class_distribution: {apple: 0.93}
pose: position + orientation         # 在 map 或 base_link 坐标系
pose_covariance: 不确定性
dimensions: 长宽高或点云包围盒
state: visible | occluded | attached | placed | lost
affordances: graspable, container, support_surface
last_seen_at / confidence
evidence_ids: 对应 Observation 的 ID 列表
```

关系最初只需：`on`、`inside`、`near`、`attached_to`、`reachable_from`。每个关系也应有时间和置信度。

### 4.3 Memory：过去经验，不是当前事实

| 记忆类型 | 保存内容 | 示例 |
|---|---|---|
| 工作记忆 | 当前任务、当前 BT 状态、最近几帧 | “正在执行 Pick(apple_001)” |
| 空间记忆 | 房间、固定物体、历史地图 | “收纳篮常位于桌面右侧” |
| 情景记忆 | 一次任务的观察—动作—结果 | “上次苹果因反光抓取失败” |
| 语义记忆 | 文档、展品知识、用户偏好 | “用户偏好语音反馈” |
| 程序记忆 | 已验证技能参数和恢复规则 | “低矮物体从上方抓取更稳定” |
| 视觉记忆 | 图像、深度、对象裁剪、embedding、场景图 | “apple_001 在第 18 帧的外观证据” |

视觉记忆的一条 `Observation` 不应只是 `图片 + Caption`，建议同时关联 RGB、深度、相机内参、相机位姿、对象掩码/框、对象 embedding、场景关系和模型版本。

## 5. 感知路线：YOLO 应放在哪里

YOLO 是一类一次前向推理就输出目标类别和二维框的目标检测模型家族；当前项目使用的是通用权重 `YOLOv8s`。它不是“视觉识别的总称”，也不是完整的机器人感知系统。

推荐链路：

```text
RGB-D 帧
  -> 目标检测（YOLO：苹果/篮子候选）
  -> 多帧追踪（稳定 object_id）
  -> 实例分割（目标精确轮廓）
  -> 深度融合（深度中位数/点云，不取单个像素）
  -> 3D 位姿与桌面关系
  -> 写入 World Model
```

实际建议：

1. 先将 `imgsz=320` 提升到 640，记录速度与漏检变化；
2. 加多帧确认，而不是检测一帧就相信；
3. 收集真实视角的苹果、篮子、桌面图片后微调本地检测/分割模型；
4. 用 Grounding DINO 一类开放词表检测作为低频的“按文字找物”补充，不作为实时控制环唯一感知；
5. 抓取使用实例分割 + 深度，而不只使用检测框中心。

## 6. 无机器人时可完成的工作

以下工作会直接复用于真机，且比等待硬件更适合当前阶段。

| 工作 | 输入 | 产出 / 验收 |
|---|---|---|
| 离线感知基准 | 手机/公开桌面图片 | 苹果、篮子漏检率、误检率、检测延迟 |
| YOLO 稳定化 | 视频帧 | 同一物体跨帧稳定 ID；连续 3–5 帧才确认 |
| World Model MVP | 模拟 JSON 检测结果 | 对象、关系、丢失、重新出现等单元测试 |
| 回放器 | RGB、Depth、相机内参录制数据 | 不接硬件也能驱动感知和世界模型 |
| 行为树 | 假 Action Server | 成功、超时、抓取失败、重观察等流程测试 |
| MoveIt 演示 | URDF + RViz | 无实体机器人验证抓取/放置轨迹和碰撞场景 |
| 仿真 | Isaac Sim 或 ManiSkill | RGB-D、物理交互、任务执行验证 |

选择仿真器的原则：

- 有 Ubuntu 和 NVIDIA RTX 显卡：优先尝试 Isaac Sim，之后可以接 ROS 2、相机和机械臂；
- 电脑性能一般或处于 Windows/WSL：先做离线回放 + RViz/MoveIt；不要先投入大量时间配置重型仿真器；
- ManiSkill 更适合操控任务、合成数据和机器人学习实验；其 Linux 支持最佳，WSL 的渲染能力受限。

## 7. 分阶段改进方案

### 阶段 0：建立基线（1 周）

- 明确 ROS 2 版本、Orbbec 型号、机械臂/夹爪型号、URDF、现有话题；
- 为 YOLO 建离线测试集和指标；
- 写出本项目的 TaskSpec、World Model、技能 Action 的接口草案；
- 不改主链路，不先训练大模型。

完成标准：可以用录制/模拟数据演示“看到苹果和篮子后，World Model 出现两个稳定对象”。

### 阶段 1：桌面感知与世界状态（2–3 周）

- RGB-D 同步、相机内参和 TF；
- YOLO 检测 + 跟踪 + 分割 + 深度融合；
- 写入对象表、关系图和观测证据；
- 用 RViz 或 Web 页面可视化世界状态。

完成标准：同一苹果被多次观察仍是同一 ID，且有可用三维位置和置信度。

### 阶段 2：可验证操控（2–4 周）

- 引入 MoveIt 2、Planning Scene、夹爪控制；
- 把 `Observe`、`FindObject`、`Pick`、`Place`、`Verify` 定义为 ROS Action；
- 用 Behavior Tree 编排抓取失败后的恢复策略。

完成标准：仿真或真机完成固定场景苹果入篮，并提供执行事件日志。

### 阶段 3：记忆和 Agent（2 周）

- 将任务日志写为情景记忆；
- 视觉记忆关联对象和场景图；
- LLM 根据 TaskSpec、World Model 摘要和相关记忆生成受约束计划；
- RAG 保持为文档知识工具，不介入底层控制。

完成标准：能回答“苹果刚才在哪里”“为什么这次抓取失败”，并能在下一轮优先采用成功策略。

### 阶段 4：开放世界能力（后续）

- 开放词表找物、用户指代消解、主动观察；
- 多房间导航与空间记忆；
- 真实数据微调、仿真到真实迁移；
- 在有稳定基线后再评估 VLA 作为技能策略。

## 8. 初学者概念查询表

| 概念 | 属于哪个板块 | 是什么 | 有什么用 | 常见应用 | 入门练习 |
|---|---|---|---|---|---|
| ROS 2 | 机器人中间件 | 节点、话题、服务、Action 的通信框架 | 让相机、机械臂、规划器协作 | 所有机器人系统 | 写发布/订阅两个节点 |
| Topic | ROS 通信 | 连续广播数据的通道 | 传 RGB、深度、关节状态 | 传感器数据流 | 发布一条模拟相机消息 |
| Service | ROS 通信 | 一问一答的短请求 | 获取配置、切换模式 | 查询类任务 | 写 `GetWorldState` 服务 |
| Action | ROS 通信 | 有进度、可取消的长任务 | 抓取、导航、放置 | 长时间机器人动作 | 写一个 5 秒可取消的假 `Pick` Action |
| TF / 坐标系 | 几何基础 | 各部件坐标系之间的变换树 | 将相机里的苹果变成机械臂可到的位置 | 所有视觉操控 | 画出 camera、base、gripper 的 TF 图 |
| RGB-D | 三维感知 | 彩色图 + 每像素深度图 | 从二维检测定位到三维 | 抓取、建图、避障 | 将一个像素反投影为 3D 点 |
| 相机标定 | 三维感知 | 获得内参和相机外参 | 让深度与真实坐标可信 | 定位、抓取 | 学会读取 `CameraInfo` |
| YOLO | 视觉感知 | 目标检测模型家族 | 找到苹果候选二维框 | 实时检测、计数 | 对 50 张桌面图测漏检率 |
| 目标追踪 | 视觉感知 | 跨帧关联同一物体 | 让 apple_001 不随画面跳变 | 视频、机器人观察 | 以 IoU 给检测框分配稳定 ID |
| 实例分割 | 视觉感知 | 为每个物体生成像素掩码 | 获得准确边界、过滤背景深度 | 抓取、物体测量 | 在掩码内取深度中位数 |
| 开放词表检测 | 视觉感知 | 用文字类别查询物体 | 找训练集中没有的“红色杯子” | 找物、用户指代 | 用 Grounding DINO 查询 `apple` |
| 点云 | 三维感知 | 大量三维点的集合 | 表示物体与桌面几何 | 抓取、建图、避障 | 从深度图生成局部点云 |
| 6D Pose | 三维感知 | 物体的位置 + 朝向 | 决定抓取方向和放置方向 | 机械臂操控 | 区分“位置”与“朝向” |
| World Model | 认知/状态 | 当前世界的结构化、带置信度的事实 | 规划前检查对象在哪、能否抓 | 自主机器人 | 用 JSON 表示苹果、篮子和 `on` 关系 |
| Scene Graph | 认知/状态 | 对象为节点、关系为边的图 | 表达 inside/on/near 等关系 | 空间问答、规划 | 写 `inside(apple,basket)` 图结构 |
| Memory | 认知/学习 | 对过去的结构化保存和检索 | 复盘任务、记住偏好和经验 | 长时交互 | 为一次失败抓取生成事件记录 |
| RAG | 知识检索 | 先检索文档，再交给 LLM 回答 | 回答展品、说明书、规则问题 | 问答助手 | 查询一篇展品文档并引用它 |
| LLM / VLM | 语义推理 | 文本模型 / 图文模型 | 理解意图、解释、生成计划候选 | 对话、开放语义 | 将“拿苹果”转为 TaskSpec JSON |
| Behavior Tree | 任务执行 | 用树结构描述条件、动作、重试 | 可靠执行并处理失败 | Nav2、机器人任务 | 画 `Find -> Pick -> Verify -> Place` 树 |
| MoveIt 2 | 运动规划 | ROS 2 机械臂规划框架 | 计算无碰撞轨迹 | 机械臂抓取/放置 | 在 RViz 中规划机械臂姿态 |
| Planning Scene | 运动规划 | MoveIt 中的机器人、障碍物和附着物状态 | 避免撞桌子；抓起后让物体随夹爪移动 | 抓取规划 | 加入桌子和苹果碰撞体 |
| MTC | 运动规划 | MoveIt Task Constructor | 把抓取拆为靠近、抓、抬升、放置 | Pick-and-place | 阅读其 pick-and-place 示例 |
| VLA | 机器人学习 | Vision-Language-Action 模型 | 从图像和指令产生动作策略 | 泛化操作研究 | 先阅读/运行示例，不先接真机 |
| 仿真器 | 测试环境 | 虚拟机器人、传感器和物理世界 | 无硬件验证、生成数据 | 算法开发、训练 | 在仿真中放苹果和篮子 |

## 9. 建议优先阅读的项目

| 项目 | 为什么看 | 暂时不必照搬 |
|---|---|---|
| [MoveIt 2](https://github.com/moveit/moveit2) | 机械臂规划和碰撞场景的标准基础 | 不先改复杂 C++ 内核 |
| [MoveIt Task Constructor](https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/moveit_task_constructor.html) | 抓取任务如何拆成可组合子步骤 | 不急着写所有高级 stage |
| [Nav2 Behavior Tree](https://docs.nav2.org/configuration/packages/configuring-bt-xml.html) | ROS Action 和行为树的成熟执行模式 | 不把导航包直接搬进项目 |
| [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) | RGB-D 到开放词表 3D Scene Graph 的参考 | 第一版不必完整复现其全部模型 |
| [Hydra](https://github.com/MIT-SPARK/Hydra) | 实时层次化 3D 场景图设计 | 依赖重，先借鉴数据结构 |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | 开放词表文字找物 | 不作为实时控制环唯一模型 |
| [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) | 文字找物、分割与视频追踪 | 先用于离线验证 |
| [Isaac Sim](https://github.com/isaac-sim) | ROS 2、传感器和物理仿真 | 无合适 GPU 时不优先安装 |
| [ManiSkill](https://github.com/mani-skill/ManiSkill) | 操控任务、合成数据、机器人学习基准 | 不把 RL 训练作为第一里程碑 |

## 10. 本周建议任务

1. 确认硬件和软件清单：ROS 2 版本、Orbbec 型号、相机话题、机械臂、夹爪、URDF、GPU；
2. 选定“苹果入篮”MVP，写一条完整成功/失败验收标准；
3. 用 50 张真实桌面图建立 YOLO 离线基准；
4. 创建 World Model 的 JSON 样例和对象状态机；
5. 阅读 MoveIt MTC 与 ConceptGraphs 的 README，只记录数据流和接口，不急于安装全部依赖；
6. 下周再决定先接 RViz/MoveIt 仿真还是 Isaac Sim。

## 11. 参考资料与调研依据

- [ROS 2 节点、Topic、Service、Action 基础](https://docs.ros.org/en/rolling/Concepts/Basic/About-Nodes.html)
- [Nav2 Behavior Tree](https://docs.ros.org/en/kilted/p/nav2_behavior_tree/index.html)
- [MoveIt Task Constructor](https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/moveit_task_constructor.html)
- [OrbbecSDK ROS 2 Wrapper](https://github.com/orbbec/OrbbecSDK_ROS2)
- [ConceptGraphs: Open-Vocabulary 3D Scene Graphs](https://concept-graphs.github.io/)
- [Hydra: Real-time 3D Scene Graph](https://github.com/MIT-SPARK/Hydra)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [ManiSkill](https://github.com/mani-skill/ManiSkill)
- [Isaac Sim ROS 2 文档](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/ros2_landing_page.html)

