# 电脑摄像头访客感知原型

该原型验证未来机器人 Agent 的一条基础闭环：**视频流 → 人体追踪 → 进入/离开事件 → World Model 当前状态 → Memory 事件历史**。它独立于当前聊天主链路，不需要 ROS、Orbbec 或实体机器人。

## 运行

在项目根目录安装依赖后执行：

```bash
python -m person_presence
```

浏览器打开 `http://127.0.0.1:8090`。首次运行 `ultralytics` 可能下载 `yolov8s.pt` 权重；若已存在本地模型，设置 `PRESENCE_YOLO_MODEL` 为其路径。

## 当前行为

- 读取默认电脑摄像头（索引 `0`）；
- 用 YOLO 的 `person` 类和 ByteTrack 生成短期 `camera_track_*`；
- 同一轨迹连续出现 3 帧后产生 `person_entered`；
- 消失超过 3 秒产生 `person_exited`；
- 事件保存在 `data/person_presence.sqlite3`；
- `/world-model` 返回当前在场人员，`/events` 返回历史记录，`/video.mjpg` 返回调试画面。

## 将人员事件转为机器人主动对话

本原型不会把每一帧视频发送给 LLM。它只将 `person_entered` 事件和结构化事实（当前在场人数、追踪置信度、身份是否确认）放入提示词，由 LLM 生成一句简短招呼。视觉线程与 LLM 网络调用在不同线程中运行，后者不会阻塞追踪。

默认关闭。确认已经在 `.env` 配置可用模型 API 后，开启网页文字招呼：

```bash
PRESENCE_SOCIAL_ENABLED=true python -m person_presence
```

同时启用语音播报（会调用现有讯飞 TTS）：

```bash
PRESENCE_SOCIAL_ENABLED=true PRESENCE_SOCIAL_TTS=true python -m person_presence
```

- 默认 45 秒全局冷却，避免同一人或多人连续进入时不停讲话；可用 `PRESENCE_SOCIAL_COOLDOWN_SECONDS` 调整。
- LLM 只得到事件文字，不得到视频帧；它不能自行断言姓名、性别、年龄、外貌或情绪。
- 生成结果保存到同一个 SQLite 数据库的 `social_messages` 表，并可从 `/social/messages` 查询。
- 本阶段只在 `person_entered` 时生成招呼；`person_exited` 不主动播报。

默认橙色区域覆盖整个画面，意为“出现在摄像头前即进入”。若摄像头固定对准门口，可用归一化矩形限制区域：

```bash
PRESENCE_ZONE=0.2,0.1,0.6,0.8 python -m person_presence
```

四个数依次为 `x, y, width, height`，范围是 0 到 1。

## 重要边界

- `track_id` 只保证同一次连续出现在镜头中的稳定性；离开后再次出现不保证还是同一人。
- 本阶段不做姓名或人脸识别，也不保存人脸特征。第二阶段可在用户知情同意的前提下，用 ReID 或用户语音确认将短期轨迹关联到长期 `person_id`。
- 用相机全画面计数只能表示“出现/消失”；要严格统计门口方向进出，需要下一步增加虚拟线和轨迹方向判断。

## 验收方法

1. 站到镜头前，页面应出现绿色人框和 `camera_track_*`；
2. 连续可见约 3 帧后，`/events` 出现一次 `person_entered`；
3. 离开画面超过 3 秒，出现一次 `person_exited`；
4. 同一次站在镜头前不会重复写入多条进入事件；
5. 运行 `python -m unittest tests.test_person_presence`，验证状态机和 SQLite 基础逻辑。

## 下一步

1. 入口虚拟线 + 行进方向，区分真正进入与离开；
2. ReID 与用户确认，将 `camera_track_*` 映射为经授权的用户档案；
3. 将 `WorldState` 与事件适配成 ROS 2 Topic，替换 `WebcamSource` 为 Orbbec RGB-D 输入；
4. 将人体位置和用户交互事件接入机器人 Agent 的 World Model / Memory。
