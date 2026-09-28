# VL-RAG-System 后端 API 设计文档

## 1. 系统基础 (System) — `api/routers/system.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | /api/status | 服务器及 ROS 节点健康检查 |
| GET | /api/health | 详细的服务依赖状态报告 |
| POST | /api/system/reset | 重置所有缓存及本地调度状态 |

---

## 2. 交互与对话 (Interaction) — `api/routers/chat.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/chat/multimodal | 提交图文混合请求。Body: {image: file, question: str} |
| GET | /api/chat/stream | 流式获取机器人实时回复内容 |
| POST | /api/chat/interrupt | 强制打断当前机器人的语音输出与推理 |

---

## 3. 检索系统 (Retrieval) — `api/routers/rag.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/rag/search | 多阶段RAG检索（向量检索 + 混合检索 + 重排序） |
| POST | /api/rag/rerank | 对候选结果进行重排序 |
| POST | /api/rag/ingest | 向知识库注入结构化数据（支持chunk） |
| GET | /api/rag/stats | 获取知识库统计信息 |

---

## 4. 记忆系统 (Memory) — `api/routers/memory.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | /api/memory/history | 获取当前会话的对话历史列表 |
| DELETE | /api/memory/clear | 清除当前用户的短时记忆缓存 |
| POST | /api/memory/summary | 触发生成当前长对话的阶段性摘要 |

---

## 5. 共鸣与 Agent (Resonance & Agent) — `api/routers/agent.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | /api/agent/state | 获取机器人当前任务规划及情感共鸣状态 |
| POST | /api/agent/vibe/set | 手动设定机器人的共鸣阈值。Body: {resonance: float} |
| GET | /api/agent/plan | 获取当前任务的 Step-by-Step 逻辑链 |
| POST | /api/agent/register | 注册新的Agent实例。Body: {agent_name: str, agent_config: object} |
| POST | /api/agent/execute | 执行指定Agent的任务。Body: {agent_name: str, task: object, context: object} |
| GET | /api/agent/status/{agent_name} | 获取指定Agent的当前状态 |
| POST | /api/agent/coordinate | 协调多个Agent共同完成复杂任务。Body: {task: object, context: object} |

---

## 6. 共鸣引擎 (Resonance) — `api/routers/resonance.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/resonance/calculate | 计算情感共鸣分值。Body: {text_input: str, user_profile: object} |
| POST | /api/resonance/filter | 应用人格化滤镜。Body: {raw_response: str, context: object} |
| GET | /api/resonance/config | 获取当前人设配置 |
| POST | /api/resonance/config | 更新人设配置。Body: {config: object} |

---

## 7. 听觉系统 (Hearing) — `api/routers/asr.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/asr/start | 开始语音监听 |
| POST | /api/asr/stop | 停止语音监听并返回识别结果 |
| POST | /api/asr/config | 设置ASR配置。Body: {api_key: str, mode: str} |
| POST | /api/asr/mode | 切换识别模式。Body: {mode: str} |
| GET | /api/asr/status | 获取当前ASR服务状态 |

---

## 8. 语言系统 (Language) — `api/routers/llm.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/llm/generate | 生成文本响应。Body: {prompts: array, history: array, image: file} |
| POST | /api/llm/stream | 流式生成文本响应。Body: {prompts: array, history: array, image: file} |
| POST | /api/llm/config | 设置DeepSeek模型配置。Body: {model_path: str, parameters: object} |
| GET | /api/llm/status | 获取当前模型状态与资源使用情况 |
| POST | /api/llm/load | 加载指定配置的模型。Body: {model_config: object} |

---

## 9. 视觉系统 (Vision) — `api/routers/vision.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/vision/capture | 捕获当前摄像头画面 |
| GET | /api/vision/latest | 获取最新捕获的画面 |
| POST | /api/vision/analyze | 分析图像内容。Body: {image: file} |

---

## 10. 语音输出 (TTS) — `api/routers/tts.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/tts/speak | 合成并播放语音。Body: {text: str, voice: str} |
| POST | /api/tts/config | 设置TTS配置。Body: {voice: str, speed: float} |
| GET | /api/tts/status | 获取当前TTS服务状态 |

---

## 11. 错误处理与响应格式

### 成功响应
```json
{
  "status": "success",
  "data": {...},
  "message": "操作成功"
}
```

### 错误响应
```json
{
  "status": "error",
  "code": 400,
  "message": "错误信息"
}
```

### 流式响应
```json
{
  "status": "streaming",
  "data": {
    "chunk": "当前生成的文本片段",
    "finish": false
  }
}
```

---

## 12. 认证与安全

- API访问控制：基于API密钥的认证机制
- 速率限制：防止恶意请求和资源滥用
- 数据加密：敏感数据传输加密
- 输入验证：所有用户输入的严格验证

---

## 13. 部署与扩展

- 容器化部署：支持Docker容器部署
- 水平扩展：支持多实例负载均衡
- 监控与日志：集成Prometheus和ELK堆栈
- 配置管理：环境变量和配置文件分离