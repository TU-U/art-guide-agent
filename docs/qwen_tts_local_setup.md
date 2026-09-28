# Qwen TTS 本地模型接入

本项目已经默认按本地目录加载 Qwen TTS 模型：

- provider: `qwen`
- 模型仓库: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- 模型目录: `models/Qwen3-TTS-0.6B-CustomVoice`
- 当前默认音色: `Vivian`

## 1. 下载模型到本地目录

推荐把模型完整下载到下面这个目录：

```text
D:\OpenResource\vl-rag-system\models\Qwen3-TTS-0.6B-CustomVoice
```

如果你本机能访问 Hugging Face，可以在项目根目录执行：

```powershell
.\\venv\\Scripts\\Activate.ps1
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir .\\models\\Qwen3-TTS-0.6B-CustomVoice
```

如果你是手动下载，请确保模型目录下至少包含这类 Hugging Face 模型文件：

```text
config.json
generation_config.json
preprocessor_config.json
tokenizer_config.json
tokenizer.json / merges.txt / vocab.json
*.safetensors
```

## 2. 检查环境变量

当前 `.env` 已配置：

```env
TTS_PROVIDER=qwen
QWEN_TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
QWEN_TTS_MODEL_PATH=models/Qwen3-TTS-0.6B-CustomVoice
QWEN_TTS_VOICE=Vivian
QWEN_TTS_LANGUAGE=zh
```

如果你把模型放到别的目录，只需要改 `QWEN_TTS_MODEL_PATH`。

## 3. 启动后端并测试

启动服务：

```powershell
.\\venv\\Scripts\\Activate.ps1
python main.py
```

查看 TTS 状态：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/api/tts/status | Select-Object -ExpandProperty Content
```

合成一段测试音频：

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8765/api/tts/synthesize `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"你好，欢迎来到展厅。"}' `
  -OutFile .\\data\\audio_out\\tts_test.wav
```

## 4. 当前已知限制

- `qwen-tts` 在 Windows 上会提示系统缺少 `sox`，目前看是告警，不一定阻塞加载。
- 首次加载模型会比较慢。
- 你当前机器人链路仍是“LLM 完整生成后，再分句做 TTS”，首句延迟的主要瓶颈还在 `LLMService.generate_response_stream()` 不是真流式。
