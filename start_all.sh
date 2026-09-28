#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "启动 VL-RAG-System: 技心机器人全链路交互系统..."
echo "========================================"

start_service() {
    local name="$1"
    local command="$2"

    if command -v gnome-terminal >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
        gnome-terminal --title "VL-RAG $name" -- bash -lc "cd '$ROOT_DIR' && echo '$name 节点运行中...'; $command; exec bash"
    else
        nohup bash -lc "cd '$ROOT_DIR' && exec $command" >"$LOG_DIR/${name}.log" 2>&1 &
        echo "$!" >"$LOG_DIR/${name}.pid"
        echo "  $name 已后台启动，日志: $LOG_DIR/${name}.log"
    fi
}

echo "[1/3] 正在启动 ASR 语音识别服务..."
start_service "asr" "$PYTHON_BIN -u services/asr_service.py"

sleep 2

echo "[2/3] 正在启动 Vision 图像采集服务..."
start_service "vision" "$PYTHON_BIN -u services/vision_service.py"

sleep 2

echo "[3/3] 正在启动 Robot Brain 大脑处理器..."
start_service "brain" "$PYTHON_BIN -u local_model_processor.py"

sleep 2

echo ""
echo "全系统启动命令已发出。"
echo "核心日志: tail -f service.log"
echo "后台模式日志: tail -f logs/*.log"
echo ""
echo "提示:"
echo "  1. 请确认已加载 ROS 2 环境，例如: source /opt/ros/humble/setup.bash"
echo "  2. 请确认 Ollama 或 DeepSeek 配置可用"
echo "  3. Web 后端可另外运行: python3 main.py"
