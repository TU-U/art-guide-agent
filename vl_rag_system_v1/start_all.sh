#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "启动 VL-RAG-System: 技心机器人全链路交互系统..."
echo "========================================"

stop_existing_service() {
    local name="$1"
    local pid_file="$LOG_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            echo "  [INFO] 停止旧的 $name 服务: $old_pid"
            kill "$old_pid" 2>/dev/null || true
            sleep 1
            kill -9 "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

# 后台启动服务的函数（用于 驱动 和 拾音）
start_backend_service() {
    local name="$1"
    local command="$2"

    stop_existing_service "$name"

    # 将 Log 丢进你规划的 logs/ 目录，不污染主终端
    nohup bash -c "$command" >"$LOG_DIR/${name}.log" 2>&1 &
    echo "$!" >"$LOG_DIR/${name}.pid"
    echo "  [OK] $name 服务已在后台打开。"
}

# [1/6] 启动音频驱动服务
echo "[1/6] 正在启动 DRIVER 音频驱动服务..."
# 拼接 ros2 的环境变量加载与 launch 命令
DRIVER_CMD=". ~/audio_ros2/install/setup.bash && ros2 launch xunfei_dev_socket xunfei_dev_all.launch.py"
start_backend_service "driver" "$DRIVER_CMD"

sleep 2

# [2/6] 启动头部深度相机服务
echo "[2/6] 正在启动 CAMERA 头部深度相机服务..."
CAMERA_CMD="cd /home/nvidia/orbbec_camera_ros2 && source install/setup.bash && ros2 launch orbbec_camera gemini_330_series.launch.py"
start_backend_service "camera" "$CAMERA_CMD"

sleep 2

# [3/6] 启动手势指向识别视频服务
echo "[3/6] 正在启动 POINT 指向识别服务..."
POINT_CMD="cd point && $PYTHON_BIN -u hand_stream.py"
start_backend_service "point" "$POINT_CMD"

sleep 2

# [4/6] 启动 Web 前端后端服务
echo "[4/6] 正在启动 WEB 前端后端服务..."
WEB_CMD="source venv/bin/activate && $PYTHON_BIN -u main.py"
start_backend_service "web" "$WEB_CMD"
echo "  [INFO] 前端入口: http://192.168.31.43:8765/robot"

sleep 2

# [5/6] 启动 ASR 语音识别服务
echo "[5/6] 正在启动 ASR 语音识别服务..."
# 进入 venv 环境并运行
ASR_CMD="source venv/bin/activate && $PYTHON_BIN -u voice_to_text.py"
start_backend_service "asr" "$ASR_CMD"

sleep 2

# [6/6] 启动 Robot Brain 大脑处理器 (在前台运行，直接输出 Log)
echo "[6/6] 正在启动 Robot Brain 大脑处理器..."
echo "----------------------------------------"
echo " -> 以下为 Brain 服务的实时 Log 输出："
echo "========================================"

# 激活环境并直接前台 exec 执行，接管当前终端的输出
source venv/bin/activate
exec $PYTHON_BIN -u local_model_processor.py
