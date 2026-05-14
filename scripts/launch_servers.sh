#!/usr/bin/env bash
# launch_servers.sh — 一键启动 LeRobot 硬件 server 进程
#
# 用法：
#   bash scripts/launch_servers.sh                   # 默认：只启 3 路 opencv_zmq
#   bash scripts/launch_servers.sh --with-hik        # 额外启动 hik_camera_server
#   bash scripts/launch_servers.sh --with-franka     # 额外启动 franka_interface_server（会 go_home）
#   bash scripts/launch_servers.sh --with-hik --with-franka
#
# 参数化（环境变量覆盖顶部默认值）：
#   OPENCV_DEVICES=0,2,4 OPENCV_PORTS=5000,5001,5002 bash scripts/launch_servers.sh
#
# 日志：各 server 输出重定向到 $LOG_DIR/<name>.log，主进程 tail -F 串流。
# 退出：Ctrl+C 一并杀掉所有子进程。

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# 顶部默认值（所有变量均可被同名环境变量覆盖）
# ─────────────────────────────────────────────────────────────

# 通用
BIND_ADDR="${BIND_ADDR:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-/tmp/lerobot_servers}"

# opencv_zmq：逗号分隔，三组需一一对应
OPENCV_DEVICES="${OPENCV_DEVICES:-2,4,6}"
OPENCV_PORTS="${OPENCV_PORTS:-4244,4245,4246}"
OPENCV_TOPICS="${OPENCV_TOPICS:-cam_top,cam_left,cam_right}"
OPENCV_WIDTH="${OPENCV_WIDTH:-640}"
OPENCV_HEIGHT="${OPENCV_HEIGHT:-480}"
OPENCV_FPS="${OPENCV_FPS:-30}"
OPENCV_WIRE_ENCODING="${OPENCV_WIRE_ENCODING:-jpeg}"
OPENCV_JPEG_QUALITY="${OPENCV_JPEG_QUALITY:-90}"

# hik_camera（需 --with-hik）
HIK_PORT="${HIK_PORT:-4243}"
HIK_TOPIC="${HIK_TOPIC:-hik}"
HIK_DEVICE_INDEX="${HIK_DEVICE_INDEX:-0}"
HIK_OUTPUT_WIDTH="${HIK_OUTPUT_WIDTH:-1280}"
HIK_OUTPUT_HEIGHT="${HIK_OUTPUT_HEIGHT:-720}"
HIK_FPS="${HIK_FPS:-30}"
HIK_WIRE_ENCODING="${HIK_WIRE_ENCODING:-jpeg}"
HIK_JPEG_QUALITY="${HIK_JPEG_QUALITY:-90}"
HIK_SERVER_DIR="${HIK_SERVER_DIR:-$(cd "$(dirname "$0")/../src/lerobot/cameras/hik_camera" && pwd)}"

# franka_interface_server（需 --with-franka）
FRANKA_CONDA_ENV="${FRANKA_CONDA_ENV:-polymetis-local}"
FRANKA_SERVER_PY="${FRANKA_SERVER_PY:-$(cd "$(dirname "$0")/.." && pwd)/src/lerobot/robots/franka/franka_interface_server.py}"

# ─────────────────────────────────────────────────────────────
# 命令行 flag 解析
# ─────────────────────────────────────────────────────────────
ENABLE_HIK=0
ENABLE_FRANKA=0

for arg in "$@"; do
    case "$arg" in
        --with-hik)    ENABLE_HIK=1    ;;
        --with-franka) ENABLE_FRANKA=1 ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $arg  (--with-hik / --with-franka)" >&2
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
PIDS=()

log()   { echo "[$(date '+%H:%M:%S')] $*"; }
err()   { echo "[$(date '+%H:%M:%S')] [ERROR] $*" >&2; }

cleanup() {
    log "收到退出信号，关闭所有子进程…"
    # 先杀记录的 PID，再 kill 0（清理所有同进程组的子孙进程）
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    kill 0 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

# IFS 分割逗号列表成数组
split_csv() {
    IFS=',' read -ra _arr <<< "$1"
    echo "${_arr[@]}"
}

# 检查命令是否存在
require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { err "缺少命令: $1"; exit 1; }
}

# 启动后台进程，记录 PID
start_bg() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    log "启动 $name  →  $logfile"
    "$@" >"$logfile" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    log "  PID=$pid"
}

# ─────────────────────────────────────────────────────────────
# 前置检查
# ─────────────────────────────────────────────────────────────
require_cmd python

mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────
# 启动 opencv_zmq server（默认总是启动）
# ─────────────────────────────────────────────────────────────
log "━━━ OpenCV ZMQ 相机 Server ━━━"

# 解析三个逗号列表为 bash 数组
IFS=',' read -ra _devs   <<< "$OPENCV_DEVICES"
IFS=',' read -ra _ports  <<< "$OPENCV_PORTS"
IFS=',' read -ra _topics <<< "$OPENCV_TOPICS"

n_devs="${#_devs[@]}"
n_ports="${#_ports[@]}"
n_topics="${#_topics[@]}"

if [[ "$n_devs" -ne "$n_ports" || "$n_devs" -ne "$n_topics" ]]; then
    err "OPENCV_DEVICES / OPENCV_PORTS / OPENCV_TOPICS 元素数量不一致（${n_devs}/${n_ports}/${n_topics}）"
    exit 1
fi

for i in "${!_devs[@]}"; do
    dev="${_devs[$i]}"
    port="${_ports[$i]}"
    topic="${_topics[$i]}"

    # 若 dev 是纯数字，检查对应 /dev/videoN 是否存在
    if [[ "$dev" =~ ^[0-9]+$ ]]; then
        if [[ ! -e "/dev/video${dev}" ]]; then
            err "/dev/video${dev} 不存在，请检查 OPENCV_DEVICES 配置（当前: $OPENCV_DEVICES）"
            exit 1
        fi
    elif [[ ! -e "$dev" ]]; then
        err "设备路径 $dev 不存在"
        exit 1
    fi

    start_bg "opencv_zmq_dev${dev}" \
        python -m lerobot.cameras.opencv_zmq.opencv_zmq_server \
            --device   "$dev" \
            --bind     "$BIND_ADDR" \
            --port     "$port" \
            --topic    "$topic" \
            --width    "$OPENCV_WIDTH" \
            --height   "$OPENCV_HEIGHT" \
            --fps      "$OPENCV_FPS" \
            --wire-encoding   "$OPENCV_WIRE_ENCODING" \
            --jpeg-quality    "$OPENCV_JPEG_QUALITY"
done

# ─────────────────────────────────────────────────────────────
# 启动 hik_camera_server（--with-hik）
# ─────────────────────────────────────────────────────────────
if [[ "$ENABLE_HIK" -eq 1 ]]; then
    log "━━━ Hik Camera Server ━━━"

    if [[ ! -d "$HIK_SERVER_DIR" ]]; then
        err "HIK_SERVER_DIR 不存在: $HIK_SERVER_DIR"
        exit 1
    fi
    if [[ ! -f "$HIK_SERVER_DIR/hik_camera_server.py" ]]; then
        err "未找到 hik_camera_server.py：$HIK_SERVER_DIR"
        exit 1
    fi

    # hik_camera_server.py 使用裸 'from hik_camera_sdk import ...'，必须 cd 进目录运行
    start_bg "hik_camera" \
        bash -c "cd '$HIK_SERVER_DIR' && python hik_camera_server.py \
            --bind '$BIND_ADDR' \
            --port '$HIK_PORT' \
            --topic '$HIK_TOPIC' \
            --device-index '$HIK_DEVICE_INDEX' \
            --fps '$HIK_FPS' \
            --output-width '$HIK_OUTPUT_WIDTH' \
            --output-height '$HIK_OUTPUT_HEIGHT' \
            --wire-encoding '$HIK_WIRE_ENCODING' \
            --jpeg-quality '$HIK_JPEG_QUALITY'"
fi

# ─────────────────────────────────────────────────────────────
# 启动 franka_interface_server（--with-franka）
# ─────────────────────────────────────────────────────────────
if [[ "$ENABLE_FRANKA" -eq 1 ]]; then
    log "━━━ Franka Interface Server ━━━"
    log "⚠️  警告：启动 franka_interface_server 会自动执行 robot_go_home() 和 gripper_grasp(10, 0.1)"
    log "    请确认机械臂周围安全，无障碍物！"
    echo -n "    输入 yes 继续，或按 Ctrl+C 取消："
    read -r _confirm
    if [[ "$_confirm" != "yes" ]]; then
        log "已取消启动 Franka server。"
        ENABLE_FRANKA=0
    else
        if [[ ! -f "$FRANKA_SERVER_PY" ]]; then
            err "未找到 franka_interface_server.py：$FRANKA_SERVER_PY"
            exit 1
        fi

        # 在 polymetis conda 环境中启动
        CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
        if [[ ! -f "$CONDA_SH" ]]; then
            CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
        if [[ ! -f "$CONDA_SH" ]]; then
            err "找不到 conda.sh，请手动设置 CONDA_SH 环境变量"
            exit 1
        fi

        start_bg "franka_server" \
            bash -c "source '$CONDA_SH' && conda activate '$FRANKA_CONDA_ENV' && python '$FRANKA_SERVER_PY'"
    fi
fi

# ─────────────────────────────────────────────────────────────
# 等待并串流日志
# ─────────────────────────────────────────────────────────────
log "━━━ 所有 server 已在后台启动，按 Ctrl+C 停止 ━━━"
log "日志目录：$LOG_DIR"
log ""

# 简单轮询各进程是否存活（2 秒后首次检查）
sleep 2
alive=0
for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
    else
        err "进程 PID=$pid 已意外退出！检查日志：$LOG_DIR/"
    fi
done
log "$alive / ${#PIDS[@]} 个进程存活"

# tail -F 串流所有日志直到收到退出信号
# shellcheck disable=SC2046
tail -F $(ls "$LOG_DIR"/*.log 2>/dev/null) &
PIDS+=($!)

# 主循环：持续检查进程是否全部存活
while true; do
    sleep 10
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || true
    done
done
