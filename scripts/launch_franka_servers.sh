#!/usr/bin/env bash
#
# 一键启动 Franka 全栈服务（在连接机器臂的 NUC 上执行）：
#
#   1) launch_robot.py          — Polymetis gRPC robot controller-manager
#   2) launch_gripper.py        — Polymetis gRPC gripper server
#   3) franka_interface_server.py — zerorpc 服务（供 lerobot FrankaMultiCam 调用）
#
# 三个服务均在 polymetis-local conda 环境中运行。Ctrl-C 或脚本退出时，
# 会依次发送 INT → KILL 信号并强制清理残留进程。
# 任一子进程意外退出时，整个脚本随之退出。
#
# 用法：
#   ./scripts/launch_franka_servers.sh              # 使用默认参数
#   ./scripts/launch_franka_servers.sh use_real_time=false   # Hydra override 透传给 launch_robot.py
#
# 环境变量覆盖（示例）：
#   ROBOT_PORT=50051 GRIPPER_PORT=50052 FRANKA_PORT=4242 ./launch_franka_servers.sh
#
# 可用环境变量：
#   CONDA_ENV               conda 环境名                       (默认: polymetis-local)
#   ROBOT_IP                Polymetis robot server IP          (默认: 172.16.0.2)
#   ROBOT_PORT              Polymetis robot server 端口        (默认: 50051)
#   ROBOT_CLIENT            Hydra robot_client config          (默认: franka_hardware)
#   GRIPPER                 Hydra gripper config               (默认: franka_hand)
#   GRIPPER_IP              Polymetis gripper server IP        (默认: 与 ROBOT_IP 相同)
#   GRIPPER_PORT            Polymetis gripper server 端口      (默认: 50052)
#   STARTUP_TIMEOUT         等待 Polymetis 接口就绪的超时秒数  (默认: 90)
#   GRIPPER_HOMING_WAIT     夹爪 homing 完成等待秒数           (默认: 12)
#   FRANKA_BIND             zerorpc 绑定地址                   (默认: 192.168.172.134)
#   FRANKA_PORT             zerorpc 端口                       (默认: 4242)
#   FRANKA_IFACE_TIMEOUT    等待 zerorpc 端口就绪的超时秒数    (默认: 120)
#   FRANKA_GO_HOME          启动后是否执行 go_home (1/0)       (默认: 0)
#   FRANKA_INTERFACE_SERVER franka_interface_server.py 路径    (默认: 自动检测)
#   LOG_DIR                 日志目录                           (默认: /tmp/franka_stack)

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 配置（均可被同名环境变量覆盖）
# ─────────────────────────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-polymetis-local}"

ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
ROBOT_PORT="${ROBOT_PORT:-50051}"
ROBOT_CLIENT="${ROBOT_CLIENT:-franka_hardware}"

GRIPPER="${GRIPPER:-franka_hand}"
GRIPPER_IP="${GRIPPER_IP:-$ROBOT_IP}"
GRIPPER_PORT="${GRIPPER_PORT:-50052}"

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-90}"
GRIPPER_HOMING_WAIT="${GRIPPER_HOMING_WAIT:-12}"

FRANKA_BIND="${FRANKA_BIND:-192.168.172.134}"
FRANKA_PORT="${FRANKA_PORT:-4242}"
FRANKA_IFACE_TIMEOUT="${FRANKA_IFACE_TIMEOUT:-120}"
FRANKA_GO_HOME="${FRANKA_GO_HOME:-0}"

LOG_DIR="${LOG_DIR:-/tmp/franka_stack}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKA_INTERFACE_SERVER="${FRANKA_INTERFACE_SERVER:-$SCRIPT_DIR/../src/lerobot/robots/franka/franka_interface_server.py}"

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] [stack] $*"; }
err() { echo "[$(date '+%H:%M:%S')] [stack] [ERROR] $*" >&2; }

# 超时时打印最近 30 行日志并退出
die_with_log() {
    local msg="$1" logfile="${2:-}"
    err "$msg"
    if [[ -n "$logfile" && -f "$logfile" ]]; then
        err "━━━ 最近日志（${logfile}）━━━"
        tail -30 "$logfile" >&2
        err "━━━ 日志结束 ━━━"
    fi
    exit 1
}

ROBOT_PID=""
GRIPPER_PID=""
IFACE_PID=""

cleanup() {
    set +e +u
    log "正在关闭所有子进程..."
    for pid in "$IFACE_PID" "$GRIPPER_PID" "$ROBOT_PID"; do
        [ -n "$pid" ] && kill -INT "$pid" 2>/dev/null
    done
    sleep 2
    for pid in "$IFACE_PID" "$GRIPPER_PID" "$ROBOT_PID"; do
        [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null
    done
    # 清理 launch_robot.py 启动的 C++ 进程（setpgrp 子进程组）
    pkill -9 -f run_server          2>/dev/null
    pkill -9 -f franka_panda_client 2>/dev/null
    pkill -9 -f franka_hand_client  2>/dev/null
    log "已关闭。"
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# 前置检查
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "$FRANKA_INTERFACE_SERVER" ]]; then
    err "未找到 franka_interface_server.py: $FRANKA_INTERFACE_SERVER"
    err "请设置环境变量 FRANKA_INTERFACE_SERVER 指向正确路径。"
    exit 1
fi

mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 激活 conda 环境
# ─────────────────────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ] && [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$("$CONDA_EXE" info --base 2>/dev/null || true)"
fi
if [ -z "$CONDA_BASE" ]; then
    err "无法找到 conda，请初始化 conda（或导出 CONDA_EXE）后重试。"
    exit 1
fi
# conda 的 activate.d hooks 中包含不兼容 set -u/-e 的脚本，临时放宽限制
set +u +e
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u -e
log "conda 环境: ${CONDA_DEFAULT_ENV:-?}   python: $(command -v python)"

cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# launch_robot.py real-time 模式需要 sudo，提前缓存凭据
# ─────────────────────────────────────────────────────────────────────────────
log "缓存 sudo 凭据（launch_robot.py real-time 模式需要）..."
sudo -v

# ─────────────────────────────────────────────────────────────────────────────
# 等待 RobotInterface 真实可用（直接调用 get_joint_positions 验证）
#
# 背景：check_server_exists 只做 TCP/gRPC 握手，端口开着就返回 True，
# 但 RobotInterface(enforce_version=True) 需要发送完整 RPC 请求，
# 在 franka_panda_client 尚未向服务端注册完元数据时仍会失败。
# 用 enforce_version=False + get_joint_positions() 做真实可用性探测，
# 确认服务端能正常响应后再继续。
# ─────────────────────────────────────────────────────────────────────────────
wait_for_robot_interface() {
    local ip="$1" port="$2" timeout="$3" logfile="${4:-}" t0
    log "等待 RobotInterface 可用 (${ip}:${port}，超时 ${timeout}s) ..."
    t0=$(date +%s)
    until python - "$ip" "$port" 2>/dev/null <<'PY'
import sys
try:
    from polymetis import RobotInterface
    r = RobotInterface(ip_address=sys.argv[1], port=int(sys.argv[2]), enforce_version=False)
    _ = r.get_joint_positions()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    do
        sleep 1
        if [ $(( $(date +%s) - t0 )) -ge "$timeout" ]; then
            die_with_log "RobotInterface (${ip}:${port}) 在 ${timeout}s 内未就绪。" "$logfile"
        fi
    done
    log "RobotInterface 已就绪 ✓"
}

# ─────────────────────────────────────────────────────────────────────────────
# 等待 GripperInterface 真实可用（直接调用 get_state 验证）
# ─────────────────────────────────────────────────────────────────────────────
wait_for_gripper_interface() {
    local ip="$1" port="$2" timeout="$3" logfile="${4:-}" t0
    log "等待 GripperInterface 可用 (${ip}:${port}，超时 ${timeout}s) ..."
    t0=$(date +%s)
    until python - "$ip" "$port" 2>/dev/null <<'PY'
import sys
try:
    from polymetis import GripperInterface
    g = GripperInterface(ip_address=sys.argv[1], port=int(sys.argv[2]))
    _ = g.get_state()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    do
        sleep 1
        if [ $(( $(date +%s) - t0 )) -ge "$timeout" ]; then
            die_with_log "GripperInterface (${ip}:${port}) 在 ${timeout}s 内未就绪。" "$logfile"
        fi
    done
    log "GripperInterface 已就绪 ✓"
}

# ─────────────────────────────────────────────────────────────────────────────
# 等待 TCP 端口可连接（用于探测 zerorpc server 绑定完成）
# ─────────────────────────────────────────────────────────────────────────────
wait_for_tcp_port() {
    local port="$1" timeout="$2" name="$3" logfile="${4:-}" t0
    log "等待 ${name} (TCP :${port}，超时 ${timeout}s) ..."
    t0=$(date +%s)
    until python - "$port" 2>/dev/null <<'PY'
import sys, socket
try:
    s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    do
        sleep 0.5
        if [ $(( $(date +%s) - t0 )) -ge "$timeout" ]; then
            die_with_log "${name} 在 ${timeout}s 内未启动。" "$logfile"
        fi
    done
    log "${name} 已就绪 ✓"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1/3  Polymetis robot controller-manager server + franka_panda_client
# ─────────────────────────────────────────────────────────────────────────────
log "━━━ 1/3  启动 Polymetis robot server ━━━"
log "    robot_client=${ROBOT_CLIENT}  ip=${ROBOT_IP}  port=${ROBOT_PORT}"
log "    额外 Hydra overrides: $*"
log "    日志: ${LOG_DIR}/polymetis_robot.log"

launch_robot.py \
    robot_client="${ROBOT_CLIENT}" \
    ip="${ROBOT_IP}" \
    port="${ROBOT_PORT}" \
    "$@" \
    >"${LOG_DIR}/polymetis_robot.log" 2>&1 &
ROBOT_PID=$!

wait_for_robot_interface \
    "$ROBOT_IP" "$ROBOT_PORT" "$STARTUP_TIMEOUT" "${LOG_DIR}/polymetis_robot.log"

# ─────────────────────────────────────────────────────────────────────────────
# 2/3  Polymetis gripper server + franka_hand_client
# ─────────────────────────────────────────────────────────────────────────────
log "━━━ 2/3  启动 Polymetis gripper server ━━━"
log "    gripper=${GRIPPER}  ip=${GRIPPER_IP}  port=${GRIPPER_PORT}"
log "    日志: ${LOG_DIR}/polymetis_gripper.log"

launch_gripper.py \
    gripper="${GRIPPER}" \
    ip="${GRIPPER_IP}" \
    port="${GRIPPER_PORT}" \
    >"${LOG_DIR}/polymetis_gripper.log" 2>&1 &
GRIPPER_PID=$!

wait_for_gripper_interface \
    "$GRIPPER_IP" "$GRIPPER_PORT" "$STARTUP_TIMEOUT" "${LOG_DIR}/polymetis_gripper.log"

log "等待夹爪 homing 完成（${GRIPPER_HOMING_WAIT}s）..."
sleep "$GRIPPER_HOMING_WAIT"

# ─────────────────────────────────────────────────────────────────────────────
# 3/3  zerorpc FrankaInterfaceServer
# ─────────────────────────────────────────────────────────────────────────────
log "━━━ 3/3  启动 zerorpc FrankaInterfaceServer ━━━"
log "    bind=${FRANKA_BIND}  port=${FRANKA_PORT}"
log "    robot=${ROBOT_IP}:${ROBOT_PORT}  gripper=${GRIPPER_IP}:${GRIPPER_PORT}"
log "    日志: ${LOG_DIR}/franka_interface_server.log"

_go_home_flag=""
[[ "$FRANKA_GO_HOME" -eq 1 ]] && _go_home_flag="--go-home"

python "$FRANKA_INTERFACE_SERVER" \
    --bind         "$FRANKA_BIND" \
    --port         "$FRANKA_PORT" \
    --robot-ip     "$ROBOT_IP" \
    --robot-port   "$ROBOT_PORT" \
    --gripper-ip   "$GRIPPER_IP" \
    --gripper-port "$GRIPPER_PORT" \
    ${_go_home_flag} \
    >"${LOG_DIR}/franka_interface_server.log" 2>&1 &
IFACE_PID=$!

wait_for_tcp_port \
    "$FRANKA_PORT" "$FRANKA_IFACE_TIMEOUT" "FrankaInterfaceServer (zerorpc)" \
    "${LOG_DIR}/franka_interface_server.log"

# ─────────────────────────────────────────────────────────────────────────────
# 全部就绪，监控子进程存活状态
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  Franka 全栈服务已就绪："
log "    [1] Polymetis robot   PID=${ROBOT_PID}   :${ROBOT_PORT}"
log "    [2] Polymetis gripper PID=${GRIPPER_PID}  :${GRIPPER_PORT}"
log "    [3] zerorpc iface     PID=${IFACE_PID}   :${FRANKA_PORT}"
log "  日志目录: ${LOG_DIR}/"
log "  按 Ctrl-C 停止所有服务。"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 任一进程退出则整体退出（cleanup trap 会负责清理其余进程）
while true; do
    sleep 5
    for _pid in "$ROBOT_PID" "$GRIPPER_PID" "$IFACE_PID"; do
        if [[ -n "$_pid" ]] && ! kill -0 "$_pid" 2>/dev/null; then
            err "进程 PID=${_pid} 已意外退出！请检查日志：${LOG_DIR}/"
            exit 1
        fi
    done
done
