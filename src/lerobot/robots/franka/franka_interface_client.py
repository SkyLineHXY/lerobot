'''
running on the user machine
to connect to the franka_interface_server
'''
import logging
import time
import numpy as np
import zerorpc

log = logging.getLogger(__name__)


class FrankaInterfaceClient:
    def __init__(self, ip='192.168.31.169', port=4242):
        try:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect(f"tcp://{ip}:{port}")
            log.info(f"[INFO] Connected to FrankaInterfaceServer")
        except:
            log.error("Failed to connect to server")

    def robot_get_joint_positions(self):
        '''
        list -> np.ndarray
        '''
        joint_positions = np.array(self.server.robot_get_joint_positions())
        return joint_positions

    def robot_get_joint_velocities(self):
        '''
        list -> np.ndarray
        '''
        joint_velocities = np.array(self.server.robot_get_joint_velocities())
        return joint_velocities

    def robot_get_ee_pose(self):
        '''
        list -> np.ndarray
        '''
        pose = np.array(self.server.robot_get_ee_pose())
        return pose

    def robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
    ):
        self.server.robot_move_to_joint_positions(
            positions.tolist(),
            time_to_go,
            delta,
            Kq.tolist() if Kq is not None else None,
            Kqd.tolist() if Kqd is not None else None
        )

    def robot_go_home(self):
        self.server.robot_go_home()

    def robot_move_to_ee_pose(
        self,
        pose: np.ndarray = None,
        time_to_go: float = None,
        delta: bool = False,
        Kx: np.ndarray = None,
        Kxd: np.ndarray = None,
        op_space_interp: bool = True,
    ):
        self.server.robot_move_to_ee_pose(
            pose.tolist(),
            time_to_go,
            delta,
            Kx.tolist() if Kx is not None else None,
            Kxd.tolist() if Kxd is not None else None,
            op_space_interp,
        )

    def robot_start_joint_impedance_control(
        self,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
        adaptive: bool = True,
    ):
        self.server.robot_start_joint_impedance_control(
            Kq.tolist() if Kq is not None else None,
            Kqd.tolist() if Kqd is not None else None,
            adaptive,
        )
        print(f"[ROBOT] Joint impedance control started")

    def robot_start_cartesian_impedance_control(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.robot_start_cartesian_impedance_control(
            Kx.tolist() if Kx is not None else None,
            Kxd.tolist() if Kxd is not None else None,
        )
        print(f"[ROBOT] Cartesian impedance control started")

    def robot_update_desired_joint_positions(self, positions: np.ndarray):
        try:
            self.server.robot_update_desired_joint_positions(positions.tolist())
        except Exception as e:
            log.warning(f"[ROBOT] robot_update_desired_joint_positions failed: {e}")

    def robot_update_desired_ee_pose(self, pose: np.ndarray):
        try:
            self.server.robot_update_desired_ee_pose(pose.tolist())
        except Exception as e:
            log.warning(f"[ROBOT] robot_update_desired_ee_pose failed: {e}")

    def robot_terminate_current_policy(self):
        self.server.robot_terminate_current_policy()

    def gripper_get_state(self) -> dict:
        """获取夹爪当前状态"""
        return self.server.gripper_get_state()

    def gripper_goto(self, width: float, speed: float, force: float, blocking: bool = True):
        """控制夹爪移动到指定宽度"""
        self.server.gripper_goto(width, speed, force, blocking)

    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = False,
    ):
        """控制夹爪执行抓取动作"""
        self.server.gripper_grasp(speed, force, grasp_width, epsilon_inner, epsilon_outer, blocking)

    def close(self):
        self.server.close()

###############################################################################
# Test utilities
###############################################################################

def test_rpc_latency(client: FrankaInterfaceClient, n: int = 100):
    """测试单次 RPC 往返延迟 (get_ee_pose)"""
    print(f"\n{'='*60}")
    print(f"  RPC Latency Test  —  {n} calls to get_ee_pose()")
    print(f"{'='*60}")

    latencies = []
    for i in range(n):
        t0 = time.perf_counter()
        client.robot_get_ee_pose()
        latencies.append(time.perf_counter() - t0)

    latencies_ms = np.array(latencies) * 1000
    print(f"  samples : {n}")
    print(f"  mean    : {latencies_ms.mean():.2f} ms")
    print(f"  median  : {np.median(latencies_ms):.2f} ms")
    print(f"  std     : {latencies_ms.std():.2f} ms")
    print(f"  min     : {latencies_ms.min():.2f} ms")
    print(f"  max     : {latencies_ms.max():.2f} ms")
    print(f"  p95     : {np.percentile(latencies_ms, 95):.2f} ms")
    print(f"  p99     : {np.percentile(latencies_ms, 99):.2f} ms")
    print(f"  max achievable Hz : {1000 / latencies_ms.mean():.1f} Hz")
    return latencies_ms


def test_throughput(client: FrankaInterfaceClient, duration: float = 5.0):
    """在固定时间窗口内尽可能多地调用 get_ee_pose，测量吞吐量"""
    print(f"\n{'='*60}")
    print(f"  Throughput Test  —  {duration}s burst read")
    print(f"{'='*60}")

    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration:
        client.robot_get_ee_pose()
        count += 1
    elapsed = time.perf_counter() - t_start
    hz = count / elapsed
    print(f"  calls   : {count}")
    print(f"  elapsed : {elapsed:.2f} s")
    print(f"  rate    : {hz:.1f} Hz")
    return hz


def test_observation_consistency(client: FrankaInterfaceClient, n: int = 50, hz: float = 30):
    """以固定频率读取位姿，检查帧间跳变是否合理"""
    print(f"\n{'='*60}")
    print(f"  Observation Consistency  —  {n} samples @ {hz} Hz")
    print(f"{'='*60}")

    interval = 1.0 / hz
    poses = []
    timestamps = []
    for _ in range(n):
        t0 = time.perf_counter()
        poses.append(client.robot_get_ee_pose())
        timestamps.append(t0)
        dt = time.perf_counter() - t0
        if dt < interval:
            time.sleep(interval - dt)

    poses = np.array(poses)  # (n, 6)
    diffs = np.diff(poses, axis=0)
    pos_jumps = np.linalg.norm(diffs[:, :3], axis=1)  # m
    rot_jumps = np.linalg.norm(diffs[:, 3:], axis=1)  # rad

    print(f"  position jump  — max: {pos_jumps.max()*1000:.2f} mm,  mean: {pos_jumps.mean()*1000:.2f} mm")
    print(f"  rotation jump  — max: {np.degrees(rot_jumps.max()):.3f}°,  mean: {np.degrees(rot_jumps.mean()):.3f}°")

    bad = pos_jumps > 0.05  # >50mm 认为异常
    if bad.any():
        print(f"  WARNING: {bad.sum()} frame(s) with position jump > 50 mm!")
    else:
        print(f"  PASS: all position jumps < 50 mm")
    return poses


def test_action_tracking_joint(
    client: FrankaInterfaceClient,
    amplitude_deg: float = 5.0,
    freq: float = 0.25,
    duration: float = 8.0,
    control_hz: float = 30,
):
    """
    关节空间正弦跟随测试。
    在当前关节位置上对第 5 关节 (wrist) 施加小幅正弦扰动，
    实时记录目标 vs 实际关节角，评估跟踪误差。
    """
    print(f"\n{'='*60}")
    print(f"  Joint Action Tracking  —  joint 5 sine wave")
    print(f"  amplitude={amplitude_deg}°  freq={freq} Hz  duration={duration}s  control={control_hz} Hz")
    print(f"{'='*60}")

    home_q = client.robot_get_joint_positions()
    print(f"  home joints: {np.round(home_q, 4).tolist()}")

    amplitude_rad = np.radians(amplitude_deg)
    dt = 1.0 / control_hz
    target_joint = 4  # 0-indexed, 第5个关节

    client.robot_start_joint_impedance_control()
    time.sleep(0.3)

    log_t, log_cmd, log_actual = [], [], []
    t0 = time.perf_counter()

    try:
        while True:
            t = time.perf_counter() - t0
            if t >= duration:
                break

            target = home_q.copy()
            target[target_joint] += amplitude_rad * np.sin(2 * np.pi * freq * t)
            client.robot_update_desired_joint_positions(target)

            actual_q = client.robot_get_joint_positions()
            log_t.append(t)
            log_cmd.append(target[target_joint])
            log_actual.append(actual_q[target_joint])

            elapsed = time.perf_counter() - t0 - t
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        client.robot_terminate_current_policy()

    log_t = np.array(log_t)
    errors = np.abs(np.array(log_cmd) - np.array(log_actual))
    errors_deg = np.degrees(errors)

    print(f"  samples       : {len(log_t)}")
    print(f"  tracking error (deg):")
    print(f"    mean  : {errors_deg.mean():.3f}°")
    print(f"    max   : {errors_deg.max():.3f}°")
    print(f"    std   : {errors_deg.std():.3f}°")
    print(f"  effective Hz  : {len(log_t) / log_t[-1]:.1f}")
    return log_t, np.array(log_cmd), np.array(log_actual)


def test_action_tracking_cartesian(
    client: FrankaInterfaceClient,
    amplitude_m: float = 0.03,
    freq: float = 0.25,
    duration: float = 8.0,
    control_hz: float = 30,
    axis: int = 1,
):
    """
    笛卡尔空间正弦跟随测试。
    在当前末端位姿上沿指定轴 (默认 Y) 施加小幅正弦扰动，
    记录目标 vs 实际 EE 位置，评估跟踪误差。
    """
    axis_name = ["X", "Y", "Z"][axis]
    print(f"\n{'='*60}")
    print(f"  Cartesian Action Tracking  —  {axis_name}-axis sine wave")
    print(f"  amplitude={amplitude_m*1000:.0f} mm  freq={freq} Hz  duration={duration}s  control={control_hz} Hz")
    print(f"{'='*60}")

    home_pose = client.robot_get_ee_pose()
    print(f"  home pose: {np.round(home_pose, 4).tolist()}")

    dt = 1.0 / control_hz

    client.robot_start_cartesian_impedance_control(
        Kx=np.array([400, 400, 400, 30, 30, 30]),
        Kxd=np.array([40, 40, 40, 5, 5, 5]),
    )
    time.sleep(0.3)

    log_t, log_cmd, log_actual = [], [], []
    t0 = time.perf_counter()

    try:
        while True:
            t = time.perf_counter() - t0
            if t >= duration:
                break

            target = home_pose.copy()
            target[axis] += amplitude_m * np.sin(2 * np.pi * freq * t)
            client.robot_update_desired_ee_pose(target)

            actual = client.robot_get_ee_pose()
            log_t.append(t)
            log_cmd.append(target[axis])
            log_actual.append(actual[axis])

            elapsed = time.perf_counter() - t0 - t
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        client.robot_terminate_current_policy()

    log_t = np.array(log_t)
    errors_mm = np.abs(np.array(log_cmd) - np.array(log_actual)) * 1000

    print(f"  samples       : {len(log_t)}")
    print(f"  tracking error (mm):")
    print(f"    mean  : {errors_mm.mean():.2f} mm")
    print(f"    max   : {errors_mm.max():.2f} mm")
    print(f"    std   : {errors_mm.std():.2f} mm")
    print(f"  effective Hz  : {len(log_t) / log_t[-1]:.1f}")
    return log_t, np.array(log_cmd), np.array(log_actual)


def save_tracking_csv(filepath, log_t, log_cmd, log_actual, label="value"):
    """保存跟踪日志到 CSV，方便后续用 matplotlib 绘图分析"""
    import csv
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", f"cmd_{label}", f"actual_{label}", f"error_{label}"])
        for t, c, a in zip(log_t, log_cmd, log_actual):
            writer.writerow([f"{t:.4f}", f"{c:.6f}", f"{a:.6f}", f"{abs(c-a):.6f}"])
    print(f"  saved to {filepath}")


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Franka Interface Client — Communication & Tracking Tests")
    parser.add_argument("--ip", default="172.20.10.2", help="NUC IP address")
    parser.add_argument("--port", type=int, default=4242, help="zerorpc port")
    parser.add_argument("--test", default="all",
                        choices=["all", "latency", "throughput", "consistency",
                                 "track_joint", "track_cartesian"],
                        help="Which test to run")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="Sine amplitude (deg for joint, mm for cartesian)")
    parser.add_argument("--freq", type=float, default=0.25, help="Sine frequency (Hz)")
    parser.add_argument("--duration", type=float, default=8.0, help="Tracking test duration (s)")
    parser.add_argument("--hz", type=float, default=30, help="Control loop rate (Hz)")
    parser.add_argument("--save-csv", action="store_true", help="Save tracking results to CSV")
    args = parser.parse_args()

    print(f"\nConnecting to NUC at {args.ip}:{args.port} ...")
    client = FrankaInterfaceClient(ip=args.ip, port=args.port)
    print(f"  joint positions: {np.round(client.robot_get_joint_positions(), 4).tolist()}")
    print(f"  ee pose:         {np.round(client.robot_get_ee_pose(), 4).tolist()}")
    print(f"  gripper state:   {client.gripper_get_state()}")
    client.robot_go_home()
    client.gripper_grasp(speed=50.0, force=0.1)
    time.sleep(5)
    print(f"  gripper state:   {client.gripper_get_state()}")
    client.close()
