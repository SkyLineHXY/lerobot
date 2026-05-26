"""
Test script for the Franka lerobot robot interface.
Assumes franka_interface_server.py is already running.

Usage:
    cd src/lerobot/robots/franka
    python test_franka.py [--ip 192.168.1.104]
"""

import sys
import time
import argparse
import logging
import numpy as np
import types
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def record(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_properties(robot):
    section("1. Static Properties")

    obs_ft = robot.observation_features
    record("observation_features returns dict", isinstance(obs_ft, dict))
    expected_obs_keys = {"ee_pose.x", "ee_pose.y", "ee_pose.z", "ee_pose.rx", "ee_pose.ry", "ee_pose.rz"}
    record("observation_features keys correct", set(obs_ft.keys()) == expected_obs_keys, str(set(obs_ft.keys())))

    act_ft = robot.action_features
    record("action_features returns dict", isinstance(act_ft, dict))
    expected_act_keys = {"ee_pose.x", "ee_pose.y", "ee_pose.z", "ee_pose.rx", "ee_pose.ry", "ee_pose.rz"}
    record("action_features keys correct", set(act_ft.keys()) == expected_act_keys, str(set(act_ft.keys())))

    record("is_calibrated == is_connected", robot.is_calibrated == robot.is_connected)
    record("name == 'franka'", robot.name == "franka")


def test_connect(robot):
    section("2. connect()")
    try:
        robot.connect()
        record("connect() succeeded", robot.is_connected)
    except Exception as e:
        record("connect() succeeded", False, str(e))
        return False
    return True


def test_get_observation(robot):
    section("3. get_observation()")
    try:
        obs = robot.get_observation()
        record("get_observation() returns dict", isinstance(obs, dict))
        expected_keys = {"ee_pose.x", "ee_pose.y", "ee_pose.z", "ee_pose.rx", "ee_pose.ry", "ee_pose.rz"}
        record("observation keys correct", set(obs.keys()) == expected_keys)
        all_float = all(isinstance(v, float) for v in obs.values())
        record("all observation values are float", all_float)
        # Sanity-check ranges
        pos_range_ok = all(abs(obs[k]) < 5.0 for k in ["ee_pose.x", "ee_pose.y", "ee_pose.z"])
        rot_range_ok = all(abs(obs[k]) <= np.pi + 0.01 for k in ["ee_pose.rx", "ee_pose.ry", "ee_pose.rz"])
        record("position values in plausible range (< 5 m)", pos_range_ok, str({k: round(obs[k], 4) for k in ["ee_pose.x", "ee_pose.y", "ee_pose.z"]}))
        record("rotation values in [-π, π]", rot_range_ok, str({k: round(obs[k], 4) for k in ["ee_pose.rx", "ee_pose.ry", "ee_pose.rz"]}))
        print(f"\n  Current EE pose:")
        for k, v in obs.items():
            print(f"    {k}: {v:.6f}")
        return obs
    except Exception as e:
        record("get_observation() succeeded", False, str(e))
        return None


def test_repeated_observation(robot, n=10, hz=10):
    section(f"4. Repeated get_observation() — {n} samples @ {hz} Hz")
    interval = 1.0 / hz
    obs_list = []
    try:
        for i in range(n):
            obs = robot.get_observation()
            obs_list.append(obs)
            time.sleep(interval)
        record("all samples succeeded", len(obs_list) == n, f"{len(obs_list)}/{n}")
        # Check consistency: positions shouldn't jump wildly between samples
        xs = [o["ee_pose.x"] for o in obs_list]
        max_jump = max(abs(xs[i+1] - xs[i]) for i in range(len(xs)-1))
        record("no wild position jumps (< 0.5 m between samples)", max_jump < 0.5, f"max Δx={max_jump:.4f} m")
    except Exception as e:
        record("repeated observation succeeded", False, str(e))


def test_calibrate_configure(robot):
    section("5. calibrate() and configure() (no-ops)")
    try:
        robot.calibrate()
        record("calibrate() runs without error", True)
    except Exception as e:
        record("calibrate() runs without error", False, str(e))
    try:
        robot.configure()
        record("configure() runs without error", True)
    except Exception as e:
        record("configure() runs without error", False, str(e))


def test_send_action_stub(robot, current_obs):
    section("6. send_action() — stub / not-implemented check")
    if current_obs is None:
        print(f"  [{SKIP}] No observation available, skipping send_action test")
        results.append(("send_action() stub", SKIP, "no observation"))
        return

    action = dict(current_obs)  # send current pose as action (zero delta)
    try:
        ret = robot.send_action(action)
        # send_action currently returns None (TODO in implementation)
        record("send_action() does not raise", True, f"returned: {ret}")
    except NotImplementedError:
        record("send_action() raises NotImplementedError (expected TODO)", True)
    except Exception as e:
        record("send_action() does not raise unexpected exception", False, str(e))


def test_reset(robot, do_reset: bool):
    section("7. reset() — moves robot to home")
    if not do_reset:
        print(f"  [{SKIP}] Skipped (pass --reset to enable)")
        results.append(("reset()", SKIP, "use --reset flag to enable"))
        return
    try:
        print("  Moving to home position — this may take a few seconds...")
        robot.reset()
        record("reset() succeeded", True)
    except Exception as e:
        record("reset() succeeded", False, str(e))


def test_disconnect(robot):
    section("8. disconnect()")
    try:
        robot.disconnect()
        record("disconnect() succeeded", not robot.is_connected)
    except Exception as e:
        record("disconnect() succeeded", False, str(e))

    # Second disconnect should be a no-op, not raise
    try:
        robot.disconnect()
        record("second disconnect() is safe (no-op)", True)
    except Exception as e:
        record("second disconnect() is safe (no-op)", False, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    section("SUMMARY")
    pass_count = sum(1 for _, s, _ in results if "PASS" in s)
    fail_count = sum(1 for _, s, _ in results if "FAIL" in s)
    skip_count = sum(1 for _, s, _ in results if "SKIP" in s)
    for name, status, detail in results:
        print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    print(f"\n  Total: {pass_count} passed, {fail_count} failed, {skip_count} skipped")
    return fail_count == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Franka lerobot interface test")
    parser.add_argument("--ip", default="192.168.31.169", help="Franka interface server IP")
    parser.add_argument("--port", type=int, default=4242, help="Franka interface server port")
    parser.add_argument("--reset", action="store_true",default=True, help="Run reset() test (moves robot to home)")
    args = parser.parse_args()

    # Import here so path issues surface early
    from config_franka import FrankaConfig
    from franka import Franka

    print(f"\nFranka Interface Test")
    print(f"Server: {args.ip}:{args.port}")
    print(f"Reset test: {'enabled' if args.reset else 'disabled (--reset to enable)'}")

    config = FrankaConfig(robot_ip=args.ip)
    robot = Franka(config=config)

    # Run tests
    test_properties(robot)

    connected = test_connect(robot)
    if not connected:
        logger.error("Cannot connect to robot — aborting remaining tests.")
        print_summary()
        sys.exit(1)

    current_obs = test_get_observation(robot)
    test_repeated_observation(robot)
    test_calibrate_configure(robot)
    test_send_action_stub(robot, current_obs)
    test_reset(robot, do_reset=args.reset)
    test_disconnect(robot)

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
