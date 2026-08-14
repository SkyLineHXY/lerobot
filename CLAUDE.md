# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a fork of LeRobot (`origin` = `SkyLineHXY/lerobot`) with two substantial in-house additions
on top of upstream: **`rlt/` online reinforcement learning** and the **Piper / DualPiper
leader-follower teleoperation data collection pipeline**. When working in either area, read the
corresponding section below first — it records constraints learned by debugging that are not
visible from the code alone.

## Common Commands

```bash
# Install
pip install -e ".[dev,test]"
pip install -e ".[smolvla,aloha,pusht]"               # policy extras
pip install -e ".[hopejr,lekiwi,unitree_g1,reachy2]"  # robot extras
pip install -e ".[all]"                               # everything except groot / unitree_g1

# Code quality
pre-commit run --all-files
pre-commit run --files <path1> <path2>            # only the changed files — much faster
pre-commit install

# Tests
git lfs install && git lfs pull                   # without LFS, artifact-based tests fail
pytest -sv ./tests
pytest -sv tests/scripts/test_record_piper.py::test_name    # a single test
pytest -q tests/rlt tests/scripts                 # by directory; tests/ mirrors src/lerobot/

# End-to-end train / eval (Makefile)
make test-end-to-end DEVICE=cpu
make test-act-ete-train DEVICE=cpu

# Train / eval
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
lerobot-eval  --policy.path=<checkpoint_dir>/pretrained_model --env.type=aloha
lerobot-train --config_path=<output_dir>/checkpoints/000002/pretrained_model/train_config.json --resume=true

# Other CLIs
lerobot-record / teleoperate / calibrate / replay / find-cameras / find-port / setup-can
lerobot-dataset-viz / edit-dataset / train-tokenizer / info
```

### Piper data collection and replay (fork-specific)

```bash
# Bring the CAN interfaces up after power-on. Naming: _l = leader arm, _f = follower arm
./can_config.sh          # rename + activate CAN by physical USB port; ./find_all_can_port.sh to inspect

# Collection. On the first hardware run always start with dry_run + init_mode=none (reads only, sends nothing)
lerobot-record-piper --config_path examples/piper/record_piper.yaml \
    --collection.dry_run=true --collection.init_mode=none
lerobot-record-piper --config_path examples/piper/record_piper.yaml       # single arm
lerobot-record-piper --config_path examples/piper/record_dual_piper.yaml  # dual arm

# Replay: plot state/action curves, stitch multi-camera video, optionally replay on hardware
lerobot-replay-piper --root=<dataset_root> --repo_id=<repo_id> --episode=0

# Teleop link health check (writes no dataset; only measures where each tick goes)
python -m lerobot.rlt.teleop_check --config_path examples/rlt/teleop_check.yaml --dry_run=true

# RLT online RL
lerobot-rlt-train-token / lerobot-rlt-train-online / lerobot-rlt-eval-token
python -m lerobot.rlt.train_online --config_path examples/rlt/mock_online.yaml   # hardware-free smoke run
```

## Architecture

### Package layout (`src/lerobot/`)

| Module | Purpose |
|---|---|
| `policies/` | ML policies (ACT, Diffusion, SmolVLA, Pi0, TDMPC, VQBeT, RLT, …) |
| `configs/` | draccus dataclass configuration system |
| `datasets/` | LeRobotDataset — Parquet + MP4, HF Hub integration; `repair.py` for crash recovery |
| `robots/` | Physical robot hardware abstraction layer |
| `motors/` | Low-level motor bus drivers (Feetech, Dynamixel, Damiao, Robstride, Piper) |
| `cameras/` | Camera drivers (OpenCV, RealSense, ZMQ, HIK) |
| `teleoperators/` | Teleop devices (SO-100/101 leader, Piper leader, gamepad, keyboard, phone) |
| `scripts/` | CLI entry points |
| `rlt/` | **Everything around online RL**: training loops, chunk replay buffer, envs, human intervention (the model itself lives in `policies/rlt/`) |
| `envs/` | Simulation environments (Aloha, PushT, LIBERO, MetaWorld) |
| `processor/` | Observation → policy-input pre/post-processing pipelines |
| `rl/` | RL utilities (SAC, online buffers, W&B logging) |
| `transport/` | Async inference / gRPC transport |
| `async_inference/` | Async inference service |
| `model/` | Shared model architecture components |
| `utils/` | General utilities; `status_view.py` is the OpenCV operator view |

### Policies

| Policy | Type | Notes |
|---|---|---|
| `act` | Imitation | Action Chunking with Transformers |
| `diffusion` | Imitation | Diffusion Policy |
| `tdmpc` | Model-predictive control | Temporal Difference MPC |
| `vqbet` | Imitation | VQ-BeT |
| `sac` | RL | Soft Actor-Critic |
| `smolvla` | VLA | SmolVLM vision-language-action model |
| `pi0` / `pi05` / `pi0_fast` | VLA | π0 family |
| `groot` | VLA | Gr00t N1 |
| `wall_x` / `xvla` / `sarm` / `rtc` | VLA / real-time control | Others |

### Robots

`so100_follower` / `so101_follower` / `bi_so100_follower` / `koch_follower` / `lekiwi` / `piper` / `dual_piper` / `hope_jr` / `franka` / `franka_gen_gripper` / `gen_gripper` / `earthrover_mini_plus` / `reachy2` / `unitree_g1`

### Key interfaces

**Policies**: subclass `PreTrainedPolicy` (`policies/pretrained.py`). Must define `config_class`,
`name`, `forward()` (training) and `select_action()` (inference). HF Hub integration comes from
`HubMixin`; a checkpoint is `model.safetensors` + `config.json`.

**Robots**: subclass `Robot` (`robots/robot.py`). Must define `observation_features`,
`action_features`, `connect()`, `disconnect()`, `get_observation()`, `send_action()`. Calibration
files live in `~/.cache/huggingface/lerobot/calibration/`.

**Configuration**: draccus dataclasses. CLI flags like `--policy.type=act` select registered
config/model classes. Each policy directory holds `configuration_<name>.py` and `modeling_<name>.py`.

> ⚠️ **Never add `from __future__ import annotations` to an entry-point script.**
> `lerobot.configs.parser.wrap()` reads `inspect.getfullargspec(fn).annotations` directly. With
> PEP 563 enabled, draccus receives strings instead of dataclass types and config parsing fails in
> a very confusing way. Both `teleop_check.py` and `lerobot_record_piper.py` carry a comment about
> this at the top — do not "fix" the missing import.

**Datasets**: Parquet (state/action) + MP4 (images), with HF Hub streaming. `configs/types.py`
defines the `FeatureType` enum (`STATE`, `VISUAL`, `ENV`, `ACTION`, `REWARD`, `LANGUAGE`).

---

## Hard constraints when writing a LeRobot v3 dataset

These apply when assembling frames yourself instead of going through `lerobot-record`. Every one of
them has bitten during real debugging.

**`add_frame(frame)` validation** (`datasets/utils.py::validate_frame`):
- `task` is **a key inside the frame dict**, not a separate function argument.
- The key set must be **exactly equal** to `set(features) - DEFAULT_FEATURES` — no more, no less.
  In particular do **not** add `timestamp` yourself; it is rejected as an extra key.
- Numeric features must be `np.ndarray` with exactly matching dtype/shape. An `int64` scalar column
  must be `np.array([i], dtype=np.int64)` with shape `(1,)`; a bare int will not pass.
- Images are `np.uint8` HWC **RGB** — `RealSenseCameraConfig` / `OpenCVCameraConfig` default to
  `color_mode=RGB`, so they pass straight through to the dataset. It is the cv2 display path that
  needs the conversion to BGR, not the dataset path.

**`build_dataset_frame` only handles 1-D float32 and images.** `int64` columns are silently
skipped, so columns like `subtask_index` / `back_event` require hand-built features and frames.

**Nothing in the repo writes `meta/subtasks.parquet`.** It is only ever read, in `__getitem__`
(`meta.subtasks.iloc[idx].name`). A `subtask_index` column on its own reads back empty — the writing
script must produce that parquet itself (see `lerobot_record_piper._write_subtasks_parquet`).

**Durability is asymmetric**: metadata is flushed per episode, but episode data goes through a
long-lived `pq.ParquetWriter` whose footer is only emitted on `close()`. A `kill -9` leaves a
directory where the metadata claims 3 episodes and the data has 2 — and on that inconsistency
`LeRobotDataset.__init__` **falls back to a HuggingFace Hub request**, which hangs for a local
repo_id. Therefore:
- Call `dataset.meta._flush_metadata_buffer()` after each saved episode (the buffer holds 10 by default).
- Every N episodes call `dataset._close_writer()` and set `dataset._writer_closed_for_reading = True`.
  Without that flag the next episode reopens a writer on the same path and truncates what was written.
- Wrap the main loop in `with VideoEncodingManager(dataset):` to shut the encoders down cleanly.
- When resuming, call `repair_dataset_consistency(root)` (`datasets/repair.py`) **before**
  constructing `LeRobotDataset`. It keeps the longest valid prefix and quarantines the rest into
  `_quarantine_*` rather than deleting it.

**Two more**: `LeRobotDatasetMetadata.create` uses `root.mkdir(exist_ok=False)`, so an existing
directory raises `FileExistsError` outright. And a resumed dataset comes back with
`episode_buffer = None`, so add `dataset.episode_buffer = dataset.create_episode_buffer()` —
otherwise discarding before any frame was added dereferences `None`.

---

## Piper leader-follower teleoperation and collection

Main scripts: `scripts/lerobot_record_piper.py` (collect) and `lerobot_replay_piper.py` (replay);
configs in `examples/piper/record_piper.yaml` and `record_dual_piper.yaml`. The driver layer reuses
`Piper` (follower) + `PiperLeader` (leader, with gravity compensation). Dual-arm means two such
pairs — it deliberately does **not** use the `DualPiper` class.

**CAN bandwidth is a hard constraint, not a tuning knob.** A gs_usb frame write costs ≈2.3 ms and
one `JointCtrl` is 3 frames. `PiperMotorsBus.write()` sends MotionCtrl_2 + JointCtrl + GripperCtrl
every tick = **5 frames ≈ 11.5 ms per arm**, so 23 ms for two arms — most of the 33 ms budget at
30 Hz, and the root cause of dual-arm stutter. That is why `ArmPair.send()` bypasses `send_action()`
and gets the steady state down to 3 frames: the mode frame is re-sent on a seconds-scale interval
(the firmware remembers the current mode) and the gripper frame only goes out when the target
actually changed.

**Leader gravity compensation is built on `JointMitCtrl(kp=0, kd=0)`**
(`teleoperators/piper_leader/gravity_compensation.py`).

**`Piper.connect()` defaults to `calibrate=True`**, which immediately drives the follower back to
its home position. Pass `calibrate=False` when the script decides its own initialization, or the arm
moves before `init_mode` ever takes effect.

**Rate limiting (`rate_limit_joints()` in `rlt/piper_env.py`) scales the whole vector uniformly**
rather than clipping per joint, so the direction is preserved. The script prints a saturation rate at
exit; a high rate means the recorded action no longer matches human intent, so raise
`max_joint_step_rad` and re-collect.

**What gets recorded as the action** (`collection.action_source`):
- `follower_next` (default) — the follower's measured pose on the **next** tick. During teaching the
  follower does not track the leader perfectly, so recording the leader command records an action
  that was never executed. The cost is dropping the last frame of each episode.
- `leader` — the rate-limited target actually sent to the follower.
- ⚠️ There is deliberately no "follower pose this tick" option: that makes `action[t] ≡ state[t]`,
  the policy learns the identity map, and the arm never moves at inference time.

**Joint naming**: single arm is `joint_1.pos`…`joint_7.pos` (6 joints + gripper); dual arm puts the
left arm in `joint_1..7` and the right in `joint_8..14`, matching the existing `DualPIPERConfig`
convention.

---

## Code Style

- Line length 110; `ruff` rule set `E, W, F, I, B, C4, T20, N, UP, SIM`
- Python ≥ 3.12; Google-style docstrings
- Strict `mypy` only for: `configs/`, `optim/`, `cameras/`, `motors/`, `transport/`, `envs/`
- **Comments (inline and docstrings) and Markdown files are written in Chinese** — this CLAUDE.md is
  the exception, by explicit request
- **No redundant comments**: explain *why* and non-obvious constraints, never restate what the code does
- In teardown paths (`finally` / `disconnect` / `close`), wrap each step in its own try/except and
  log it — an exception raised during teardown masks the real one, so hardware debugging ends up
  chasing an unrelated error

## Testing

- `tests/` mirrors `src/lerobot/`. Hardware-facing tests always use fake buses / robots / datasets
  and never touch real hardware.
- When a test needs a dataset, build a real `LeRobotDataset` rather than mocking it. Pinning the
  write side and the read side separately misses schema mismatches — the round-trip test is what
  caught the missing `meta/subtasks.parquet` and an image channel-order bug.

## Git Conventions

Commit immediately after each complete change:
- Stage only the files belonging to that change. **Do not use `git add -A` / `git add .`** — the
  working tree often holds the user's own pre-staged edits, and a `git commit` without explicit
  paths will sweep them in.
- Commit messages are written in Chinese, briefly describing what changed and why.
- Never skip pre-commit hooks (no `--no-verify`).
- Never `git push` unless explicitly asked.
