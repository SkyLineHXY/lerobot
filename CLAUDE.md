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
./can_config.sh          # bring up CAN interfaces after power-on; ./find_all_can_port.sh to inspect

# first hardware run: dry_run + init_mode=none (reads only, sends nothing)
lerobot-record-piper --config_path examples/piper/record_piper.yaml --collection.dry_run=true --collection.init_mode=none
lerobot-record-piper --config_path examples/piper/record_piper.yaml       # single arm
lerobot-record-piper --config_path examples/piper/record_dual_piper.yaml  # dual arm
lerobot-replay-piper --root=<dataset_root> --repo_id=<repo_id> --episode=0

# RLT online RL
lerobot-rlt-train-token / lerobot-rlt-train-online / lerobot-rlt-eval-token
lerobot-eval --config_path examples/rlt/eval_libero_smolvla.yaml   # stage-2 baseline: frozen VLA before RL token/critic
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
| `rlt/` | **Everything around online RL**: training loops, chunk replay buffer, envs, human intervention (the model itself lives in `policies/rlt/`). See `src/lerobot/rlt/README.md` |
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
> that import every annotation is a *string*, so `wrap()` cannot resolve `main`'s config class and
> draccus never builds the config tree. The resulting error names neither the import nor the
> config class, so it costs an afternoon to trace back. Helper modules may use it freely — only
> the module holding the `@parser.wrap()`-decorated `main()` is affected.

**Datasets**: Parquet (state/action) + MP4 (images), with HF Hub streaming. `configs/types.py`
defines the `FeatureType` enum (`STATE`, `VISUAL`, `ENV`, `ACTION`, `REWARD`, `LANGUAGE`).

---

## Hard constraints when writing a LeRobot v3 dataset

These apply when assembling frames yourself instead of going through `lerobot-record`; each has
bitten during real debugging.

- **`add_frame` (`datasets/utils.py::validate_frame`)**: `task` goes *inside* the frame dict, not a
  separate arg. Keys must exactly match `set(features) - DEFAULT_FEATURES` — don't add `timestamp`
  yourself. Numeric features need exact `np.ndarray` dtype/shape (e.g. `np.array([i], dtype=np.int64)`,
  not a bare int). Images are `np.uint8` HWC RGB — only the cv2 *display* path needs BGR.
- **`build_dataset_frame` only handles 1-D float32 and images** — `int64` columns like
  `subtask_index` need hand-built features/frames.
- **`meta/subtasks.parquet` is only ever read, never written by the repo.** A `subtask_index` column
  alone reads back empty; the writing script must produce that parquet itself (see
  `lerobot_record_piper._write_subtasks_parquet`).
- **Durability is asymmetric**: metadata flushes per episode, but episode data sits behind a
  long-lived `pq.ParquetWriter` that only writes its footer on `close()`. A `kill -9` mid-run leaves
  metadata/data episode counts mismatched, and `LeRobotDataset.__init__` then falls back to a HF Hub
  request that hangs for a local repo_id. Mitigate with: `dataset.meta._flush_metadata_buffer()`
  after each episode; `dataset._close_writer()` (+ `_writer_closed_for_reading = True`) every N
  episodes; `with VideoEncodingManager(dataset):` around the main loop; and
  `repair_dataset_consistency(root)` (`datasets/repair.py`) before reopening a dataset that may have
  crashed mid-write.
- `LeRobotDatasetMetadata.create` uses `mkdir(exist_ok=False)` — an existing directory raises.
  A resumed dataset's `episode_buffer` comes back `None`; set it via `create_episode_buffer()` before
  discarding, or a frame-less discard dereferences `None`.

---

## Piper leader-follower teleoperation and collection

Main scripts: `scripts/lerobot_record_piper.py` (collect) and `lerobot_replay_piper.py` (replay);
configs in `examples/piper/record_piper.yaml` and `record_dual_piper.yaml`. Driver layer: `Piper`
(follower) + `PiperLeader` (leader, gravity compensation via `JointMitCtrl(kp=0, kd=0)`). Dual-arm
uses two such pairs, not the `DualPiper` class.

**CAN bandwidth caps update rate.** `PiperMotorsBus.write()` costs ~11.5 ms/arm (23 ms for two),
most of the 33 ms budget at 30 Hz — the root cause of dual-arm stutter. `ArmPair.send()` (used
instead of `send_action()`) works around this by only resending the mode frame periodically and the
gripper frame on change.

**`Piper.connect()` defaults to `calibrate=True`** and immediately homes the follower — pass
`calibrate=False` when the script controls its own init, or the arm moves before `init_mode` applies.

**An arm stuck in teach/master `ctrl_mode` self-recovers on connect** via
`guard_piper_ctrl_mode_on_connect` (`utils/piper_sdk.py`): switches role back to slave, seeds
`JointCtrl` with the arm's current pose (avoids snapping to a stale latched target), then requests
CAN mode. Only raises — asking for a physical power-cycle — after `recover_attempts` rounds fail.

**`rate_limit_joints()` (`rlt/piper_env.py`) scales the whole vector uniformly** to preserve
direction. A high saturation rate at exit means the recorded action no longer matches human intent —
raise `max_joint_step_rad` and re-collect.

**`collection.action_source` default is `follower_next`**, not the leader command: during teaching
the follower doesn't track the leader perfectly, so recording the leader's target would record an
action that was never executed. There is deliberately no "follower pose this tick" option — that
would make `action[t] ≡ state[t]` and the policy would learn the identity map instead of moving.

**Joint naming**: single arm `joint_1.pos`…`joint_7.pos`; dual arm splits left=`joint_1..7`,
right=`joint_8..14` (matches `DualPIPERConfig`).

---

## Code Style

### Comments: the budget is close to zero

Nothing in `ruff` or `pre-commit` enforces this, so it is on you. **Most functions need no comment
at all; almost none need more than one.** Before writing a comment, ask: *without this, would a
careful reader get it wrong?* If not, delete it.

Never write:

- section banners — `# 1) read leader`, `# ---- reset`, `# Timing buffers`. They rot: the numbered
  run in `rlt/teleop_check.py` reads 1, 2, 4, 5 because step 3's label was deleted and the rest
  were never renumbered. That is the maintenance cost of a comment carrying no information.
- a comment naming what the next line already names — `# Baseline: mean-pooled embeddings` above
  the two lines that visibly compute the mean pool.
- a docstring that only restates the signature.
- narration of the edit you just made — `# NEW`, `# changed to ...`, `# was: ...`. Git holds that.

Do write: why a non-obvious constant has that value, a constraint learned by debugging, a
correctness trap the code steps around. `src/lerobot/rlt/replay_buffer.py` is the reference for the
right density — long "why" blocks where the invariant is genuinely subtle, nothing anywhere else.

### The rest

- Line length 110; `ruff` rule set `E, W, F, I, B, C4, T20, N, UP, SIM`
- Python ≥ 3.12; Google-style docstrings
- Strict `mypy` only for: `configs/`, `optim/`, `model/`, `cameras/`, `motors/`, `transport/`, `envs/`
- **Comments and docstrings are written in English.** Markdown files stay Chinese. Existing Chinese
  comments are fine where they are — don't do a sweeping translation pass, just write new ones in
  English
- In teardown paths (`finally` / `disconnect` / `close`), wrap each step in its own try/except and
  log it — an exception raised during teardown masks the real one, so hardware debugging ends up
  chasing an unrelated error

## Testing

- `tests/` mirrors `src/lerobot/`. Hardware-facing tests always use fake buses / robots / datasets
  and never touch real hardware — see `tests/rlt/test_piper_ctrl_mode_guard.py` for the pattern
  (a `FakeArm` whose mode only changes in response to specific commands, not on every read).
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
