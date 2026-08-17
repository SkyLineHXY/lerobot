"""Configuration dataclasses for the RLT reproduction on SmolVLA.

Defaults follow the paper (App. B) where specified; unspecified values use
sensible defaults documented inline. VLA-dependent widths/dims default to
the `lerobot/smolvla_base` checkpoint (SmolVLM2-500M backbone, 3 cameras,
6-dim SO-100 action space).

Both stages are driven by draccus, like the rest of the repo: a YAML file
supplies the experiment and any field can still be overridden on the command
line.

    lerobot-rlt-train-online --config_path examples/rlt/piper_online.yaml \\
        --rl.utd=8 --env.control_hz=20

Example configs live in ``examples/rlt/``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import draccus


@dataclass
class RLTokenConfig:
    """Stage 1: RL-token encoder/decoder (Eq. 1-2)."""

    # Width of the VLA final-layer embeddings (SmolVLM2-500M text -> 960).
    vla_width: int = 960
    # Transformer width for encoder/decoder. The paper depicts a 1x2048 RL
    # token on pi0.6 (= VLA width); 512 keeps the module light (~15M params)
    # on the 960-wide SmolVLA. Use --d-model 960 to match "RL token = VLA
    # width" exactly.
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Paper footnote 1: with a fixed per-task instruction, language embeddings
    # are dropped from the reconstruction. Everything else is kept — for
    # SmolVLA that means the per-camera image tokens *and* the single state
    # token, which is the only proprioceptive entry point in the VLM prefix.
    drop_language_tokens: bool = True
    # Stricter variant that also drops the state token (the original behaviour
    # of this module). Kept for ablations; `drop_language_tokens` wins when
    # both are set.
    use_image_tokens_only: bool = False
    # SmolVLA scales image embeddings by sqrt(dim) (~31) inside embed_prefix,
    # so a raw MSE on z is dominated by a few large-magnitude channels. Fitting
    # the bottleneck in a standardised space makes L_ro measure information
    # rather than scale.
    normalize_targets: bool = True
    # Max number of prefix tokens kept for reconstruction (subsample if more;
    # SmolVLA: 64 connector tokens per camera x 3 cameras = 192 image tokens,
    # so 256 also covers the language+state tokens if image_only is disabled).
    max_recon_tokens: int = 256

    # Stage-1 training (paper: 2000-10000 steps on task demos)
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    steps: int = 5000
    batch_size: int = 8
    # Weight alpha for optional joint VLA supervised fine-tuning (Algorithm 1
    # line 3). alpha = 0 keeps the VLA fully frozen.
    vla_sft_alpha: float = 0.0
    # The VLA gets its own optimizer and learning rate when alpha > 0. A fresh
    # ~15M encoder/decoder wants lr=1e-4; driving a pretrained 500M VLA at that
    # rate wrecks it. Keeping the two groups separate also keeps grad clipping
    # from coupling their norms (openpi-RLT `rl_token_trainer._step_joint`).
    vla_lr: float = 1e-5


@dataclass
class ActorCriticConfig:
    """Stage 2 networks (paper App. B)."""
    # RL chunk length C (paper: 10) and per-step action dim d
    # (smolvla_base / SO-100: 6).
    chunk_len: int = 10
    action_dim: int = 6
    # Dim of proprioceptive state s^p appended to the RL token.
    proprio_dim: int = 6
    rl_token_dim: int = 512

    # 2-layer MLP hidden 256 for most tasks; 3-layer 512 for screw task.
    hidden_dim: int = 256
    n_layers: int = 2

    # Fixed Gaussian std of the actor ("small fixed standard deviation").
    action_std: float = 0.05
    # Reference-action dropout probability during training (paper: 50%).
    ref_dropout: float = 0.5

    # The actor predicts a *residual* on top of the VLA reference chunk with a
    # zero-initialised output layer, so at step 0 the RL policy is bit-exact
    # equal to the base VLA (rlt-openpi `models/actor.py`). `max_residual`
    # bounds how far RL may deviate per action dim in normalized action space —
    # this is the safety box whose absence made HIL-SERL fail in the paper's
    # 50 Hz setting (App. C).
    max_residual: float = 0.3
    # Normalized action space of the VLA is [-1, 1]; executed chunks are clamped.
    action_clip: float = 1.0

    n_critics: int = 2  # TD3-style double Q, min for targets

@dataclass
class OnlineRLConfig:
    """Stage 2 training loop (Algorithm 1 + App. B)."""
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    discount: float = 0.97  # per control step; chunk backup uses gamma^C
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    tau: float = 0.005  # polyak for target critics (TD3)
    # BC-regularizer weight beta in Eq. 5 (task-dependent in the paper).
    # Convention: ||a - a~||^2 *summed* over the C x d chunk, matching the
    # paper's squared L2 norm. A mean-reduction would silently divide beta by
    # C*d (= 60 here) and leave the actor essentially unanchored.
    bc_beta: float = 0.3

    # TD3 target policy smoothing (paper only says "sample from pi_theta"; the
    # reference implementations use explicit smoothing to stop the critic from
    # being exploited at sharp Q peaks).
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    # High UTD + sparse rewards + a small buffer is the classic recipe for Q
    # divergence. With binary rewards Q is bounded by 1/(1-gamma^C); clamping
    # the bootstrap to [0, q_max_scale/(1-gamma^C)] keeps it from running away.
    target_q_clip_scale: float = 1.0
    grad_clip_norm: float = 1.0

    batch_size: int = 256
    # Update-to-data ratio: gradient steps per environment *chunk* collected
    # (paper: UTD 5 counted in transitions; stride-2 subsampling stores
    # C/stride transitions per chunk, so per-chunk updates = utd * C/stride).
    utd: int = 5
    critic_updates_per_actor_update: int = 2

    # Replay buffer
    buffer_capacity: int = 200_000
    subsample_stride: int = 2

    # Rollout
    warmup_env_steps: int = 2_000  # N_warm: prefill with base VLA rollouts
    total_env_steps: int = 100_000
    max_episode_steps: int = 400

    seed: int = 0
    device: str = "cuda"


# --------------------------------------------------------------------- envs
@dataclass
class ChunkEnvConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base for the environments the stage-2 rollout worker can drive."""

    max_episode_steps: int = 400

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@dataclass
class ConcurrencyConfig:
    """How the rollout and the learner are separated.

    `threads` is the default and the reference implementation. `processes`
    removes GIL contention and the shared-CUDA-stream drain, but **not** GPU
    compute contention — the two processes still time-share the SMs, so the
    gradient-burst cap still matters. Needs grpcio: `pip install -e ".[async]"`.
    """

    mode: str = "threads"  # threads | processes
    learner_host: str = "127.0.0.1"
    learner_port: int = 50061
    # How often the learner pushes actor weights. The rollout only reads them at
    # chunk boundaries, so pushing faster than that is wasted bandwidth.
    parameters_push_hz: float = 10.0
    # Bound on undelivered buffer ops. Reaching it blocks the rollout, which is
    # visible and recoverable; dropping data instead would look like RL quietly
    # not working.
    max_pending_ops: int = 4096
    # Intra-op threads inside the learner process. The actor-critic is a 2-layer
    # MLP on batches of 256, so torch's default (one thread per core) spends far
    # more on synchronisation than it saves: measured 6.7 updates/s at 48
    # threads versus 163 at 1. It also stops the learner from saturating every
    # core and starving the rollout. `torch.set_num_threads` is global, which is
    # why only the process backend can set it without also throttling the VLA.
    learner_torch_threads: int = 1


@dataclass
class SimTeleopConfig:
    """Hand-held device used for human interventions in simulation."""

    # keyboard / gamepad / spacemouse. All three are matched to the env by
    # channel *name*, so a device missing the rotation axes simply leaves them
    # at zero rather than misaligning the vector.
    device: str = "keyboard"
    use_rotation: bool = True
    # Device output is normalized to [-1, 1] and LIBERO's delta-EE box is also
    # [-1, 1], so these are pure gain: 1.0 means "full stick = full step". Lower
    # them for finer corrections near the object.
    position_scale: float = 0.3
    rotation_scale: float = 0.2
    # Gamepad only: overrides for `DEFAULT_GAMEPAD_BINDINGS`, e.g.
    # {"delta_yaw": "-rightx"} to flip one stick without restating the layout.
    gamepad_bindings: dict[str, str] = field(default_factory=dict)
    gamepad_deadzone: float = 0.1
    # Accept the device's own buttons for takeover / success / failure as well as
    # the keyboard. Off by default so sim and hardware share one operator-key
    # story; on for a gamepad, where both hands are off the keyboard anyway.
    use_device_events: bool = False


@ChunkEnvConfig.register_subclass("mock")
@dataclass
class MockEnvConfig(ChunkEnvConfig):
    """Abstract reaching task; regression path only, never a validation signal."""

    max_episode_steps: int = 120
    image_size: int = 256
    success_eps: float = 0.15
    prompt: str = "reach the target"
    teleop: SimTeleopConfig | None = None


@ChunkEnvConfig.register_subclass("libero")
@dataclass
class LiberoEnvConfig(ChunkEnvConfig):
    """LIBERO simulation: the pre-hardware end-to-end validation path."""

    suite: str = "libero_10"
    task_id: int = 0
    control_mode: str = "relative"
    observation_size: int = 256
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    teleop: SimTeleopConfig | None = None


@dataclass
class CameraSpec:
    """One camera; `serial` selects RealSense, `index_or_path` selects OpenCV."""

    name: str = "cam"
    index_or_path: str | None = None
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class PiperLeaderTeleopConfig:
    """Leader Piper arm used for human interventions."""

    port: str = "can1"
    id: str = "piper_leader"
    # `PiperLeader.get_action()` returns offsets from the calibrated neutral
    # pose. Stage-1 dataset actions are *absolute* joint angles, so the default
    # reads the leader's absolute joints instead — otherwise every human
    # correction is shifted by the leader's neutral pose and the actor's BC term
    # is pulled toward a biased target. Only set this when the leader's neutral
    # pose is deliberately the follower's zero.
    use_calibrated_offsets: bool = False
    # Refuse a takeover if the leader is further than this from the follower.
    # The leader holds the follower's pose between interventions, so a large
    # gap means the operator moved it while it was limp; releasing then would
    # make the follower chase it. 0 disables the check.
    max_takeover_delta_rad: float = 0.15


@ChunkEnvConfig.register_subclass("piper")
@dataclass
class PiperEnvConfig(ChunkEnvConfig):
    """Real Piper arm with human-in-the-loop interventions."""

    can_port: str = "can0"
    task: str = ""
    control_hz: float = 30.0
    # Per-step joint rate limit applied *after* un-normalisation; the whole
    # chunk is scaled rather than clipped per joint so limiting never bends the
    # commanded direction.
    max_joint_step_rad: float = 0.05
    reset_pose: list[float] | None = None
    reset_noise_rad: float = 0.0
    # Empty keeps PIPERConfig's own camera defaults; the camera set has to match
    # what the VLA was fine-tuned on.
    cameras: list[CameraSpec] = field(default_factory=list)
    # Plan, un-normalize and rate-limit, but never command the arm. Always the
    # first real-hardware step.
    dry_run: bool = False
    teleop: PiperLeaderTeleopConfig | None = None


@dataclass
class RolloutViewConfig:
    """OpenCV operator window for the stage-2 rollout.

    Disabled by default: an unattended run has no one to look at it, and a
    headless box has no display to put it on (it degrades to a no-op there
    anyway). Turn it on whenever a human is in the loop.
    """

    enabled: bool = False
    max_width: int = 960
    panel_height: int = 200
    tile_height: int = 320
    # Redraw is per env step. Cap it when the env is much faster than the eye —
    # 0 means "every step".
    min_period_s: float = 0.0


# ------------------------------------------------------------- entry points
@dataclass
class RLTokenTrainConfig:
    """Stage 1: train the RL token on task demonstrations."""

    checkpoint: str = "lerobot/smolvla_base"
    dataset: str = "lerobot/libero_10"
    dataset_root: str | None = None
    out: str = "outputs/rl_token"
    device: str = "cuda"
    dtype: str | None = None
    num_workers: int = 4
    log_freq: int = 20
    save_freq: int = 500
    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)


@dataclass
class RLTOnlineTrainConfig:
    """Stage 2: online RL with the RL token."""

    # Directory written by stage 1: RL-token weights *and* the processors that
    # define the normalisation both stages must agree on.
    rl_token: str = "outputs/rl_token"
    checkpoint: str = "lerobot/smolvla_base"
    out: str = "outputs/rlt_online"
    device: str = "cuda"
    dtype: str | None = None
    seed: int = 0

    env: ChunkEnvConfig = field(default_factory=MockEnvConfig)
    rl: OnlineRLConfig = field(default_factory=OnlineRLConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Operator-key backend: "auto" prefers termios (needs stdin to be a real
    # terminal) and falls back to pynput's global X11 hook, which is what makes
    # the keys work from an IDE console / nohup / roslaunch where stdin is a
    # pipe. "none" disables operator input entirely.
    keyboard_backend: str = "auto"

    # Which key discards the current episode. `left` (arrow) collides with a
    # keyboard teleop device, which steers with the arrows — both readers see
    # the same global keystream, so the only fix is to rebind.
    discard_key: str = "left"

    # Start every episode under the base VLA and hand control to the RL policy
    # when the operator presses `r` (paper Sec. V).
    critical_phase: bool = False

    # Engage the takeover toggle automatically at the start of every *warmup*
    # episode, so the human demonstrates the prefill instead of watching the base
    # VLA fail at it. Worth turning on whenever the base policy's success rate is
    # near zero: warmup exists to give the critic a reward signal to learn from,
    # and it cannot do that from trajectories that never score.
    warmup_human_control: bool = False

    view: RolloutViewConfig = field(default_factory=RolloutViewConfig)
    num_inference_steps: int | None = None
    resume_buffer: str | None = None
    log_freq: int = 200
    save_freq: int = 2000

    def __post_init__(self):
        # One source of truth for the knobs that appear in both places.
        self.rl.device = self.device
        self.rl.seed = self.seed
        self.rl.max_episode_steps = self.env.max_episode_steps
