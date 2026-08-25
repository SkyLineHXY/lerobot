"""Configuration dataclasses for the RLT reproduction on SmolVLA.

Defaults follow the paper (App. B) where specified; unspecified values use
sensible defaults documented inline. VLA-dependent widths/dims default to
the `lerobot/smolvla_base` checkpoint (SmolVLM2-500M backbone, 3 cameras,
6-dim SO-100 action space).

Both stages are driven by draccus, like the rest of the repo: a YAML file
supplies the experiment and any field can still be overridden on the command
line.

    lerobot-rlt-train-online --config_path examples/rlt/piper/online.yaml \\
        --rl.utd=8 --env.control_hz=20

Example configs live in ``examples/rlt/``.
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field

import draccus


@dataclass
class RLTokenConfig:
    """Stage 1: RL-token encoder/decoder (Eq. 1-2)."""

    # Width of the VLA final-layer embeddings (SmolVLM2-500M text -> 960).
    vla_width: int = 960
    # Learned queries the encoder reads z_rl out of. The RL state stage 2 sees
    # is the flattened stack, so its width is `d_model * num_rl_tokens`.
    num_rl_tokens: int = 1
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
    # Caps the reconstruction target length, and with it the decoder's learned
    # positional queries. Prefixes longer than this are subsampled by linspace.
    max_recon_tokens: int = 256
    # SmolVLA scales image embeddings by sqrt(dim) (~31) inside embed_prefix,
    # so a raw MSE on z is dominated by a few large-magnitude channels. Fitting
    # the bottleneck in a standardised space makes L_ro measure information
    # rather than scale.
    normalize_targets: bool = True

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
    """Stage 2 networks (paper App. B), laid out as openpi-RLT `networks.py`."""
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

    # Widths of the three input projections that feed the trunk. openpi-RLT
    # hardcodes 256/64/256; they are fields here only so the magic numbers have
    # a name.
    z_proj_dim: int = 256
    proprio_proj_dim: int = 64
    ref_proj_dim: int = 256

    # Fixed Gaussian std of the actor ("small fixed standard deviation"). It is
    # both the exploration noise and the target-policy smoothing noise, so 0
    # leaves the TD target completely unsmoothed.
    action_std: float = 0.05
    # Reference-action dropout probability during training (paper: 0.5).
    ref_dropout: float = 0.5
    # 0 is the paper/openpi-RLT actor: emit the complete action chunk directly.
    # A positive value keeps the old bounded-residual parameterisation only as
    # an explicit ablation; it is incompatible with the intended semantics of
    # reference dropout because a narrow residual cannot reconstruct a full
    # action when the reference is hidden.
    residual_scale: float = 0.0
    n_critics: int = 2  # TD3-style double Q, min for targets

    def __post_init__(self):
        if not 0.0 <= self.ref_dropout <= 1.0:
            raise ValueError(f"ref_dropout must be in [0, 1], got {self.ref_dropout}")
        if self.residual_scale < 0:
            raise ValueError(f"residual_scale must be non-negative, got {self.residual_scale}")
        if self.residual_scale > 0 and self.ref_dropout > 0:
            warnings.warn(
                "residual_scale>0 with reference dropout cannot reconstruct a full action when the "
                "reference is hidden; use residual_scale=0 for RL Token, or ref_dropout=0 for a "
                "bounded-residual ablation.",
                stacklevel=2,
            )

@dataclass
class OnlineRLConfig:
    """Stage 2 training loop (Algorithm 1 + App. B)."""
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    discount: float = 0.99
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    tau: float = 0.005  # polyak for both target networks (TD3)

    # Paper Eq. (5) is ``-Q + beta * BC``.  Keep the two terms in those native
    # units by default, as do openpi-RLT and Evo-RLT's paper config (beta=1).
    # The former 10/0.1 -> 5/0.1 pair was inherited from a TD3+BC variant that
    # divided Q by mean|Q|.  Once that non-paper normalisation is removed it is
    # equivalent to beta=100/50 and leaves essentially no policy-improvement
    # gradient (measured weighted terms: ~0.40 BC versus ~0.001 Q).
    warmup_bc_weight: float = 1.0
    warmup_q_weight: float = 1.0
    online_bc_weight: float = 1.0
    online_q_weight: float = 1.0
    # Both reference implementations switch these coefficients at handover.
    # A positive value is retained as an ablation for gradual interpolation.
    weight_ramp_updates: int = 0

    # Optional advantage-weighted self-imitation ablation. It is not part of
    # RL Token Eq. (5), whose policy objective is BC - beta*Q.
    #
    # `awr_beta` multiplies the batch-standardised advantage, so it is unitless;
    # weights are clipped and then renormalised to mean 1.
    awr_weight: float = 0.0
    awr_beta: float = 1.0
    awr_weight_max: float = 20.0

    # Mild upweighting of operator-driven steps inside the BC term. The actor
    # remains conditioned on the VLA proposal and the target becomes the action
    # actually executed by the operator on those steps.
    human_bc_weight: float = 1.0

    # Paper Sec. V's prose says to replace the replay reference on intervention
    # rows, while the current openpi-RLT implementation preserves the VLA
    # proposal and uses the human action only as the BC target.  The latter
    # avoids a train/deploy input shift and is the executable-reference default;
    # true keeps the paper-prose variant as an explicit ablation.
    human_ref_override: bool = False

    # Optional Evo-RLT stability ablation; the paper/openpi path does not clip.
    target_q_clip: float = 0.0

    # Step-to-step smoothness penalty on the predicted chunk. openpi-RLT only
    # enables it for one real-robot task, in denormalised joint space; left off.
    delta_weight: float = 0.0

    # Convex blend of the TD target with the episode's own discounted
    # return-to-go, applied per row and only where `mc_valid` (terminated
    # episodes; a truncated one's return is a lower bound, not a sample).
    # 0 is pure TD, the reference behaviour.
    #
    # Why this exists: measured on a LIBERO run, pure TD had propagated the
    # terminal reward almost nowhere after 50k gradient steps — the buffer's
    # true mean value was 0.193 while the critic reported 0.023, with the error
    # growing the further a state sat from the reward. The same network fitted
    # those returns by plain supervised regression to R^2 = 0.61 in 8k steps, so
    # the features were never the problem; the bootstrap chain was. Blending
    # rather than replacing keeps the policy-improvement property of the
    # bootstrap, since these returns were earned by a mixture of base VLA, actor
    # and human takeovers and are badly off-policy.
    #
    critic_mc_weight: float = 0.0

    # openpi-RLT clips no gradients. Kept as a switch because a diverging critic
    # is the first thing worth trying it on.
    grad_clip_norm: float = 0.0

    batch_size: int = 128
    # Update-to-data ratio: gradient steps per stored transition (paper: 5).
    utd: int = 5
    # A simulator's time limit is a real failure with return 0, so its Monte
    # Carlo target is valid. On hardware a truncated episode really does mean
    # "we stopped looking", so this stays off there.
    truncation_is_failure: bool = False
    actor_update_period: int = 2
    # Mirrored from `RLTWandBConfig.diagnostics_every`; the learner process only
    # ever receives this sub-config.
    diagnostics_every: int = 100

    # Replay buffer
    buffer_capacity: int = 200_000
    subsample_stride: int = 2
    # "stratified" reserves a share of every batch for the three groups uniform
    # sampling under-weights: recent online transitions, warmup/demo
    # transitions, and operator takeovers. Pools overlap and any shortfall is
    # topped up uniformly, so an empty pool costs nothing (openpi-RLT
    # `replay.py::_sample_stratified_indices`). Uniform is the reference default
    # — and with it, takeovers reach the learner at exactly the rate the
    # operator happened to produce them (measured: 0.16 of batches), which is why
    # human data looked like it was doing nothing.
    sample_strategy: str = "uniform"  # uniform | stratified
    recent_episode_window: int = 20
    recent_online_ratio: float = 0.4
    warmup_demo_ratio: float = 0.3
    # Takeovers are the only place the buffer holds an action the policy itself
    # would not have produced at that state, so they are what teaches the critic
    # to depend on the action rather than on the state alone.
    human_intervention_ratio: float = 0.2
    # Optional stratum for windows that lie inside a single planning decision —
    # the only rows the actor's BC/AWR terms can use. Off by default: they are
    # already `subsample_stride / chunk_len` of every pool, and the three strata
    # above deliberately claim 0.9 of the batch, so anything here comes out of
    # the uniform top-up. Watch `train/aligned_ratio` before raising it.
    aligned_ratio: float = 0.0

    # Rollout
    warmup_env_steps: int = 2_000  # N_warm: prefill with base VLA rollouts
    # Gradient steps the learner must also have taken before the actor is
    # allowed to drive (openpi-RLT `warmup_post_collect_updates`, 20000 in its
    # real-robot task config). Without it warmup ends on wall-clock interaction
    # alone and the actor takes over a critic that is still worthless — measured
    # handover at ~4k updates with `q_mean = -0.44`, after which the run
    # degraded monotonically. 0 keeps the env-steps-only behaviour.
    warmup_post_collect_updates: int = 0
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
    removes GIL contention and permits the lightweight learner to run on CPU or
    another GPU while VLA rollout keeps its accelerator. Needs grpcio:
    `pip install -e ".[async]"`.
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
    # None inherits the rollout device. "cpu" is often faster end-to-end for
    # this small MLP because it leaves the GPU exclusively to the 500M VLA.
    learner_device: str | None = None
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
    # Frame the operator's delta-EE command is expressed in: "base" (the env's
    # own action frame) or "tcp" (the current end-effector frame, rotated into
    # the base frame before execution). What reaches the env, the dataset and
    # the replay buffer is a base-frame action either way. "tcp" is what makes
    # an insertion correctable once the gripper has yawed: the stick then means
    # the same thing relative to the tool, and to the wrist camera, all episode.
    frame: str = "base"
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
    # Slice of the task's 50 fixed initial states this env draws from. LIBERO
    # selects the initial state by a counter rather than by the seed, so a
    # train run and an eval run otherwise see the same states no matter how
    # they are seeded. `n_init_states = 0` means "all of them".
    init_state_offset: int = 0
    n_init_states: int = 0
    teleop: SimTeleopConfig | None = None


@ChunkEnvConfig.register_subclass("insertion")
@dataclass
class InsertionEnvConfig(ChunkEnvConfig):
    """Rectangular bar into a rectangular slot: short-horizon, fine manipulation.

    Same 7-dim OSC_POSE action space and 8-dim proprio as LIBERO, so a stage-1
    run and a SmolVLA checkpoint carry over untouched. What differs is the shape
    of the reward problem: episodes are under 100 steps and the single terminal
    bit is decided by the last centimetre of motion rather than by a long chain
    of sub-goals.
    """

    max_episode_steps: int = 250
    prompt: str = "insert the red bar into the blue slot"
    # "pregrasped" starts with the bar already held — pure alignment and
    # insertion. "full" drops it on the table and the policy has to pick it up
    # first, which stretches the same one bit of reward over a much longer
    # horizon; start with "pregrasped".
    task_stage: str = "pregrasped"
    control_mode: str = "relative"
    control_freq: int = 20
    observation_size: int = 256
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    # Slice of the fixed initial-state pool this env draws from. As in LIBERO the
    # state is chosen by a counter, not by the seed, so training and evaluating
    # over the same window is train-on-test no matter how the run is seeded.
    # `n_init_states = 0` means "all of them".
    init_state_offset: int = 0
    n_init_states: int = 0
    pool_size: int = 50
    pool_seed: int = 0
    # Task geometry overrides forwarded to `RectPegInsertion`, e.g.
    # {"clearance": 0.004} to loosen the fit while the policy is still weak.
    task_kwargs: dict = field(default_factory=dict)
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
    # `align()` issues an asynchronous JointCtrl and does not wait for the
    # leader to arrive; measuring the takeover gap before it settles refuses
    # takeovers for a gap that no longer exists. Runs off the control thread.
    align_settle_s: float = 1.0
    # What the buffer stores as the human's action: "leader" is the operator's
    # own pose, "issued" is the slew-limited command the follower executed.
    # They differ only while the limiter saturates.
    action_source: str = "leader"


@ChunkEnvConfig.register_subclass("piper")
@dataclass
class PiperEnvConfig(ChunkEnvConfig):
    """Real Piper arm with human-in-the-loop interventions."""

    can_port: str = "can0"
    task: str = ""
    # How often the chunk loop samples an observation and names a new target.
    # It is no longer the rate the arm is commanded at — see `stream_hz`.
    control_hz: float = 30.0
    # Joint speed limit applied after un-normalisation, anchored on the last
    # issued command. The whole vector is scaled rather than clipped per joint
    # so limiting never bends the commanded direction. Its job is to catch
    # anomalies (a struck arm, a torn CAN read), not to damp everyday jitter —
    # that is what the high command rate is for.
    max_joint_vel_rad_s: float = 6.0
    # How far the command may run ahead of the measured pose. Stops a blocked
    # arm from accumulating a target it would lunge to once freed.
    max_lead_rad: float = 0.5
    # Deprecated: a per-tick radian budget. Its physical meaning drifted with
    # `control_hz` (0.05 was 1.5 rad/s at 30 Hz, 1.25 at 25 Hz). Setting it
    # still works and is converted, with a warning.
    max_joint_step_rad: float | None = None
    # Command-thread ceiling. The steady state is the leader's own feedback
    # rate during a takeover (~188 Hz measured); this only stops an abnormally
    # fast feedback stream from flooding the bus.
    stream_hz: float = 250.0
    # Poll interval while waiting for a new leader sample. Never precise_sleep:
    # it busy-waits its last 10 ms holding the GIL.
    idle_poll_s: float = 0.001
    # Joint deadband (rad). Leave at 0. It compares against the last commanded
    # target, so at streaming rates it quantizes slow motion into steps —
    # exactly during the slow precise moves this path exists to keep smooth.
    joint_deadband_rad: float = 0.0
    # Gripper deadband in 1e-6 m. 200 = 0.2mm.
    gripper_deadband: int = 200
    # How often the CAN mode frame is resent.
    mode_refresh_interval_s: float = 1.0
    # Follower speed percentage in position mode. Without MIT impedance this is
    # the only compliance knob: lower is softer but blunter.
    move_speed_ratio: int = 60
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


@dataclass
class RLTWandBConfig:
    """Stage-2 W&B run. See `rlt/wandb_logger.py` for what gets logged."""

    enable: bool = False
    project: str = "lerobot_rlt"
    entity: str | None = None
    run_name: str | None = None
    notes: str | None = None
    # Set to resume an interrupted run into the same curves.
    run_id: str | None = None
    mode: str | None = None  # online | offline | disabled
    # Gradient steps between diagnostic probes (extra critic forwards on one
    # batch). They are what make the curves worth reading, but they are not
    # free, so they run on their own slower clock than the losses.
    diagnostics_every: int = 100


# ------------------------------------------------------------- entry points
@dataclass
class RLTokenTrainConfig:
    """Stage 1: train the RL token on task demonstrations."""

    checkpoint: str = "lerobot/smolvla_base"
    dataset: str = "lerobot/libero_10"
    dataset_root: str | None = None
    # Train the bottleneck on demonstrations from one fixed task, as in the
    # paper. None keeps all dataset tasks for cross-task ablations.
    dataset_task_index: int | None = None
    # Optional episode slice after task filtering. This makes a true
    # episode-level representation holdout possible without copying a dataset.
    dataset_episode_start: int = 0
    dataset_num_episodes: int | None = None
    out: str = "outputs/rl_token"
    device: str = "cuda"
    dtype: str | None = None
    num_workers: int = 4
    seed: int = 0
    log_freq: int = 20
    save_freq: int = 500
    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)
    wandb: RLTWandBConfig = field(default_factory=RLTWandBConfig)


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

    view: RolloutViewConfig = field(default_factory=RolloutViewConfig)
    wandb: RLTWandBConfig = field(default_factory=RLTWandBConfig)
    num_inference_steps: int | None = None
    resume_buffer: str | None = None
    log_freq: int = 200
    save_freq: int = 2000

    def __post_init__(self):
        # One source of truth for the knobs that appear in both places.
        self.rl.device = self.device
        self.rl.seed = self.seed
        self.rl.max_episode_steps = self.env.max_episode_steps
        self.rl.diagnostics_every = self.wandb.diagnostics_every
