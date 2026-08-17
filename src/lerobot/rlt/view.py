"""Operator view for stage-2: camera strip plus an RL status panel.

The collection view (`utils/status_view.py`) answers "am I recording and how
many frames do I have"; during online RL the operator needs a different set of
numbers — which phase the run is in, whether the human or the policy is driving
this chunk, and whether the critic has any signal at all to learn from. Only the
panel changes; the window, threading and headless degradation come from
`StatusView` unchanged.

**Rendering happens per env step, not per chunk.** `StatusView.render_once` must
run on the main thread, and both the rollout loop and the teleop intervention
loop do. Drawing only at chunk boundaries would leave the picture a whole chunk
(~1 s at 10 Hz) behind the arm, which is unusable for steering it.
"""

from __future__ import annotations

import time

import numpy as np

from lerobot.utils.status_view import StatusView

PHASE_COLORS = {
    "WARMUP": (0, 230, 230),
    "RL": (60, 230, 60),
    "VLA": (200, 200, 200),
}


class RolloutStatusView(StatusView):
    """`StatusView` with the stage-2 panel."""

    def _draw_panel(self, panel: np.ndarray, status: dict) -> None:
        white = (235, 235, 235)
        grey = (150, 150, 150)
        red = (60, 60, 235)
        width = panel.shape[1]

        def put(text, x, y, color=white, scale=0.5, thick=1):
            self._put(panel, text, x, y, color, scale, thick)

        y = 22
        task = status.get("task")
        if task:
            self._put_fit(panel, f"TASK: {task}", 10, y, (60, 230, 60), 0.6, 1, width - 20)
            y += 26

        phase = str(status.get("phase", "")).upper()
        if phase:
            put(phase, 10, y, PHASE_COLORS.get(phase, white), 0.7, 2)
        if status.get("human"):
            # The one thing that must be readable at a glance: whether the
            # buffer is currently recording the human or the policy.
            put("HUMAN", 120, y, red, 0.7, 2)
        task_id = status.get("task_id")
        if task_id is not None:
            put(f"task {task_id}", width - 90, y, grey, 0.5, 1)
        y += 26

        steps, total = status.get("env_steps"), status.get("total_env_steps")
        line = []
        if steps is not None:
            line.append(f"steps {steps}/{total}" if total else f"steps {steps}")
        if status.get("episodes") is not None:
            line.append(f"ep {status['episodes']}")
        if status.get("fps") is not None:
            line.append(f"{status['fps']:.1f} steps/s")
        if status.get("elapsed") is not None:
            e = status["elapsed"]
            line.append(f"{int(e // 60):02d}:{int(e % 60):02d}")
        if line:
            put(" | ".join(line), 10, y, grey, 0.5)
            y += 22

        ep_line = []
        if status.get("ep_step") is not None:
            limit = status.get("max_episode_steps")
            ep_line.append(f"step {status['ep_step']}/{limit}" if limit else f"step {status['ep_step']}")
        if status.get("ep_return") is not None:
            ep_line.append(f"return {status['ep_return']:.2f}")
        if status.get("ep_interventions") is not None:
            ep_line.append(f"human chunks {status['ep_interventions']}")
        if ep_line:
            put("episode: " + "  ".join(ep_line), 10, y, white, 0.5)
            y += 22

        # A success rate stuck at 0 with a buffer full of transitions is the
        # signature of an empty reward channel: the critic collapses to Q=0 and
        # the actor has nothing to maximise.
        sr = status.get("success_rate")
        rl_line = []
        if sr is not None:
            rl_line.append(f"success(20) {sr:.2f}")
        if status.get("successes") is not None:
            rl_line.append(f"rewarded {status['successes']}")
        if status.get("buffer") is not None:
            rl_line.append(f"buffer {status['buffer']}")
        if status.get("updates") is not None:
            rl_line.append(f"updates {int(status['updates'])}")
        if rl_line:
            put("  ".join(rl_line), 10, y, red if sr == 0.0 and status.get("buffer") else white, 0.5)
            y += 22

        metrics = status.get("metrics") or {}
        shown = [(k, metrics[k]) for k in ("critic_loss", "q_mean", "actor_loss", "bc_dist") if k in metrics]
        if shown:
            put("  ".join(f"{k.split('_')[0]} {v:.4f}" for k, v in shown), 10, y, grey, 0.5)
            y += 22

        hint = status.get("hint")
        if hint:
            self._put_fit(panel, str(hint), 10, y, grey, 0.45, 1, width - 20)


class RolloutView:
    """Thin driver: owns the window, the status dict, and the per-step pump.

    Degrades to a no-op when disabled or when OpenCV/DISPLAY is missing, so a
    headless training run needs no config change.
    """

    def __init__(
        self,
        env,
        *,
        enabled: bool = True,
        max_width: int = 960,
        panel_height: int = 200,
        tile_height: int = 320,
        window_name: str = "RLT Rollout",
        min_period_s: float = 0.0,
    ):
        self.env = env
        self._status: dict = {}
        self._last_render = 0.0
        self.min_period_s = min_period_s
        self.quit_requested = False
        self._active = False
        self._view = (
            RolloutStatusView(
                camera_names=None,
                bgr=True,
                max_width=max_width,
                window_name=window_name,
                panel_height=panel_height,
                tile_height=tile_height,
            )
            if enabled
            else None
        )

    def start(self) -> RolloutView:
        if self._view is not None:
            self._view.start()
            self._active = True
        return self

    def stop(self) -> None:
        if self._view is not None:
            self._view.stop()
            self._active = False

    def set_active(self, active: bool) -> None:
        """Open or close the window without discarding the view.

        `StatusView.start()` re-imports cv2 and re-probes DISPLAY every time, so
        a headless run would pay that probe once per chunk if this were called
        unconditionally; only transitions act.
        """
        if self._view is None or active == self._active:
            return
        if active:
            self.start()
        else:
            self.stop()

    @property
    def enabled(self) -> bool:
        return self._view is not None and self._view.enabled

    def set(self, **fields) -> None:
        """Merge coarse state that only changes at chunk or episode boundaries."""
        self._status.update(fields)

    def on_step(self, obs=None, **fields) -> bool:
        """Draw one frame. Returns False once the operator asked to quit."""
        if not self.enabled:
            return True
        if self.min_period_s:
            now = time.monotonic()
            if now - self._last_render < self.min_period_s:
                return not self.quit_requested
            self._last_render = now
        self._status.update(fields)
        self._view.update(self._frames(obs), self._status)
        if not self._view.render_once():
            self.quit_requested = True
        return not self.quit_requested

    def _frames(self, obs) -> dict | None:
        if obs is None:
            return None
        render = getattr(self.env, "render_frames", None)
        if render is None:
            return None
        try:
            return render(obs)
        except Exception:
            return None
