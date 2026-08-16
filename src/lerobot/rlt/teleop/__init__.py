"""Human-in-the-loop for RLT online RL.

Two independent channels, matching the paper's system (Sec. V):

* **Sparse outcome labels** (`keys`) — a human supervisor ends each episode with
  a success/failure keypress; success is the only source of reward (r_T = 1).
  The same listener carries the critical-phase handover key.
* **Teleoperated interventions** (`base`, `device`, `piper_leader`) — the
  operator takes over mid-episode. The taken-over actions replace both the
  executed action *and* the stored VLA reference in the replay buffer, which is
  what lets the actor's BC term pull toward human corrections rather than toward
  the VLA's failed attempt.

`piper_leader` is imported lazily: it pulls in the Piper env and the CAN stack,
which a simulation run has no reason to load.
"""

from .base import InterventionManager, InterventionResult
from .device import DeviceIntervention, build_sim_intervention
from .keys import (
    KEY_DISCARD,
    KEY_FAILURE,
    KEY_HANDOVER,
    KEY_INTERVENE,
    KEY_QUIT,
    KEY_SUCCESS,
    KeyboardEventListener,
    key_backend_candidates,
    start_key_backend,
    wait_for_key,
)

__all__ = [
    "KEY_DISCARD",
    "KEY_FAILURE",
    "KEY_HANDOVER",
    "KEY_INTERVENE",
    "KEY_QUIT",
    "KEY_SUCCESS",
    "DeviceIntervention",
    "InterventionManager",
    "InterventionResult",
    "KeyboardEventListener",
    "build_sim_intervention",
    "key_backend_candidates",
    "start_key_backend",
    "wait_for_key",
]
