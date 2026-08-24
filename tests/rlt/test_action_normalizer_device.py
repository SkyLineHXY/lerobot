"""ActionNormalizer must not drag the stage-1 statistics onto its own device.

The stage-1 `NormalizerProcessorStep` migrates its statistics lazily to whatever
device it was last called with. Teleop commands are normalized on the CPU from
the control thread while the rollout worker's assembler thread is inside the very
same step with device-resident observations, so sharing the step makes the
observation path read a CPU mean against a CUDA tensor and crash a few steps into
a takeover.
"""

import threading

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
)
from lerobot.rlt.envs.base import ActionNormalizer, find_action_normalizer
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_DIM = 7
STATE_DIM = 8

# `meta` is enough to expose a device mismatch (its arithmetic refuses CPU
# operands) and keeps the test meaningful on a machine without a GPU.
OTHER_DEVICE = "cuda" if torch.cuda.is_available() else "meta"


def make_preprocessor(device: str) -> PolicyProcessorPipeline:
    stats = {
        ACTION: {
            "mean": np.arange(ACTION_DIM, dtype=np.float32),
            "std": np.full(ACTION_DIM, 2.0, dtype=np.float32),
        },
        OBS_STATE: {
            "mean": np.zeros(STATE_DIM, dtype=np.float32),
            "std": np.ones(STATE_DIM, dtype=np.float32),
        },
    }
    features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    norm_map = {
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
    }
    return PolicyProcessorPipeline(
        steps=[
            DeviceProcessorStep(device=device),
            NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
        ],
        name="test_stage1_preprocessor",
    )


def observation_frame() -> dict:
    return {OBS_STATE: torch.ones(1, STATE_DIM), "task": "insert the red bar into the blue slot"}


def test_normalized_action_matches_the_stage1_transform():
    preprocessor = make_preprocessor("cpu")
    raw = torch.arange(ACTION_DIM, dtype=torch.float32) + 1.0

    out = ActionNormalizer(preprocessor, ACTION_DIM)(raw)

    expected = find_action_normalizer(preprocessor)._normalize_action(raw, inverse=False)
    torch.testing.assert_close(out, expected)


def test_pipeline_stats_stay_on_the_policy_device():
    preprocessor = make_preprocessor(OTHER_DEVICE)
    step = find_action_normalizer(preprocessor)
    preprocessor(observation_frame())
    assert step._tensor_stats[OBS_STATE]["mean"].device.type == OTHER_DEVICE

    ActionNormalizer(preprocessor, ACTION_DIM)(torch.zeros(ACTION_DIM))

    assert step._tensor_stats[OBS_STATE]["mean"].device.type == OTHER_DEVICE


def test_teleop_normalization_does_not_break_concurrent_observation_batching():
    """The regression itself: a takeover crashing the assembler thread."""
    preprocessor = make_preprocessor(OTHER_DEVICE)
    normalizer = ActionNormalizer(preprocessor, ACTION_DIM)
    errors: list[BaseException] = []
    stop = threading.Event()

    def assemble():
        try:
            for _ in range(400):
                preprocessor(observation_frame())
        except BaseException as exc:  # noqa: BLE001 - reported on the main thread
            errors.append(exc)
        finally:
            stop.set()

    thread = threading.Thread(target=assemble, name="assembler")
    thread.start()
    try:
        while not stop.is_set():
            normalizer(torch.zeros(ACTION_DIM))
    finally:
        thread.join(timeout=30.0)

    assert not errors, errors[0]
