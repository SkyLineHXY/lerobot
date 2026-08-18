# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from draccus.cfgparsing import load_config

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig

logger = getLogger(__name__)


def _explicit_policy_keys() -> set[str]:
    """Names of the `policy` fields the user set, in the `--config_path` file or on the CLI."""
    keys = {arg.lstrip("-").split("=")[0].split(".")[0] for arg in parser.get_cli_overrides("policy") or []}
    config_path = parser.parse_arg("config_path")
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            raw = load_config(f) or {}
        policy = raw.get("policy")
        if isinstance(policy, dict):
            keys |= set(policy)
    return keys


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    # Explicit consent to execute remote code from the Hub (required for hub environments).
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        elif self.policy is not None and self.policy.pretrained_path is not None:
            # A config file cannot reach `--policy.path` (it is read from the CLI only), so it names the
            # checkpoint through `pretrained_path`. `make_policy` hands whatever config it is given
            # straight to `from_pretrained`, so without reloading here the trained weights would run under
            # this file's hyperparameters — a different `n_action_steps`, no `input_features` — and score
            # near zero with nothing raised. Fields the file sets explicitly still win.
            pretrained_path = self.policy.pretrained_path
            overrides = {
                key: getattr(self.policy, key)
                for key in _explicit_policy_keys() - {"type", "pretrained_path"}
                if hasattr(self.policy, key)
            }
            checkpoint_policy = PreTrainedConfig.from_pretrained(pretrained_path)
            if type(checkpoint_policy) is not type(self.policy):
                raise ValueError(
                    f"Config file asks for policy type '{self.policy.type}' but the checkpoint at "
                    f"{pretrained_path} holds a '{checkpoint_policy.type}' policy."
                )
            for key, value in overrides.items():
                setattr(checkpoint_policy, key, value)
            checkpoint_policy.pretrained_path = pretrained_path
            self.policy = checkpoint_policy

        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random "
                "weights) — every eval metric will come out at chance level. Pass `--policy.path=<dir>` "
                "or set `policy.pretrained_path` in the config file to evaluate a trained checkpoint."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            else:
                self.job_name = (
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                )

            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
