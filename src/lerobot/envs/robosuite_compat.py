# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Import-time workaround for robosuite's hard-coded log file.

``robosuite.utils.log_utils`` builds ``logging.FileHandler("/tmp/robosuite.log")``
at import time (its ``macros.FILE_LOGGING_LEVEL`` defaults to DEBUG). On a shared
machine that path is usually already owned by whoever imported robosuite first, so
every other account gets ``PermissionError`` from ``import robosuite`` — and
therefore from ``import libero`` — before any of our code runs.

The level is read during ``robosuite/__init__``, so it cannot be overridden after
the fact, and there is no environment variable for it. Redirecting the handler is
the only hook left that does not require writing into site-packages.
"""

from __future__ import annotations

import logging
import os
import tempfile

_HARDCODED_LOG = "/tmp/robosuite.log"  # noqa: S108 - robosuite's literal, not our choice
_patched = False


def _fallback_log_path() -> str:
    return os.path.join(tempfile.gettempdir(), f"robosuite-{os.getuid()}.log")


def _writable(path: str) -> bool:
    try:
        with open(path, "a"):
            return True
    except OSError:
        return False


def patch_robosuite_log_path() -> None:
    """Import robosuite with its log file redirected, if the hard-coded path is not ours.

    Idempotent, and a no-op when ``/tmp/robosuite.log`` is already writable. The
    global ``logging.FileHandler`` override is held only for the duration of the
    import that constructs the handler; robosuite is cached afterwards, so
    everything importing it later (libero included) sees stock logging.
    """
    global _patched
    if _patched or _writable(_HARDCODED_LOG):
        return
    _patched = True

    fallback = _fallback_log_path()
    original = logging.FileHandler

    class _RedirectingFileHandler(original):
        def __init__(self, filename, *args, **kwargs):
            if str(filename) == _HARDCODED_LOG:
                filename = fallback
            super().__init__(filename, *args, **kwargs)

    logging.FileHandler = _RedirectingFileHandler
    try:
        import robosuite  # noqa: F401
    except ImportError:
        pass
    finally:
        logging.FileHandler = original
