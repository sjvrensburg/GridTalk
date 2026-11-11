"""pytest setup for the hermit mirror tests.

The tests check GridTalk's mirrored parsers and judge, not hermit. When
`hermit-harness` is not installed (as in CI), register a stand-in for
the two types `gridtalk_task` imports from `hermit.task`, with the same
fields, so the tests run with nothing but pytest installed.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import hermit.task  # noqa: F401
except ImportError:

    @dataclass
    class RunOutput:
        output: Any
        log: Path | None = None
        cost_usd: float | None = None
        wall_s: float = 0.0
        api_speed: dict | None = None
        messages: list | None = None
        tokens: int | None = None
        extra: dict = field(default_factory=dict)

    @dataclass
    class JudgeResult:
        score: float
        feedback: str = ""
        extra: dict = field(default_factory=dict)

    hermit = types.ModuleType("hermit")
    task = types.ModuleType("hermit.task")
    task.RunOutput = RunOutput
    task.JudgeResult = JudgeResult
    hermit.task = task
    sys.modules["hermit"] = hermit
    sys.modules["hermit.task"] = task
