"""Feature-cache CLI keeps transport failures separate from data failures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cache_moonvit_features


def test_progress_broken_pipe_does_not_fail_a_cache_record(monkeypatch):
    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("detached client")

    monkeypatch.setattr("builtins.print", broken_print)

    cache_moonvit_features.emit("valid row")
