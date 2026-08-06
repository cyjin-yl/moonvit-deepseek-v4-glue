import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import manifest_experiment_artifacts


def test_manifest_writer_uses_git_stable_lf_bytes(tmp_path, monkeypatch):
    (tmp_path / "payload.txt").write_bytes(b"payload\n")
    monkeypatch.setattr(sys, "argv", ["manifest_experiment_artifacts.py", str(tmp_path)])

    manifest_experiment_artifacts.main()

    encoded = (tmp_path / "ARTIFACT_MANIFEST.json").read_bytes()
    assert encoded.endswith(b"\n")
    assert b"\r\n" not in encoded
