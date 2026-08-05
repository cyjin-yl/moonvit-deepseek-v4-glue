from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from extract_layerwise_representations import override_projector_checkpoint


def test_override_projector_checkpoint_records_source_and_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-000050"
    checkpoint.mkdir()
    (checkpoint / "projector.safetensors").write_bytes(b"projector-state")
    config = {
        "checkpoints": [
            {"id": "step-001500", "kind": "trained", "path": "/old/checkpoint"}
        ]
    }

    override_projector_checkpoint(config, checkpoint, "projector-step50")

    assert config["checkpoints"] == [
        {
            "id": "projector-step50",
            "source_id": "step-001500",
            "kind": "trained",
            "path": str(checkpoint),
        }
    ]
    assert config["projector_checkpoint_override"]["directory"] == str(checkpoint)
    assert len(config["projector_checkpoint_override"]["weights_sha256"]) == 64


def test_override_projector_checkpoint_requires_one_source(tmp_path: Path) -> None:
    config = {"checkpoints": [{"id": "a"}, {"id": "b"}]}

    try:
        override_projector_checkpoint(config, tmp_path, "projector-step50")
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multiple source checkpoints must be rejected")
