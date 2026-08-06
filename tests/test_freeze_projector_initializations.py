import json
from pathlib import Path

import pytest

from freeze_projector_initializations import projector_config_from_source


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_sidecar_metadata_is_not_forwarded_to_projector_config():
    raw = json.loads(
        (
            ROOT / "configs/qwen2.5-3b-projector-moonvit-v1-community.json"
        ).read_text(encoding="utf-8")
    )
    config = projector_config_from_source(raw)
    assert config.vision_width == 1152
    assert config.language_width == 4096
    assert config.projector_width == 4608
    assert not hasattr(config, "schema_version")


def test_projector_config_rejects_missing_required_width():
    with pytest.raises(ValueError, match="language_width"):
        projector_config_from_source({"vision_width": 1024})
