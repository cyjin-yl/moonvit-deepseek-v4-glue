"""Prefetch shard filtering: split/prefix matching must be exact and config-aware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from prefetch_parquet import matches_shard


def test_matches_split_files_only():
    assert matches_shard("data/train-00000-of-00002-abc.parquet", "train", "")
    assert not matches_shard("data/test-00000-of-00001-abc.parquet", "train", "")
    assert not matches_shard("README.md", "train", "")


def test_validation_split_does_not_match_train_prefix_loosely():
    # "validation-" contains "train" as a substring after position 4 — the
    # matcher must anchor on "split-": train must NOT match validation files.
    assert not matches_shard("data/validation-00000-of-00001-abc.parquet", "train", "")


def test_config_prefix_scans_only_that_config_dir():
    assert matches_shard("standard/test-00000-of-00001-abc.parquet", "test", "standard/")
    assert not matches_shard("vision/test-00000-of-00001-abc.parquet", "test", "standard/")
