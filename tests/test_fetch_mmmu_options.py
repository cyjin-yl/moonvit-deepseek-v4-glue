import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fetch_eval_data import format_mmmu_options


def test_options_as_python_literal_string_are_parsed_per_option():
    raw = "['Political instability leading to population decline', 'Climate change', 'B']"

    rendered = format_mmmu_options(raw)

    assert rendered == (
        "Political instability leading to population decline\nClimate change\nB"
    )


def test_options_as_list_still_work():
    assert format_mmmu_options(["opt A", "opt B"]) == "opt A\nopt B"


def test_options_as_plain_string_becomes_one_block():
    assert format_mmmu_options("just one option") == "just one option"
