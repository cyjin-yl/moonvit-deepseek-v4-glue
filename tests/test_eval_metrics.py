from moonvit_glue.metrics import (
    anls,
    exact_match,
    extract_point,
    grounding_metrics,
    levenshtein,
    normalize_answer,
    score_record,
    soft_vqa_accuracy,
    summarize,
    token_f1,
)


def test_normalize_answer_matches_vqa_convention():
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("  an   APPLE ") == "apple"


def test_exact_match_uses_any_reference():
    assert exact_match("Bus", ["a bus", "car"]) == 1.0
    assert exact_match("train", ["a bus", "car"]) == 0.0


def test_soft_vqa_accuracy_counts_human_agreement():
    answers = ["yes", "yes", "yes", "no"]
    assert soft_vqa_accuracy("yes", answers) == 1.0
    assert soft_vqa_accuracy("no", answers) == 1 / 3
    assert soft_vqa_accuracy("maybe", answers) == 0.0
    assert soft_vqa_accuracy("", answers) == 0.0


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0


def test_anls_threshold_and_identity():
    assert anls("invoice", ["invoice"]) == 1.0
    assert anls("", [""]) == 1.0
    assert anls("completely wrong", ["totally different"]) == 0.0
    # Small typo keeps similarity above the 0.5 threshold.
    score = anls("inv0ice", ["invoice"])
    assert 0.5 < score < 1.0


def test_token_f1():
    assert token_f1("a red bus", "red bus") == 1.0
    assert token_f1("train", "bus") == 0.0
    assert 0.0 < token_f1("red bus station", "red bus") < 1.0


def test_extract_point_formats():
    assert extract_point("click at (512, 341)") == (512.0, 341.0)
    assert extract_point("<point>(512, 341)</point>") == (512.0, 341.0)
    assert extract_point('<|point|>(0.5, 0.25)') == (0.5, 0.25)
    assert extract_point('{"x": 512, "y": 341}') == (512.0, 341.0)
    assert extract_point("[512, 341]") == (512.0, 341.0)
    assert extract_point("no coordinates here") is None
    assert extract_point("(512,)") is None


def test_grounding_metrics_point_distance():
    hit = grounding_metrics("(500, 500)", gt_point=(505.0, 495.0))
    assert hit["parse_ok"] and hit["correct"]
    assert abs(hit["error"] - (50.0**0.5)) < 1e-6

    miss = grounding_metrics("(100, 100)", gt_point=(900.0, 900.0))
    assert miss["parse_ok"] and not miss["correct"]
    assert miss["error"] > 1000.0

    unparsed = grounding_metrics("I cannot tell", gt_point=(0.0, 0.0))
    assert not unparsed["parse_ok"] and not unparsed["correct"]
    assert unparsed["error"] is None


def test_grounding_metrics_fractional_predictions_are_scaled():
    hit = grounding_metrics("(0.5, 0.5)", gt_point=(499.5, 499.5), scale=999.0)
    assert hit["parse_ok"] and hit["correct"]


def test_grounding_metrics_box_containment_and_center_error():
    inside = grounding_metrics("(500, 500)", gt_box=[400.0, 400.0, 600.0, 600.0])
    assert inside["correct"] and abs(inside["error"]) < 1e-6

    outside = grounding_metrics("(100, 100)", gt_box=[400.0, 400.0, 600.0, 600.0])
    assert not outside["correct"]


def test_grounding_requires_exactly_one_target():
    try:
        grounding_metrics("(0, 0)")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no target is given")


def test_score_record_dispatch_and_summary():
    records = [
        score_record("bus", {"metric": "exact_match", "answers": ["a bus"]}),
        score_record("(0, 0)", {"metric": "grounding", "gt_point": [10.0, 10.0]}),
        score_record("no idea", {"metric": "grounding", "gt_point": [10.0, 10.0]}),
    ]
    summary = summarize(records)
    assert summary["count"] == 3
    assert summary["exact_match"] == 1.0
    assert summary["grounding_count"] == 2
    assert summary["parse_rate"] == 0.5
    assert summary["accuracy"] == 0.5
    assert abs(summary["mean_error"] - (200.0**0.5)) < 1e-6
