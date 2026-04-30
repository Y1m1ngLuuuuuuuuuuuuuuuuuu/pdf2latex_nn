from src.perception.style_spans import (
    baseline_font_size,
    bucket_font_size,
    is_inline_math_font,
    merge_raw_spans,
    normalize_font_name,
)


def test_normalize_font_name_removes_subset_prefix():
    assert normalize_font_name("ABCDEE+NimbusRomNo9L-Regu") == "nimbusromno9l-regu"


def test_math_font_detection_is_narrower_than_plain_cm():
    assert is_inline_math_font("cmmi10", "x")
    assert is_inline_math_font("cmsy10", "=")
    assert is_inline_math_font("msbm10", "R")
    assert not is_inline_math_font("cmr10", "normal text")
    assert not is_inline_math_font("cmbx10", "bold text")


def test_symbol_font_requires_math_like_text():
    assert is_inline_math_font("symbol", "≤")
    assert not is_inline_math_font("symbol", "plain")


def test_merge_raw_spans_state_machine_merges_equal_style_only():
    spans = [
        {
            "text": "hello",
            "font_name": "times",
            "font_size": 10.0,
            "is_bold": False,
            "is_italic": False,
            "is_inline_math": False,
            "is_inline_code": False,
        },
        {
            "text": " world",
            "font_name": "times",
            "font_size": 10.0,
            "is_bold": False,
            "is_italic": False,
            "is_inline_math": False,
            "is_inline_code": False,
        },
        {
            "text": "!",
            "font_name": "times-bold",
            "font_size": 10.0,
            "is_bold": True,
            "is_italic": False,
            "is_inline_math": False,
            "is_inline_code": False,
        },
    ]

    merged = merge_raw_spans(spans)

    assert len(merged) == 2
    assert merged[0]["text"] == "hello world"
    assert merged[0]["char_count"] == 11
    assert merged[1]["text"] == "!"


def test_baseline_font_size_uses_char_weighted_mode():
    spans = [
        {"font_size": 9.0, "char_count": 5},
        {"font_size": 10.0, "char_count": 20},
        {"font_size": 9.0, "char_count": 5},
    ]

    assert baseline_font_size(spans) == 10.0


def test_bucket_font_size_rounds_to_half_points():
    assert bucket_font_size(9.24) == 9.0
    assert bucket_font_size(9.26) == 9.5
