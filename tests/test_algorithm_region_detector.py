from __future__ import annotations

from copy import deepcopy

from src.reasoning.algorithm_region_detector import detect_algorithm_candidates


def _payload(items):
    return {"schema_version": "test", "doc_id": "doc", "items": items, "atomic_blocks": []}


def _doc(nodes=None):
    return {"schema_version": "test", "doc_id": "doc", "nodes": nodes or []}


def test_algorithm_caption_detected():
    result = detect_algorithm_candidates(
        _payload([{"id": "a", "type": "text", "text": "Algorithm 1: Training procedure", "page_idx": 0, "bbox": [0, 0, 100, 20]}]),
        _doc(),
    )
    assert len(result["algorithm_caption_candidates"]) == 1
    assert result["algorithm_caption_candidates"][0]["candidate_type"] == "ALGORITHM_CAPTION"


def test_algorithm_reference_not_caption():
    result = detect_algorithm_candidates(
        _payload([{"id": "a", "type": "text", "text": "Algorithm 1 shows the full training loop.", "page_idx": 0, "bbox": [0, 0, 100, 20]}]),
        _doc(),
    )
    assert not result["algorithm_caption_candidates"]
    assert any(c["candidate_type"] == "FALSE_ALGORITHM_REFERENCE" for c in result["all_candidates"])


def test_input_output_require_ensure_body_signal():
    result = detect_algorithm_candidates(
        _payload(
            [
                {
                    "id": "b",
                    "type": "text",
                    "text": "Input: graph G\nOutput: set S\nRequire: k > 0\nEnsure: feasible solution",
                    "page_idx": 0,
                    "bbox": [0, 20, 200, 90],
                }
            ]
        ),
        _doc(),
    )
    assert result["algorithm_body_candidates"]


def test_pseudocode_lines_grouped():
    result = detect_algorithm_candidates(
        _payload(
            [
                {"id": "c1", "type": "text", "text": "for i = 1 to n do", "page_idx": 0, "bbox": [0, 20, 200, 35]},
                {"id": "c2", "type": "text", "text": "if score > best then return x", "page_idx": 0, "bbox": [0, 38, 200, 55]},
            ]
        ),
        _doc(),
    )
    assert result["algorithm_region_candidates"]
    assert result["algorithm_region_candidates"][0]["body_candidate_ids"]


def test_code_config_block_compile_risk():
    result = detect_algorithm_candidates(
        _payload(
            [
                {
                    "id": "cfg",
                    "type": "text",
                    "text": "learning_rate=float(1e-5,1e-1), max_depth=choice(2,4), min_child_weight=1",
                    "page_idx": 0,
                    "bbox": [0, 20, 250, 45],
                }
            ]
        ),
        _doc(),
    )
    assert result["algorithm_body_candidates"]
    assert result["pseudocode_compile_risk"]


def test_table_like_pseudocode_not_normal_table():
    result = detect_algorithm_candidates(
        _payload(
            [
                {
                    "id": "tbl",
                    "type": "table",
                    "text": "Input: X\nfor each row do\nreturn prediction",
                    "page_idx": 0,
                    "bbox": [0, 20, 250, 70],
                }
            ]
        ),
        _doc(),
    )
    assert result["algorithm_body_candidates"][0]["candidate_type"] == "ALGORITHM_AS_TABLE_LIKE"


def test_only_caption_failure_hint():
    result = detect_algorithm_candidates(
        _payload([{"id": "cap", "type": "text", "text": "Procedure 2: Search", "page_idx": 0, "bbox": [0, 0, 100, 20]}]),
        _doc(),
    )
    assert result["algorithm_region_candidates"][0]["failure_hint"] == "CAPTION_EXISTS_BODY_MISSING"


def test_only_body_failure_hint():
    result = detect_algorithm_candidates(
        _payload([{"id": "body", "type": "text", "text": "Input: X\nOutput: y\nreturn y", "page_idx": 0, "bbox": [0, 0, 100, 50]}]),
        _doc(),
    )
    assert result["algorithm_region_candidates"][0]["failure_hint"] == "BODY_EXISTS_CAPTION_MISSING"


def test_no_v8_mutation():
    content = _payload([{"id": "x", "type": "text", "text": "Algorithm 1: Test", "page_idx": 0, "bbox": [0, 0, 100, 20]}])
    original = deepcopy(content)
    detect_algorithm_candidates(content, _doc())
    assert content == original


def test_no_renderer_mutation_contract():
    result = detect_algorithm_candidates(_payload([{"id": "x", "type": "text", "text": "Algorithm 1: Test", "page_idx": 0, "bbox": [0, 0, 100, 20]}]), _doc())
    assert "render_tree" not in result
    assert "generated_tex" not in result
