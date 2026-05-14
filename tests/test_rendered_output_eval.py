import numpy as np

from src.evaluation.compile_eval import summarize_latex_error
from src.evaluation.visual_qa import compare_gray_pages


def test_compare_gray_pages_scores_identical_layout() -> None:
    image = np.full((80, 60), 255, dtype=np.uint8)
    image[10:30, 8:52] = 0
    image[45:60, 8:40] = 0
    report = compare_gray_pages(image, image.copy(), ink_threshold=245)
    assert report["layout_similarity"] > 0.999
    assert report["ink_bbox_iou"] == 1.0


def test_compare_gray_pages_penalizes_shifted_layout() -> None:
    gold = np.full((80, 60), 255, dtype=np.uint8)
    pred = np.full((80, 60), 255, dtype=np.uint8)
    gold[10:30, 8:52] = 0
    pred[40:60, 8:52] = 0
    report = compare_gray_pages(gold, pred, ink_threshold=245)
    assert report["layout_similarity"] < 0.95


def test_summarize_latex_error_prefers_bang_line() -> None:
    log = "line one\n! Undefined control sequence.\nfoo\nbar"
    assert "Undefined control sequence" in summarize_latex_error(log)
