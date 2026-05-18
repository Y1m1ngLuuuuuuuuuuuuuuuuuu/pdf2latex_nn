from src.adapters.mineru_v7_document_ir import map_v7_type_to_block_type, text_from_v7_item
from src.ir import BlockType


def test_figure_caption_role_overrides_raw_table_type() -> None:
    item = {
        "type": "table",
        "layout_role": "figure_caption",
        "text": "Figure 3: Task 1 Experiment 1 Results",
    }
    assert map_v7_type_to_block_type(item) == BlockType.FIGURE
    assert text_from_v7_item(item) == "Figure 3: Task 1 Experiment 1 Results"


def test_table_caption_role_keeps_table_caption_cleaning() -> None:
    item = {
        "type": "table",
        "layout_role": "table_caption",
        "text": "Table 2: Scores. A B 1 2",
    }
    assert map_v7_type_to_block_type(item) == BlockType.TABLE
    assert text_from_v7_item(item).startswith("Table 2: Scores")
