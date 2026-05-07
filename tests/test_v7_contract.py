import json

import pytest

from src.pipeline.v7_contract import V7ContractError, assert_v7_content_json, assert_v7_graph_data


def test_v7_content_contract_accepts_styled_payload(tmp_path):
    path = tmp_path / "paper_content_list_v7_styles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "content_v7_columnfix_listmarkers_with_styles",
                "items": [{"text": "Intro", "bbox": [0, 0, 10, 10]}],
            }
        ),
        encoding="utf-8",
    )

    payload = assert_v7_content_json(path, require_styles=True)

    assert payload["items"][0]["text"] == "Intro"


def test_v7_content_contract_rejects_old_visual_order_payload(tmp_path):
    path = tmp_path / "paper_content_list_v4.json"
    path.write_text(
        json.dumps({"schema_version": "content_v4_visual_order", "items": []}),
        encoding="utf-8",
    )

    with pytest.raises(V7ContractError):
        assert_v7_content_json(path)


def test_v7_graph_contract_accepts_metadata_or_v7_source_path():
    class Graph:
        pipeline_version = "v7"
        graph_schema_version = "graph_v7"

    assert_v7_graph_data(Graph())

    class LegacyV7Graph:
        source_path = "paper_content_list_v7_styles.json"

    assert_v7_graph_data(LegacyV7Graph())


def test_v7_graph_contract_rejects_ambiguous_graph():
    class Graph:
        source_path = "paper_content_list_v4_styles.json"

    with pytest.raises(V7ContractError):
        assert_v7_graph_data(Graph())
