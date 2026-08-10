from types import SimpleNamespace

from src.live_mcp.live_state_enrichment import enrich_readonly_entity_records


class _Executor:
    def execute(self, _session_id, call, *, domain):
        assert domain == "shopping"
        observation = (
            {
                "product_id": call.arguments["product_id"],
                "reviews": [],
                "count": 0,
            }
            if call.name == "get_reviews"
            else {
                "product": {
                    "product_id": call.arguments["product_id"],
                    "stock": 3,
                }
            }
        )
        return SimpleNamespace(
            success=True,
            state_changed=False,
            observation=observation,
            error_type=None,
        )


def test_declared_detail_probe_merges_public_observation() -> None:
    records = [{
        "type": "product",
        "id": "prd-1",
        "data": {"product_id": "prd-1", "name": "Drive"},
    }]

    audits = enrich_readonly_entity_records(
        executor=_Executor(),
        session_id="session",
        server_name="shopping",
        entity_records=records,
    )

    assert records[0]["data"]["stock"] == 3
    assert records[0]["data"]["reviews"] == []
    assert audits == [{
        "tool": "get_product",
        "arguments": {"product_id": "prd-1"},
        "entity_type": "product",
        "entity_id": "prd-1",
        "success": True,
        "state_changed": False,
        "error_type": None,
        "output_field_counts": {"product_id": 1},
        "output_field_values": {"product_id": ["prd-1"]},
    }, {
        "tool": "get_reviews",
        "arguments": {"product_id": "prd-1"},
        "entity_type": "product",
        "entity_id": "prd-1",
        "success": True,
        "state_changed": False,
        "error_type": None,
        "output_field_counts": {"product_id": 1, "review_id": 0},
        "output_field_values": {"product_id": ["prd-1"], "review_id": []},
    }]
