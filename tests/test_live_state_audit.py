from src.live_mcp.contracts.factory import build_contract_registry
from src.live_mcp.live_state_availability import build_live_state_availability
from src.live_mcp.servers.crm.server import TOOLS as CRM_TOOLS
from src.live_mcp.servers.shopping.server import TOOLS as SHOPPING_TOOLS


CREATE_ORDER = {
    "name": "create_order",
    "input_schema": {
        "properties": {
            "restaurant_id": {}, "items": {}, "delivery_address": {},
        },
        "required": ["restaurant_id", "items", "delivery_address"],
    },
    "annotations": {"readonly": False, "mutating": True},
}


def test_availability_separates_unknown_state_from_known_unusable_state() -> None:
    tools = [CREATE_ORDER]
    ids = [
        {"type": "restaurant", "id": "unknown"},
        {"type": "restaurant", "id": "empty"},
        {"type": "restaurant", "id": "ready"},
    ]
    records = [
        {"type": "restaurant", "id": "unknown", "data": {"name": "A"}},
        {"type": "restaurant", "id": "empty", "data": {"items": []}},
        {"type": "restaurant", "id": "ready", "data": {"items": [{"id": "x"}]}},
    ]

    report = build_live_state_availability(
        server_name="food_delivery",
        tool_schemas=tools,
        entity_ids=ids,
        entity_records=records,
        contract_registry=build_contract_registry({"food_delivery": tools}),
    )

    target = report["target_tool_availability"]["create_order"]
    assert target["state_known_by_type"] == {"restaurant": 2}
    assert target["state_unknown_by_type"] == {"restaurant": 1}
    assert target["usable_by_type"] == {"restaurant": 1}


def test_registered_target_state_fields_fail_closed() -> None:
    report = build_live_state_availability(
        server_name="food_delivery",
        tool_schemas=[CREATE_ORDER],
        entity_ids=[
            {"type": "restaurant", "id": "unknown"},
            {"type": "restaurant", "id": "known"},
        ],
        entity_records=[
            {"type": "restaurant", "id": "unknown", "data": {"name": "A"}},
            {"type": "restaurant", "id": "known", "data": {"menu": []}},
        ],
        contract_registry=build_contract_registry({
            "food_delivery": [CREATE_ORDER],
        }),
    )

    target = report["target_tool_availability"]["create_order"]
    assert target["state_known_by_type"] == {"restaurant": 1}
    assert target["state_unknown_by_type"] == {"restaurant": 1}
    assert target["usable_by_type"] == {"restaurant": 0}


def test_any_of_entity_group_accepts_one_live_crm_alternative() -> None:
    create_deal = next(tool for tool in CRM_TOOLS if tool["name"] == "create_deal")
    report = build_live_state_availability(
        server_name="crm",
        tool_schemas=[create_deal],
        entity_ids=[{"type": "contact", "id": "contact-1"}],
        entity_records=[{
            "type": "contact", "id": "contact-1", "data": {"name": "A"},
        }],
        contract_registry=build_contract_registry({"crm": [create_deal]}),
    )

    target = report["target_tool_availability"]["create_deal"]
    assert target["required_entity_types"] == []
    assert target["alternative_entity_groups"] == [{
        "entity_types": ["contact", "lead"],
        "state_known": True,
        "has_usable_entity": True,
    }]
    assert target["has_usable_entities"] is True


def test_global_cart_state_is_required_for_checkout() -> None:
    checkout = next(tool for tool in SHOPPING_TOOLS if tool["name"] == "checkout")
    registry = build_contract_registry({"shopping": [checkout]})
    empty = build_live_state_availability(
        server_name="shopping",
        tool_schemas=[checkout],
        entity_ids=[],
        entity_records=[],
        probe_results=[{
            "tool": "get_cart", "success": True, "state_changed": False,
        }],
        contract_registry=registry,
    )["target_tool_availability"]["checkout"]
    nonempty = build_live_state_availability(
        server_name="shopping",
        tool_schemas=[checkout],
        entity_ids=[{"type": "cart_item", "id": "product-1"}],
        entity_records=[{
            "type": "cart_item", "id": "product-1", "data": {"cart_member": True},
        }],
        probe_results=[{
            "tool": "get_cart", "success": True, "state_changed": False,
        }],
        contract_registry=registry,
    )["target_tool_availability"]["checkout"]

    assert empty["global_state_known"] is True
    assert empty["has_usable_entities"] is False
    assert nonempty["has_usable_entities"] is True
