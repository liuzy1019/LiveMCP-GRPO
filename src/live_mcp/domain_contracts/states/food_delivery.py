"""Food-delivery state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import (
    arg,
    argument_value,
    facts,
    out,
)


FOOD_DELIVERY_STATE_FACTS = {
    "list_restaurants": facts(),
    "search_restaurants": facts(),
    "get_restaurant": facts(pre=(arg("restaurant", "restaurant_id", "restaurant.exists"),)),
    "get_menu": facts(pre=(arg("restaurant", "restaurant_id", "restaurant.exists"),)),
    "filter_by_dietary": facts(pre=(arg("restaurant", "restaurant_id", "restaurant.exists"),)),
    "get_popular_items": facts(pre=(arg("restaurant", "restaurant_id", "restaurant.exists"),)),
    "create_order": facts(
        pre=(
            arg("restaurant", "restaurant_id", "restaurant.exists"),
            arg("restaurant", "restaurant_id", "restaurant.menu_nonempty"),
        ),
        post=(
            out("order", "order_id", "order.exists"),
            out("order", "order_id", "order.status", "placed"),
            out("order", "order_id", "order.tip_present", False),
            out("order", "order_id", "order.rated", False),
        ),
    ),
    "get_order": facts(pre=(arg("order", "order_id", "order.exists"),)),
    "list_orders": facts(),
    "update_order_status": facts(pre=(
        arg("order", "order_id", "order.exists"),
        arg("order", "order_id", "order.transition_allowed"),
    ), post=(
        arg(
            "order", "order_id", "order.status",
            argument_value("status"),
        ),
    )),
    "cancel_order": facts(
        pre=(
            arg("order", "order_id", "order.exists"),
            arg("order", "order_id", "order.cancellable"),
        ),
        post=(arg("order", "order_id", "order.status", "cancelled"),),
    ),
    "get_estimated_time": facts(pre=(arg("order", "order_id", "order.exists"),)),
    "track_rider": facts(pre=(
        arg("order", "order_id", "order.exists"),
        arg("order", "order_id", "order.status", "delivering"),
    )),
    "rate_order": facts(
        pre=(
            arg("order", "order_id", "order.exists"),
            arg("order", "order_id", "order.status", "delivered"),
            arg("order", "order_id", "order.rated", False),
        ),
        post=(arg("order", "order_id", "order.rated", True),),
    ),
    "add_tip": facts(
        pre=(
            arg("order", "order_id", "order.exists"),
            arg("order", "order_id", "order.tip_present", False),
        ),
        post=(arg("order", "order_id", "order.tip_present", True),),
    ),
    "reorder": facts(
        pre=(arg("order", "order_id", "order.exists"),),
        post=(
            out("order", "order_id", "order.exists"),
            out("order", "order_id", "order.status", "placed"),
        ),
    ),
    "contact_support": facts(
        pre=(arg("order", "order_id", "order.exists"),),
        post=(out("ticket", "ticket_id", "ticket.exists"),),
    ),
}
