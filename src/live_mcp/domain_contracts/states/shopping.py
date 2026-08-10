"""Shopping state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, global_fact, out


_PRODUCT_EXISTS = lambda: arg("product", "product_id", "product.exists")
_ORDER_EXISTS = lambda: arg("order", "order_id", "order.exists")


SHOPPING_STATE_FACTS = {
    "search_products": facts(),
    "get_product": facts(pre=(_PRODUCT_EXISTS(),)),
    "list_categories": facts(),
    "compare_products": facts(pre=(
        arg("product", "product_ids", "product.exists"),
    )),
    "get_recommendations": facts(),
    "add_to_cart": facts(
        pre=(
            _PRODUCT_EXISTS(),
            arg("product", "product_id", "product.stock_sufficient"),
        ),
        post=(
            arg("product", "product_id", "cart.membership", True),
            global_fact("cart", "cart.contents", "nonempty"),
        ),
    ),
    "update_cart_quantity": facts(pre=(
        arg("product", "product_id", "cart.membership", True),
    )),
    "remove_from_cart": facts(
        pre=(arg("product", "product_id", "cart.membership", True),),
        post=(arg("product", "product_id", "cart.membership", False),),
    ),
    "get_cart": facts(),
    "clear_cart": facts(post=(global_fact(
        "cart", "cart.contents", "empty",
    ),)),
    "apply_coupon": facts(),
    "get_coupons": facts(),
    "checkout": facts(
        pre=(global_fact("cart", "cart.contents", "nonempty"),),
        post=(
            global_fact("cart", "cart.contents", "empty"),
            out("order", "order_id", "order.exists"),
            out("order", "order_id", "order.status", "placed"),
        ),
    ),
    "get_order": facts(pre=(_ORDER_EXISTS(),)),
    "list_orders": facts(),
    "cancel_order": facts(
        pre=(
            _ORDER_EXISTS(),
            arg("order", "order_id", "order.cancellable"),
        ),
        post=(arg("order", "order_id", "order.status", "cancelled"),),
    ),
    "return_order": facts(
        pre=(
            _ORDER_EXISTS(),
            arg("order", "order_id", "order.status", "shipped"),
        ),
        post=(
            arg("order", "order_id", "order.status", "returning"),
            out("return", "return_id", "return.exists"),
        ),
    ),
    "get_return_status": facts(pre=(
        arg("return", "return_id", "return.exists"),
    )),
    "add_review": facts(
        pre=(
            _PRODUCT_EXISTS(),
            arg("product", "product_id", "product.review_eligible"),
            arg("product", "product_id", "product.reviewed_by_user", False),
        ),
        post=(
            arg("product", "product_id", "product.reviewed_by_user", True),
            out("review", "review_id", "review.exists"),
        ),
    ),
    "get_reviews": facts(pre=(_PRODUCT_EXISTS(),)),
    "add_to_wishlist": facts(
        pre=(_PRODUCT_EXISTS(),),
        post=(arg("product", "product_id", "wishlist.membership", True),),
    ),
    "remove_from_wishlist": facts(
        pre=(_PRODUCT_EXISTS(),),
        post=(arg("product", "product_id", "wishlist.membership", False),),
    ),
    "get_wishlist": facts(),
}
