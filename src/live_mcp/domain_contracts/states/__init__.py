"""Handler-audited state facts, split by domain."""

from src.live_mcp.domain_contracts.states.banking import BANKING_STATE_FACTS
from src.live_mcp.domain_contracts.states.calendar import CALENDAR_STATE_FACTS
from src.live_mcp.domain_contracts.states.crm import CRM_STATE_FACTS
from src.live_mcp.domain_contracts.states.email import EMAIL_STATE_FACTS
from src.live_mcp.domain_contracts.states.filesystem import FILESYSTEM_STATE_FACTS
from src.live_mcp.domain_contracts.states.food_delivery import FOOD_DELIVERY_STATE_FACTS
from src.live_mcp.domain_contracts.states.issue_tracker import ISSUE_TRACKER_STATE_FACTS
from src.live_mcp.domain_contracts.states.payments import PAYMENTS_STATE_FACTS
from src.live_mcp.domain_contracts.states.shopping import SHOPPING_STATE_FACTS
from src.live_mcp.domain_contracts.states.team_chat import TEAM_CHAT_STATE_FACTS
from src.live_mcp.contracts.state_relations import predicate_payload


DOMAIN_STATE_FACTS = {
    "banking": BANKING_STATE_FACTS,
    "calendar": CALENDAR_STATE_FACTS,
    "crm": CRM_STATE_FACTS,
    "email": EMAIL_STATE_FACTS,
    "filesystem": FILESYSTEM_STATE_FACTS,
    "food_delivery": FOOD_DELIVERY_STATE_FACTS,
    "issue_tracker": ISSUE_TRACKER_STATE_FACTS,
    "payments": PAYMENTS_STATE_FACTS,
    "shopping": SHOPPING_STATE_FACTS,
    "team_chat": TEAM_CHAT_STATE_FACTS,
}


def domain_state_fact_payload(domain: str) -> dict:
    """Return deterministic typed state provenance for cache fingerprints."""
    return {
        tool_name: {
            "preconditions": [
                predicate_payload(predicate)
                for predicate in facts.preconditions
            ],
            "precondition_groups": [
                [predicate_payload(predicate) for predicate in group]
                for group in facts.precondition_groups
            ],
            "postconditions": [
                predicate_payload(predicate)
                for predicate in facts.postconditions
            ],
        }
        for tool_name, facts in sorted(DOMAIN_STATE_FACTS.get(domain, {}).items())
    }
