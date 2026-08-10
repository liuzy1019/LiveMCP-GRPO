"""Audited output fields that can participate in PROVE value flow.

Every tool is present.  An empty tuple means the handler was reviewed and does
not expose a stable field useful as a downstream required argument.
"""

from __future__ import annotations

from src.live_mcp.domain_contracts.dependency import _DEPENDENCY_TOOL_OUTPUT_FIELDS
from src.live_mcp.domain_contracts.states import DOMAIN_STATE_FACTS


DOMAIN_VALUE_OUTPUT_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    domain: {tool_name: () for tool_name in state_facts}
    for domain, state_facts in DOMAIN_STATE_FACTS.items()
}

for domain, tool_fields in _DEPENDENCY_TOOL_OUTPUT_FIELDS.items():
    for tool_name, fields in tool_fields.items():
        DOMAIN_VALUE_OUTPUT_FIELDS[domain][tool_name] = tuple(fields)

_ADDITIONAL_HANDLER_OUTPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "crm": {
        "get_deal": ("deal_id", "contact_id", "lead_id"),
        "list_tasks": ("task_id", "deal_id", "contact_id"),
    },
    "email": {
        "send_email": ("email_id", "thread_id"),
        "create_draft": ("email_id",),
        "forward_email": ("email_id", "thread_id"),
        "reply_email": ("email_id", "thread_id"),
    },
    "filesystem": {
        "pwd": ("cwd",),
        "ls": ("path",),
        "find": ("path",),
        "mkdir": ("path",),
        "touch": ("path",),
        "mv": ("source", "target"),
        "cp": ("source", "target"),
        "symlink": ("link_path", "target"),
        "readlink": ("path", "target"),
        "tar_create": ("archive",),
        "tar_extract": ("archive", "extracted_paths"),
        "zip": ("archive",),
        "unzip": ("archive", "extracted_paths"),
        "split": ("path", "parts"),
    },
    "food_delivery": {
        "search_restaurants": ("restaurant_id",),
        "get_order": ("order_id", "restaurant_id"),
    },
    "issue_tracker": {
        "list_members": ("user_id",),
        "get_issue": ("issue_id", "sprint_id", "assignee"),
        "list_sprints": ("sprint_id",),
        "list_subtasks": ("subtask_id", "issue_id", "assignee"),
        "get_time_report": ("entry_id", "issue_id", "assignee"),
    },
    "team_chat": {
        "get_channel": ("channel_id",),
        "send_message": ("message_id", "channel_id", "thread_id"),
        "send_dm": ("dm_id",),
        "create_thread": ("thread_id", "channel_id", "message_id"),
        "get_thread": ("thread_id", "channel_id", "message_id"),
        "search_messages": ("message_id", "channel_id", "thread_id"),
    },
}

for domain, tool_fields in _ADDITIONAL_HANDLER_OUTPUTS.items():
    for tool_name, fields in tool_fields.items():
        existing = DOMAIN_VALUE_OUTPUT_FIELDS[domain][tool_name]
        DOMAIN_VALUE_OUTPUT_FIELDS[domain][tool_name] = tuple(sorted(
            set(existing) | set(fields)
        ))

for domain, tool_facts in DOMAIN_STATE_FACTS.items():
    for tool_name, facts in tool_facts.items():
        postcondition_outputs = {
            predicate.subject.name
            for predicate in facts.postconditions
            if predicate.subject.source == "output"
        }
        existing = DOMAIN_VALUE_OUTPUT_FIELDS[domain][tool_name]
        DOMAIN_VALUE_OUTPUT_FIELDS[domain][tool_name] = tuple(sorted(
            set(existing) | postcondition_outputs
        ))
