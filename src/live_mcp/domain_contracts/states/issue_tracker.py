"""Issue-tracker state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import (
    arg,
    argument_value,
    facts,
    out,
)


_ISSUE_EXISTS = lambda: arg("issue", "issue_id", "issue.exists")


ISSUE_TRACKER_STATE_FACTS = {
    "create_issue": facts(post=(
        out("issue", "issue_id", "issue.exists"),
        out("issue", "issue_id", "issue.state", "open"),
        out("issue", "issue_id", "issue.assigned", False),
        out("issue", "issue_id", "issue.in_sprint", False),
    )),
    "get_issue": facts(pre=(_ISSUE_EXISTS(),)),
    "list_issues": facts(),
    "list_members": facts(),
    "update_issue": facts(pre=(_ISSUE_EXISTS(),)),
    "assign_issue": facts(
        pre=(
            _ISSUE_EXISTS(),
            arg("user", "assignee", "user.exists"),
        ),
        post=(arg("issue", "issue_id", "issue.assigned", True),),
    ),
    "transition_issue": facts(
        pre=(
            _ISSUE_EXISTS(),
            arg("issue", "issue_id", "issue.transition_allowed"),
        ),
        post=(arg(
            "issue", "issue_id", "issue.state", argument_value("state"),
        ),),
    ),
    "comment_issue": facts(pre=(_ISSUE_EXISTS(),)),
    "add_label": facts(pre=(_ISSUE_EXISTS(),)),
    "remove_label": facts(pre=(_ISSUE_EXISTS(),)),
    "add_watcher": facts(pre=(
        _ISSUE_EXISTS(),
        arg("user", "user", "user.exists"),
    )),
    "remove_watcher": facts(pre=(_ISSUE_EXISTS(),)),
    "create_sprint": facts(post=(
        out("sprint", "sprint_id", "sprint.exists"),
        out("sprint", "sprint_id", "sprint.status", "active"),
    )),
    "list_sprints": facts(),
    "add_to_sprint": facts(
        pre=(
            _ISSUE_EXISTS(),
            arg("sprint", "sprint_id", "sprint.exists"),
        ),
        post=(arg("issue", "issue_id", "issue.in_sprint", True),),
    ),
    "remove_from_sprint": facts(
        pre=(_ISSUE_EXISTS(),),
        post=(arg("issue", "issue_id", "issue.in_sprint", False),),
    ),
    "create_subtask": facts(
        pre=(_ISSUE_EXISTS(),),
        post=(
            out("subtask", "subtask_id", "subtask.exists"),
            out("subtask", "subtask_id", "subtask.status", "open"),
        ),
    ),
    "list_subtasks": facts(pre=(_ISSUE_EXISTS(),)),
    "time_track": facts(
        pre=(_ISSUE_EXISTS(),),
        post=(out("time_entry", "entry_id", "time_entry.exists"),),
    ),
    "get_time_report": facts(),
    "set_milestone": facts(pre=(_ISSUE_EXISTS(),)),
}
