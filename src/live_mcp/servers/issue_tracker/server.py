"""Stateful issue tracker — 20 tools (PROVE-aligned).
Workflow transitions: open→in_progress→in_review→resolved→closed.
Features: sprints, labels, watchers, subtasks, time tracking, milestones.
"""

from __future__ import annotations
from typing import Any
from src.live_mcp.server_base import StatefulToolServer, _result, serve

TRANSITIONS = {"open": ["in_progress", "cancelled"], "in_progress": ["in_review", "blocked"], "in_review": ["resolved", "in_progress"], "resolved": ["closed", "in_progress"], "blocked": ["in_progress", "cancelled"], "closed": [], "cancelled": []}

TOOLS = [
    {"name": "create_issue", "description": "Create a new issue.", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "description": {"type": "string"}, "priority": {"type": "string"}, "labels": {"type": "array"}}, "required": ["title"]}, "annotations": {"mutating": True}},
    {"name": "get_issue", "description": "Get full issue details.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}}, "required": ["issue_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "list_issues", "description": "List issues with filters.", "input_schema": {"type": "object", "properties": {"state": {"type": "string"}, "assignee": {"type": "string"}, "priority": {"type": "string"}, "sprint_id": {"type": "string"}, "label": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "update_issue", "description": "Update issue fields (title, description, priority).", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "fields": {"type": "object"}}, "required": ["issue_id", "fields"]}, "annotations": {"mutating": True}},
    {"name": "assign_issue", "description": "Assign issue to a team member.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "assignee": {"type": "string"}}, "required": ["issue_id", "assignee"]}, "annotations": {"mutating": True}},
    {"name": "transition_issue", "description": "Transition issue to a new workflow state.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "state": {"type": "string"}, "comment": {"type": "string"}}, "required": ["issue_id", "state"]}, "annotations": {"mutating": True}},
    {"name": "comment_issue", "description": "Add a comment to an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "body": {"type": "string"}}, "required": ["issue_id", "body"]}, "annotations": {"mutating": True}},
    {"name": "add_label", "description": "Add a label to an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "label": {"type": "string"}}, "required": ["issue_id", "label"]}, "annotations": {"mutating": True}},
    {"name": "remove_label", "description": "Remove a label from an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "label": {"type": "string"}}, "required": ["issue_id", "label"]}, "annotations": {"mutating": True}},
    {"name": "add_watcher", "description": "Add a watcher to an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "user": {"type": "string"}}, "required": ["issue_id", "user"]}, "annotations": {"mutating": True}},
    {"name": "remove_watcher", "description": "Remove a watcher from an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "user": {"type": "string"}}, "required": ["issue_id", "user"]}, "annotations": {"mutating": True}},
    {"name": "create_sprint", "description": "Create a new sprint.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "goal": {"type": "string"}}, "required": ["name", "start_date", "end_date"]}, "annotations": {"mutating": True}},
    {"name": "list_sprints", "description": "List sprints by status.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "add_to_sprint", "description": "Add an issue to a sprint.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "sprint_id": {"type": "string"}}, "required": ["issue_id", "sprint_id"]}, "annotations": {"mutating": True}},
    {"name": "remove_from_sprint", "description": "Remove an issue from a sprint.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}}, "required": ["issue_id"]}, "annotations": {"mutating": True}},
    {"name": "create_subtask", "description": "Create a subtask under an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "title": {"type": "string"}, "assignee": {"type": "string"}}, "required": ["issue_id", "title"]}, "annotations": {"mutating": True}},
    {"name": "list_subtasks", "description": "List subtasks for an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}}, "required": ["issue_id"]}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "time_track", "description": "Log time spent on an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "hours": {"type": "number"}, "description": {"type": "string"}}, "required": ["issue_id", "hours"]}, "annotations": {"mutating": True}},
    {"name": "get_time_report", "description": "Get time tracking report for issues/sprints.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "sprint_id": {"type": "string"}, "assignee": {"type": "string"}}, "required": []}, "annotations": {"readonly": True, "mutating": False}},
    {"name": "set_milestone", "description": "Set a milestone on an issue.", "input_schema": {"type": "object", "properties": {"issue_id": {"type": "string"}, "milestone": {"type": "string"}}, "required": ["issue_id", "milestone"]}, "annotations": {"mutating": True}},
]

class IssueTrackerServer(StatefulToolServer):
    def __init__(self) -> None:
        super().__init__("issue_tracker", TOOLS)
        self.handlers = {t["name"]: getattr(self, t["name"]) for t in TOOLS}

    def _iss(self, state, iid):
        if iid not in state["issues"]: raise KeyError(f"issue not found: {iid}")
        return state["issues"][iid]

    def create_issue(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); iid = f"iss_{state['next_issue_num']:04d}"; state["next_issue_num"] += 1
        issue = {"issue_id": iid, "title": arguments["title"], "description": arguments.get("description", ""), "priority": arguments.get("priority", "medium"), "labels": arguments.get("labels", []), "state": "open", "assignee": None, "watchers": [], "sprint_id": None, "milestone": None, "comments": [], "created_at": "2026-06-24"}
        state["issues"][iid] = issue
        return _result(True, {"issue": issue}, None, "", True)

    def get_issue(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _result(True, {"issue": self._iss(self._state(session_id), arguments["issue_id"])}, None, "", False)

    def list_issues(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); issues = list(state["issues"].values())
        if arguments.get("state"): issues = [i for i in issues if i["state"] == arguments["state"]]
        if arguments.get("assignee"): issues = [i for i in issues if i["assignee"] == arguments["assignee"]]
        if arguments.get("priority"): issues = [i for i in issues if i["priority"] == arguments["priority"]]
        if arguments.get("sprint_id"): issues = [i for i in issues if i.get("sprint_id") == arguments["sprint_id"]]
        if arguments.get("label"): issues = [i for i in issues if arguments["label"] in i.get("labels", [])]
        return _result(True, {"issues": issues, "count": len(issues)}, None, "", False)

    def update_issue(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        issue = self._iss(self._state(session_id), arguments["issue_id"])
        for k, v in arguments["fields"].items():
            if k in ("title", "description", "priority"): issue[k] = v
        return _result(True, {"issue": issue}, None, "", True)

    def assign_issue(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); issue = self._iss(state, arguments["issue_id"])
        if arguments["assignee"] not in state["members"]: raise KeyError(f"member not found: {arguments['assignee']}")
        issue["assignee"] = arguments["assignee"]
        return _result(True, {"issue": issue}, None, "", True)

    def transition_issue(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); issue = self._iss(state, arguments["issue_id"]); new_state = arguments["state"]
        old_state = issue["state"]; allowed = TRANSITIONS.get(old_state, [])
        if new_state not in allowed: raise KeyError(f"invalid transition: {old_state} -> {new_state}")
        if new_state in ("in_review", "resolved", "closed") and issue["assignee"] is None: raise KeyError("cannot transition unassigned issue")
        issue["state"] = new_state
        if arguments.get("comment"): issue["comments"].append({"author": "system", "body": arguments["comment"]})
        issue["comments"].append({"author": "system", "body": f"State changed: {old_state} -> {new_state}", "timestamp": "2026-06-24"})
        return _result(True, {"issue": issue}, None, "", True)

    def comment_issue(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        issue = self._iss(self._state(session_id), arguments["issue_id"])
        issue["comments"].append({"author": "user", "body": arguments["body"], "timestamp": "2026-06-24"})
        return _result(True, {"issue": issue}, None, "", True)

    def add_label(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        issue = self._iss(self._state(session_id), arguments["issue_id"]); label = arguments["label"]
        if label not in issue.setdefault("labels", []): issue["labels"].append(label)
        return _result(True, {"issue_id": issue["issue_id"], "labels": issue["labels"]}, None, "", True)

    def remove_label(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        issue = self._iss(self._state(session_id), arguments["issue_id"]); label = arguments["label"]
        issue["labels"] = [l for l in issue.get("labels", []) if l != label]
        return _result(True, {"issue_id": issue["issue_id"], "labels": issue["labels"]}, None, "", True)

    def add_watcher(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); issue = self._iss(state, arguments["issue_id"]); user = arguments["user"]
        if user not in state["members"]: raise KeyError(f"member not found: {user}")
        if user not in issue.setdefault("watchers", []): issue["watchers"].append(user)
        return _result(True, {"issue_id": issue["issue_id"], "watchers": issue["watchers"]}, None, "", True)

    def remove_watcher(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        issue = self._iss(self._state(session_id), arguments["issue_id"]); user = arguments["user"]
        issue["watchers"] = [w for w in issue.get("watchers", []) if w != user]
        return _result(True, {"issue_id": issue["issue_id"], "watchers": issue["watchers"]}, None, "", True)

    def create_sprint(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); sid = f"spr_{state['next_issue_num']:04d}"; state["next_issue_num"] += 1
        sprint = {"sprint_id": sid, "name": arguments["name"], "start_date": arguments["start_date"], "end_date": arguments["end_date"], "goal": arguments.get("goal", ""), "status": "active", "issues": []}
        state.setdefault("sprints", {})[sid] = sprint
        return _result(True, {"sprint": sprint}, None, "", True)

    def list_sprints(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        sprints = list(self._state(session_id).get("sprints", {}).values())
        if arguments.get("status"): sprints = [s for s in sprints if s["status"] == arguments["status"]]
        return _result(True, {"sprints": sprints, "count": len(sprints)}, None, "", False)

    def add_to_sprint(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); iid, sid = arguments["issue_id"], arguments["sprint_id"]
        issue = self._iss(state, iid)
        if sid not in state.get("sprints", {}): raise KeyError(f"sprint not found: {sid}")
        issue["sprint_id"] = sid; state["sprints"][sid].setdefault("issues", []).append(iid)
        return _result(True, {"issue_id": iid, "sprint_id": sid}, None, "", True)

    def remove_from_sprint(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); issue = self._iss(state, arguments["issue_id"])
        if issue.get("sprint_id") and issue["sprint_id"] in state.get("sprints", {}):
            state["sprints"][issue["sprint_id"]]["issues"] = [x for x in state["sprints"][issue["sprint_id"]].get("issues", []) if x != issue["issue_id"]]
        issue["sprint_id"] = None
        return _result(True, {"issue_id": issue["issue_id"], "removed_from_sprint": True}, None, "", True)

    def create_subtask(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); iid = arguments["issue_id"];
        self._iss(state, iid)  # validate exists
        sid = f"sub_{state['next_issue_num']:04d}"; state["next_issue_num"] += 1
        subtask = {"subtask_id": sid, "issue_id": iid, "title": arguments["title"], "assignee": arguments.get("assignee"), "status": "open"}
        state.setdefault("subtasks", {})[sid] = subtask
        return _result(True, {"subtask": subtask}, None, "", True)

    def list_subtasks(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); iid = arguments["issue_id"]; self._iss(state, iid)
        subtasks = [st for st in state.get("subtasks", {}).values() if st["issue_id"] == iid]
        return _result(True, {"subtasks": subtasks, "count": len(subtasks)}, None, "", False)

    def time_track(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); iid = arguments["issue_id"]; self._iss(state, iid); hours = float(arguments["hours"])
        entry = {"entry_id": f"time_{state['next_issue_num']:04d}", "issue_id": iid, "hours": hours, "description": arguments.get("description", ""), "date": "2026-06-24", "user": "current_user"}
        state.setdefault("time_entries", []).append(entry); state["next_issue_num"] += 1
        return _result(True, {"time_entry": entry}, None, "", True)

    def get_time_report(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self._state(session_id); entries = state.get("time_entries", [])
        if arguments.get("issue_id"): entries = [e for e in entries if e["issue_id"] == arguments["issue_id"]]
        if arguments.get("assignee"):
            issues_by_assignee = {iid for iid, iss in state["issues"].items() if iss["assignee"] == arguments["assignee"]}
            entries = [e for e in entries if e["issue_id"] in issues_by_assignee]
        total = sum(e["hours"] for e in entries)
        return _result(True, {"entries": entries, "total_hours": total, "count": len(entries)}, None, "", False)

    def set_milestone(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        issue = self._iss(self._state(session_id), arguments["issue_id"])
        issue["milestone"] = arguments["milestone"]
        return _result(True, {"issue_id": issue["issue_id"], "milestone": issue["milestone"]}, None, "", True)

if __name__ == "__main__":
    serve(IssueTrackerServer())
