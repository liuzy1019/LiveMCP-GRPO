"""Calendar state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


_EVENT_EXISTS = lambda: arg("event", "event_id", "event.exists")


CALENDAR_STATE_FACTS = {
    "list_events": facts(),
    "search_events": facts(),
    "get_event": facts(pre=(_EVENT_EXISTS(),)),
    "create_event": facts(post=(
        out("event", "event_id", "event.exists"),
        out("event", "event_id", "event.recurring", False),
    )),
    "update_event": facts(pre=(_EVENT_EXISTS(),)),
    "delete_event": facts(
        pre=(_EVENT_EXISTS(),),
        post=(arg("event", "event_id", "event.exists", False),),
    ),
    "create_recurring": facts(post=(
        out("event", "event_id", "event.exists"),
        out("event", "event_id", "event.recurring", True),
    )),
    "add_attendee": facts(
        pre=(_EVENT_EXISTS(),),
        post=(arg(
            "attendee", "email", "attendee.member", True, observed=False,
        ),),
    ),
    "remove_attendee": facts(
        pre=(
            _EVENT_EXISTS(),
            arg(
                "attendee", "email", "attendee.member", True,
                observed=False,
            ),
        ),
        post=(arg(
            "attendee", "email", "attendee.member", False, observed=False,
        ),),
    ),
    "get_free_busy": facts(),
    "check_conflicts": facts(),
    "set_reminder": facts(
        pre=(_EVENT_EXISTS(),),
        post=(arg("event", "event_id", "event.reminder_set", True),),
    ),
    "get_working_hours": facts(),
    "change_timezone": facts(),
    "respond_to_event": facts(pre=(
        _EVENT_EXISTS(),
        arg(
            "attendee", "email", "attendee.member", True, observed=False,
        ),
    )),
    "export_calendar": facts(),
    "get_recurring_info": facts(pre=(
        _EVENT_EXISTS(),
        arg("event", "event_id", "event.recurring", True),
    )),
}
