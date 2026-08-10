"""Email state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


_EMAIL_EXISTS = lambda: arg("email", "email_id", "email.exists")


EMAIL_STATE_FACTS = {
    "list_inbox": facts(),
    "search_emails": facts(),
    "get_email": facts(pre=(_EMAIL_EXISTS(),)),
    "send_email": facts(),
    "create_draft": facts(post=(
        out("draft", "email_id", "draft.exists"),
        out("draft", "email_id", "draft.status", "draft"),
    )),
    "forward_email": facts(pre=(_EMAIL_EXISTS(),)),
    "reply_email": facts(pre=(_EMAIL_EXISTS(),)),
    "add_label": facts(pre=(_EMAIL_EXISTS(),)),
    "remove_label": facts(pre=(_EMAIL_EXISTS(),)),
    "move_to_thread": facts(pre=(
        _EMAIL_EXISTS(),
        arg("thread", "thread_id", "thread.exists"),
    )),
    "get_thread": facts(pre=(arg("thread", "thread_id", "thread.exists"),)),
    "archive_email": facts(
        pre=(_EMAIL_EXISTS(),),
        post=(arg("email", "email_id", "email.archived", True),),
    ),
    "mark_read": facts(
        pre=(_EMAIL_EXISTS(),),
        post=(arg("email", "email_id", "email.read", True),),
    ),
    "mark_unread": facts(
        pre=(_EMAIL_EXISTS(),),
        post=(arg("email", "email_id", "email.read", False),),
    ),
    "create_filter": facts(post=(out("filter", "filter_id", "filter.exists"),)),
    "list_filters": facts(),
    "get_attachments": facts(pre=(_EMAIL_EXISTS(),)),
}
