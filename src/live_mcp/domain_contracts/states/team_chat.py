"""Team-chat state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


TEAM_CHAT_STATE_FACTS = {
    "list_channels": facts(),
    "create_channel": facts(post=(
        out("channel", "channel_id", "channel.exists"),
        out("channel", "channel_id", "channel.archived", False),
    )),
    "archive_channel": facts(
        pre=(arg("channel", "channel_id", "channel.exists"),),
        post=(arg("channel", "channel_id", "channel.archived", True),),
    ),
    "get_channel": facts(pre=(arg("channel", "channel_id", "channel.exists"),)),
    "send_message": facts(
        pre=(
            arg("channel", "channel_id", "channel.exists"),
            arg("channel", "channel_id", "channel.archived", False),
        ),
        post=(out("message", "message_id", "message.exists"),),
    ),
    "send_dm": facts(
        pre=(arg("user", "recipient", "user.exists"),),
        post=(out("dm", "dm_id", "dm.exists"),),
    ),
    "create_thread": facts(
        pre=(
            arg("channel", "channel_id", "channel.exists"),
            arg("message", "message_id", "message.exists"),
            arg("message", "message_id", "message.threaded", False),
        ),
        post=(
            out("thread", "thread_id", "thread.exists"),
            arg("message", "message_id", "message.threaded", True),
        ),
    ),
    "get_thread": facts(pre=(arg("thread", "thread_id", "thread.exists"),)),
    "react_message": facts(pre=(
        arg("channel", "channel_id", "channel.exists"),
        arg("message", "message_id", "message.exists"),
    )),
    "search_messages": facts(),
    "get_user_status": facts(),
}
