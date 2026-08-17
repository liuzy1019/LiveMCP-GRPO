"""CRM state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


CRM_STATE_FACTS = {
    "create_lead": facts(post=(
        out("lead", "lead_id", "lead.exists"),
        out("lead", "lead_id", "lead.status", "new"),
        out("lead", "lead_id", "lead.converted", False),
    )),
    "update_lead": facts(pre=(arg("lead", "lead_id", "lead.exists"),)),
    "convert_lead": facts(
        pre=(
            arg("lead", "lead_id", "lead.exists"),
            arg("lead", "lead_id", "lead.convertible"),
        ),
        post=(
            arg("lead", "lead_id", "lead.status", "converted"),
            arg("lead", "lead_id", "lead.converted", True),
            out("contact", "contact_id", "contact.exists"),
            out("contact", "contact_id", "contact.referenced_by_lead", True),
        ),
    ),
    "delete_lead": facts(
        pre=(
            arg("lead", "lead_id", "lead.exists"),
            arg("lead", "lead_id", "lead.deletable"),
        ),
        post=(arg("lead", "lead_id", "lead.exists", False),),
    ),
    "list_leads": facts(),
    "create_contact": facts(post=(out("contact", "contact_id", "contact.exists"),)),
    "list_contacts": facts(),
    "update_contact": facts(pre=(arg("contact", "contact_id", "contact.exists"),)),
    "delete_contact": facts(
        pre=(
            arg("contact", "contact_id", "contact.exists"),
            arg("contact", "contact_id", "contact.deletable"),
        ),
        post=(arg("contact", "contact_id", "contact.exists", False),),
    ),
    "create_deal": facts(
        any_of=((
            arg("contact", "contact_id", "contact.exists"),
            arg("lead", "lead_id", "lead.exists"),
        ),),
        post=(out("deal", "deal_id", "deal.exists"),),
    ),
    "update_deal": facts(pre=(arg("deal", "deal_id", "deal.exists"),)),
    "list_deals": facts(),
    "get_deal": facts(pre=(arg("deal", "deal_id", "deal.exists"),)),
    "create_task": facts(
        any_of=((
            arg("deal", "deal_id", "deal.exists"),
            arg("contact", "contact_id", "contact.exists"),
        ),),
        post=(
            out("task", "task_id", "task.exists"),
            out("task", "task_id", "task.status", "open"),
        ),
    ),
    "list_tasks": facts(),
    "complete_task": facts(
        pre=(arg("task", "task_id", "task.exists"),),
        post=(arg("task", "task_id", "task.status", "completed"),),
    ),
    "add_note": facts(post=(out("note", "note_id", "note.exists"),)),
}
