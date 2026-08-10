"""Audited observation-field to required-argument aliases."""

OUTPUT_ARGUMENT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "banking": {
        "account_id": ("from_account", "to_account"),
        "from_account": ("account_id",),
        "to_account": ("account_id",),
    },
    "shopping": {
        "product_id": ("product_ids",),
        "coupon": ("code",),
    },
    "filesystem": {
        "path": ("source", "target", "file1", "file2", "archive"),
        "target": ("path", "source", "file1", "file2", "archive"),
        "link_path": ("path", "source", "file1", "file2", "archive"),
    },
    "issue_tracker": {
        "user_id": ("assignee", "user"),
        "assignee": ("user",),
    },
}
