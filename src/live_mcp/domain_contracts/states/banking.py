"""Banking state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


_ACCOUNT_EXISTS = lambda name="account_id": arg("account", name, "account.exists")
_ACCOUNT_ACTIVE = lambda name="account_id": arg("account", name, "account.frozen", False)
_TRANSACTION_CREATED = lambda: out(
    "transaction", "txn_id", "transaction.exists",
)


BANKING_STATE_FACTS = {
    "list_accounts": facts(),
    "get_account_info": facts(pre=(_ACCOUNT_EXISTS(),)),
    "get_balance": facts(pre=(_ACCOUNT_EXISTS(),)),
    "get_history": facts(pre=(_ACCOUNT_EXISTS(),)),
    "get_statement": facts(pre=(_ACCOUNT_EXISTS(),)),
    "transfer": facts(
        pre=(
            _ACCOUNT_EXISTS("from_account"), _ACCOUNT_EXISTS("to_account"),
            _ACCOUNT_ACTIVE("from_account"), _ACCOUNT_ACTIVE("to_account"),
            arg("account", "from_account", "account.balance_sufficient"),
        ),
        post=(_TRANSACTION_CREATED(),),
    ),
    "wire_transfer": facts(
        pre=(
            _ACCOUNT_EXISTS("from_account"), _ACCOUNT_ACTIVE("from_account"),
            arg("account", "from_account", "account.balance_sufficient"),
        ),
        post=(_TRANSACTION_CREATED(),),
    ),
    "deposit": facts(
        pre=(_ACCOUNT_EXISTS(), _ACCOUNT_ACTIVE()),
        post=(_TRANSACTION_CREATED(),),
    ),
    "withdraw": facts(
        pre=(
            _ACCOUNT_EXISTS(), _ACCOUNT_ACTIVE(),
            arg("account", "account_id", "account.balance_sufficient"),
        ),
        post=(_TRANSACTION_CREATED(),),
    ),
    "bill_pay": facts(
        pre=(
            _ACCOUNT_EXISTS(), _ACCOUNT_ACTIVE(),
            arg("account", "account_id", "account.balance_sufficient"),
        ),
        post=(_TRANSACTION_CREATED(),),
    ),
    "schedule_transfer": facts(
        pre=(
            _ACCOUNT_EXISTS("from_account"), _ACCOUNT_EXISTS("to_account"),
            _ACCOUNT_ACTIVE("from_account"), _ACCOUNT_ACTIVE("to_account"),
        ),
        post=(
            out("scheduled_transfer", "scheduled_txn_id", "scheduled_transfer.exists"),
            out("scheduled_transfer", "scheduled_txn_id", "scheduled_transfer.status", "scheduled"),
        ),
    ),
    "list_scheduled_transfers": facts(),
    "cancel_transfer": facts(
        pre=(
            arg("scheduled_transfer", "scheduled_txn_id", "scheduled_transfer.exists"),
            arg("scheduled_transfer", "scheduled_txn_id", "scheduled_transfer.cancellable"),
        ),
        post=(arg("scheduled_transfer", "scheduled_txn_id", "scheduled_transfer.status", "cancelled"),),
    ),
    "freeze_account": facts(
        pre=(_ACCOUNT_EXISTS(),),
        post=(arg("account", "account_id", "account.frozen", True),),
    ),
    "unfreeze_account": facts(
        pre=(_ACCOUNT_EXISTS(),),
        post=(arg("account", "account_id", "account.frozen", False),),
    ),
    "verify_account": facts(pre=(_ACCOUNT_EXISTS(),)),
    "get_exchange_rate": facts(),
    "apply_loan": facts(pre=(
        _ACCOUNT_EXISTS(),
        arg("account", "account_id", "account.collateral_sufficient"),
    )),
}
