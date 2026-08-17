"""Deterministic state builder for banking."""

from __future__ import annotations

import datetime as _datetime
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _sample_entities,
    _seed_scoped_id,
)
_BANKING_ACCOUNT_TEMPLATES: list[tuple[str, str, float, str, str]] = [
    ("acc_savings", "Alice Johnson", 25000.00, "savings", "2018-03-15"),
    ("acc_checking", "Alice Johnson", 5000.00, "checking", "2018-03-15"),
    ("acc_business", "Bob Smith", 100000.00, "business", "2020-01-10"),
    ("acc_frozen_demo", "Carol White", 1500.00, "savings", "2022-06-01"),
    ("acc_emergency", "Alice Johnson", 8000.00, "savings", "2019-11-22"),
    ("acc_investment", "Bob Smith", 45000.00, "investment", "2020-06-15"),
    ("acc_joint", "Alice Johnson", 12000.00, "checking", "2021-01-05"),
    ("acc_travel", "Bob Smith", 3200.00, "savings", "2021-08-30"),
    ("acc_booked", "Carol White", 9800.00, "savings", "2022-04-12"),
    ("acc_trust", "Alice Johnson", 75000.00, "investment", "2017-09-01"),
    ("acc_expense", "Bob Smith", 2100.00, "checking", "2023-02-14"),
    ("acc_retirement", "Bob Smith", 180000.00, "retirement", "2015-06-30"),
    ("acc_college", "Carol White", 32000.00, "savings", "2019-08-01"),
    ("acc_hsa", "Alice Johnson", 4500.00, "health", "2021-03-10"),
    ("acc_brokerage", "Bob Smith", 62000.00, "investment", "2018-11-20"),
    ("acc_escrow", "Carol White", 15000.00, "escrow", "2023-05-15"),
    ("acc_payroll", "Alice Johnson", 7800.00, "checking", "2020-07-01"),
    ("acc_reserve", "Bob Smith", 28000.00, "savings", "2016-04-22"),
    ("acc_minor", "Carol White", 3100.00, "savings", "2024-01-10"),
    ("acc_forex", "Alice Johnson", 9200.00, "foreign", "2022-09-30"),
]

def _banking_state(seed: int) -> dict[str, Any]:
    from src.live_mcp.generation.teacher_contracts import reference_datetime_for_seed

    rng = random.Random(seed)
    selected = _sample_entities(rng, _BANKING_ACCOUNT_TEMPLATES, target_count=20, id_prefix="acc")
    accounts = {}
    frozen_ids = set()
    for idx, (aid, owner, balance, atype, opened) in enumerate(selected):
        scoped_aid = _seed_scoped_id("acc", seed, idx, width=3)
        is_frozen = aid == "acc_frozen_demo" or (aid.startswith("acc_booked") and rng.random() < 0.3)
        if is_frozen:
            frozen_ids.add(scoped_aid)
        accounts[scoped_aid] = {
            "account_id": scoped_aid, "owner": owner, "balance": round(balance + rng.randint(-200, 200), 2),
            "currency": "USD", "type": atype, "frozen": is_frozen,
            "account_last4": f"{1000 + ((seed % 8000 + idx * 379) % 9000):04d}",
            "opened_date": opened,
        }
    account_ids = list(accounts)
    current_date = reference_datetime_for_seed(seed).date()
    # Seed a rich transaction history so get_history/get_statement chains
    # produce diverse, distinguishable results across accounts.
    transaction_seeds = [
        # (from_account_idx, to_account_idx, amount, txn_type, memo)
        (0, 1, 500.0, "transfer", "Rent payment"),
        (None, 2, 2500.0, "deposit", "Salary"),
        (3, None, 120.0, "bill_pay", "Electric bill"),
        (4, None, 75.5, "withdrawal", "ATM withdrawal"),
        (0, 3, 300.0, "transfer", "Shared expenses"),
        (None, 1, 1800.0, "deposit", "Freelance payment"),
        (5, 6, 1200.0, "transfer", "Investment transfer"),
        (2, None, 350.0, "bill_pay", "Internet bill"),
        (7, None, 200.0, "withdrawal", "Cash withdrawal"),
        (None, 8, 950.0, "deposit", "Tax refund"),
        (9, 10, 400.0, "transfer", "Monthly savings"),
        (11, None, 80.0, "bill_pay", "Phone bill"),
        (12, None, 150.0, "withdrawal", "Groceries"),
        (None, 13, 2200.0, "deposit", "Bonus payment"),
        (14, 15, 85.0, "transfer", "Dinner split"),
    ]
    transactions = []
    for txn_idx, (from_idx, to_idx, amount, txn_type, memo) in enumerate(transaction_seeds):
        tid = f"txn_s{seed}_{txn_idx:04d}"
        txn_date = current_date - _datetime.timedelta(days=txn_idx * 3 + 1)
        txn = {
            "txn_id": tid,
            "amount": amount,
            "currency": "USD",
            "type": txn_type,
            "memo": memo,
            "timestamp": txn_date.isoformat(),
        }
        if from_idx is not None and from_idx < len(account_ids):
            txn["from_account"] = account_ids[from_idx]
        if to_idx is not None and to_idx < len(account_ids):
            txn["to_account"] = account_ids[to_idx]
        transactions.append(txn)
    scheduled_id = _seed_scoped_id("sched", seed, 0, width=3)
    scheduled_id_2 = _seed_scoped_id("sched", seed, 1, width=3)
    scheduled_id_3 = _seed_scoped_id("sched", seed, 2, width=3)
    year_ago = current_date - _datetime.timedelta(days=365)
    scheduled_transfers = {
        scheduled_id: {
            "scheduled_txn_id": scheduled_id,
            "from_account": account_ids[0],
            "to_account": account_ids[1],
            "amount": 25.0,
            "execute_date": (current_date + _datetime.timedelta(days=7)).isoformat(),
            "status": "scheduled",
        },
        scheduled_id_2: {
            "scheduled_txn_id": scheduled_id_2,
            "from_account": account_ids[2],
            "to_account": account_ids[3],
            "amount": 100.0,
            "execute_date": (current_date + _datetime.timedelta(days=14)).isoformat(),
            "status": "scheduled",
        },
        scheduled_id_3: {
            "scheduled_txn_id": scheduled_id_3,
            "from_account": account_ids[4],
            "to_account": account_ids[5],
            "amount": 500.0,
            "execute_date": (current_date + _datetime.timedelta(days=30)).isoformat(),
            "status": "scheduled",
        },
    }
    # Seed three loans with different statuses so apply_loan/get_loan chains
    # have diverse targets without requiring an apply_loan mutation first.
    loan_1 = _seed_scoped_id("loan", seed, 0, width=3)
    loan_2 = _seed_scoped_id("loan", seed, 1, width=3)
    loan_3 = _seed_scoped_id("loan", seed, 2, width=3)
    year_ago = current_date - _datetime.timedelta(days=365)
    six_months_ago = current_date - _datetime.timedelta(days=180)
    three_months_ago = current_date - _datetime.timedelta(days=90)
    loans = {
        loan_1: {
            "loan_id": loan_1,
            "account_id": account_ids[0],
            "amount": 10000.0,
            "term_months": 36,
            "rate": 0.045,
            "status": "active",
            "origination_date": year_ago.isoformat(),
            "monthly_payment": round(10000.0 * 0.045 / 12 + 10000.0 / 36, 2),
        },
        loan_2: {
            "loan_id": loan_2,
            "account_id": account_ids[2],
            "amount": 5000.0,
            "term_months": 12,
            "rate": 0.035,
            "status": "active",
            "origination_date": six_months_ago.isoformat(),
            "monthly_payment": round(5000.0 * 0.035 / 12 + 5000.0 / 12, 2),
        },
        loan_3: {
            "loan_id": loan_3,
            "account_id": account_ids[4],
            "amount": 25000.0,
            "term_months": 60,
            "rate": 0.055,
            "status": "paid",
            "origination_date": three_months_ago.isoformat(),
            "monthly_payment": round(25000.0 * 0.055 / 12 + 25000.0 / 60, 2),
        },
    }
    return {"accounts": accounts, "transactions": transactions, "freeze_log": [],
            "current_date": current_date.isoformat(),
            "next_txn_num": len(transactions) + 1, "scheduled_transfers": scheduled_transfers,
            "loans": loans}
