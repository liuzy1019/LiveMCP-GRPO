"""Replay validation and provenance check for LiveMCP task generation."""

from __future__ import annotations

import hashlib
import json as _json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from loguru import logger

from src.live_mcp.registry.tool_semantics import is_mutating_tool


# ═══════════════════════════════════════════════════════════════════════
# Replay validation
# ═══════════════════════════════════════════════════════════════════════

def replay_validate(
    oracle_calls: list[OracleCall],
    manager: "LiveMCPManager",
    executor: "LiveMCPExecutor",
    seed: int,
    domain: str,
    success_criteria: list[dict[str, Any]] | None = None,
    max_error_rate: float = 0.30,
    blocked_tools: set[str] | None = None,
    state_profiles: dict[str, str] | None = None,
    trace_recorder: Any = None,
    trace_include_state: bool = False,
) -> tuple[bool, float, int, int, bool, int]:
    """Replay oracle trace against a fresh session to verify it's reproducible.

    Counts unexpected schema-level and execution errors, not empty-result
    responses. A replayed call that was explicitly recorded as an expected
    failure is consistent when it fails with a valid schema.
    The default threshold is 30%. Training export separately validates terminal
    shape and tool-call budget.

    Returns:
        (passed, error_rate, num_errors, num_calls, criteria_ok, criteria_failed)
        - passed: True if schema/execution error_rate <= max_error_rate
        - error_rate: fraction of tool calls that failed
        - num_errors: count of schema/execution errors only
        - num_calls: total tool calls replayed
        - criteria_ok: True if replay session satisfies all success_criteria
          (or criteria list is empty). NOT merged into error_rate.
        - criteria_failed: actual count of failed success_criteria
    """
    replay_server_names = list(dict.fromkeys([
        domain,
        *(
            str(getattr(call, "server_name", "") or "")
            for call in oracle_calls
            if str(getattr(call, "server_name", "") or "")
        ),
    ]))
    session = (
        manager.create_session(seed=seed, server_names=replay_server_names)
        if state_profiles is None
        else manager.create_session(
            seed=seed,
            state_profiles=state_profiles,
            server_names=replay_server_names,
        )
    )
    num_errors = 0
    num_calls = 0
    criteria_ok = True
    criteria_failed = 0
    replay_consistent = True
    try:
        manager.discover_tools(session.session_id)
        if callable(trace_recorder):
            initial_state = manager.get_state(session.session_id)
            initial_serialized = _json.dumps(
                initial_state, sort_keys=True, ensure_ascii=True, default=str,
            )
            trace_recorder(
                "replay_start",
                session_id=session.session_id,
                replay_seed=seed,
                replay_domain=domain,
                blocked_tools=sorted(blocked_tools or set()),
                initial_state_hash=hashlib.sha256(
                    initial_serialized.encode("utf-8")
                ).hexdigest(),
                initial_state=initial_state if trace_include_state else None,
            )
        for idx, call in enumerate(oracle_calls):
            # Terminal actions are oracle contract metadata, not MCP calls.
            if call.action != "tool_call":
                continue
            from src.live_mcp.types import ToolCall
            result = executor.execute(
                session.session_id,
                ToolCall(call.tool_name, dict(call.arguments), call_id=f"replay_{idx}"),
                blocked_tools=blocked_tools,
                domain=getattr(call, "server_name", "") or domain,
            )
            num_calls += 1
            if callable(trace_recorder):
                replay_owner = getattr(call, "server_name", "") or domain
                trace_recorder(
                    "replay_call",
                    call_index=idx,
                    tool_name=call.tool_name,
                    server_name=replay_owner,
                    arguments=dict(call.arguments),
                    expected_success=getattr(call, "expected_success", None),
                    success=bool(getattr(result, "success", False)),
                    schema_valid=bool(getattr(result, "schema_valid", False)),
                    execution_status=str(
                        getattr(result, "execution_status", "FAILURE")
                    ),
                    state_changed=bool(getattr(result, "state_changed", False)),
                    observation=getattr(result, "observation", None),
                    error_type=str(getattr(result, "error_type", "") or ""),
                    error_message=str(
                        getattr(result, "error_message", "") or ""
                    ),
                )
            expected_success = getattr(call, "expected_success", None)
            if expected_success is False and result.success:
                num_errors += 1
                replay_consistent = False
                if callable(trace_recorder):
                    trace_recorder(
                        "replay_outcome_mismatch",
                        call_index=idx,
                        tool_name=call.tool_name,
                        expected_success=False,
                        actual_success=True,
                        state_changed=bool(getattr(result, "state_changed", False)),
                    )
                break
            if not result.schema_valid:
                num_errors += 1
            elif expected_success is not False and not result.success:
                num_errors += 1

        # ── Criteria check (independent of tool-error-rate) ──
        if success_criteria:
            from src.live_mcp.oracle import criterion_satisfied

            replay_state = manager.get_state(session.session_id)
            criteria_failed = sum(
                1 for criterion in success_criteria
                if not criterion_satisfied(replay_state, criterion)
            )
            criteria_ok = (criteria_failed == 0)
            if not criteria_ok:
                logger.debug(
                    f"Replay criteria check: {criteria_failed}/{len(success_criteria)} "
                    f"failed in fresh session — criteria_ok=False"
                )
        else:
            tool_call_count = sum(1 for c in oracle_calls if c.action == "tool_call")
            if tool_call_count > 0:
                has_mutating = any(
                    is_mutating_tool(
                        c.tool_name,
                        getattr(c, "server_name", "") or domain,
                    )
                    for c in oracle_calls if c.action == "tool_call"
                )
                if has_mutating:
                    logger.warning(
                        "Empty success_criteria for trace with {} tool call(s) "
                        "including mutating tool(s). Accepting — R_coverage "
                        "will use pure tool-call matching.",
                        tool_call_count,
                    )

        error_rate = num_errors / num_calls if num_calls > 0 else float(num_errors > 0)
        passed = replay_consistent and error_rate <= max_error_rate

        if callable(trace_recorder):
            replay_state = manager.get_state(session.session_id)
            replay_serialized = _json.dumps(
                replay_state, sort_keys=True, ensure_ascii=True, default=str,
            )
            trace_recorder(
                "replay_result",
                passed=passed,
                error_rate=error_rate,
                num_errors=num_errors,
                num_calls=num_calls,
                max_error_rate=max_error_rate,
                criteria_ok=criteria_ok,
                criteria_failed=criteria_failed,
                replay_consistent=replay_consistent,
                final_state_hash=hashlib.sha256(
                    replay_serialized.encode("utf-8")
                ).hexdigest(),
                final_state=replay_state if trace_include_state else None,
            )

        return passed, error_rate, num_errors, num_calls, criteria_ok, criteria_failed
    finally:
        manager.close_session(session.session_id)


# ═══════════════════════════════════════════════════════════════════════
# Sensitive-parameter provenance check.
# ═══════════════════════════════════════════════════════════════════════

# Parameter names indicative of sensitive data.
# Recognised terminal action types (must match the prompt format).
_VALID_TERMINALS: tuple[str, ...] = ("final_answer", "report_error", "ask_clarification")

_SENSITIVE_PARAM_PATTERNS: tuple[str, ...] = (
    "password", "passwd", "token", "api_key", "apikey", "secret",
    "access_key", "private_key", "credential", "auth_token",
    "session_token", "refresh_token", "otp",
)

# Parameter names that carry security-relevant values but are NOT inherently
# suspicious (e.g., account numbers used in transfers). These are checked but
# with lower severity — they should be traceable but don't fail the provenance
# check on their own unless they appear with a sensitive param.
_SECURITY_RELEVANT_PARAMS: tuple[str, ...] = (
    "account_number", "account_id", "routing_number",
)

_NUMERIC_SOURCE_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])"
)


def _canonical_number(value: Any) -> Decimal | None:
    """Return a canonical number only when the complete value is numeric."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, Decimal, str)):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _numeric_value_appears_in_sources(value: Any, sources: list[str]) -> bool:
    expected = _canonical_number(value)
    if expected is None:
        return False
    for source in sources:
        for token in _NUMERIC_SOURCE_TOKEN_RE.findall(source):
            try:
                if Decimal(token.replace(",", "")) == expected:
                    return True
            except InvalidOperation:
                continue
    return False


def provenance_check(
    oracle_calls: list[OracleCall],
    user_query: str,
    aligned_observations: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    domain: str,
    user_queries: list[str] | None = None,
    call_round_indices: list[int] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Check that sensitive parameters are traceable.

    Sensitive parameters (passwords, tokens, API keys, etc.) must appear ONLY
    when traceable to prior user turns or tool outputs. Parameters that appear
    "from nowhere" indicate the teacher LLM hallucinated them, which is a
    security risk in training data.

    aligned_observations is 1:1 aligned with oracle_calls by index:
    aligned_observations[i] is the observation produced by oracle_calls[i]
    (empty dict for terminal actions like final_answer/ask_clarification).

    Returns:
        (passed, violations)
        - passed: True if all sensitive parameters are traceable
        - violations: list of dicts describing each violation
          [{"param": str, "value": str, "tool": str, "reason": str}, ...]
    """
    violations: list[dict[str, Any]] = []
    from src.live_mcp.registry.tool_semantics import build_tool_semantics

    contracts: dict[str, Any] = {}
    for schema in tool_schemas:
        owner = str(schema.get("_server_name") or domain)
        owner_contracts = build_tool_semantics(owner, [schema])
        for tool_name, contract in owner_contracts.items():
            if tool_name in contracts:
                raise ValueError(
                    "ambiguous provenance schema owner for tool "
                    f"{tool_name!r}"
                )
            contracts[tool_name] = contract

    def _traceable(value: Any, sources: list[str]) -> bool:
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
        }
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None or item == "":
                continue
            item_text = str(item).strip()
            if not item_text:
                continue
            if any(item_text.casefold() in source.casefold() for source in sources):
                continue
            if _numeric_value_appears_in_sources(item, sources):
                continue
            if (
                item_text.upper() in currency_symbols
                and currency_symbols[item_text.upper()] in " ".join(sources)
            ):
                continue
            currency_re = re.compile(
                r"(?:^|[^\d\w\$€£¥])(\$|€|£|¥)\s*\d+(?:,\d{3})*(?:\.\d{2})?", re.I
            )
            for src in sources:
                if currency_re.search(src) and currency_re.search(item_text):
                    if (
                        currency_re.search(item_text).group(1)
                        == currency_re.search(src).group(1)
                    ):
                        return True
            return False
        return True

    # Build traceable values for each call: user query + all prior observations
    # Sensitive params appearing in early calls are traced only to user queries
    # of the same round; later calls can use any prior observation.
    round_indices = call_round_indices or [0] * len(oracle_calls)
    observation_pool: list[str] = []
    round_user_pool: dict[int, set[str]] = {}
    if user_queries:
        for r_idx, q in enumerate(user_queries):
            round_user_pool.setdefault(r_idx, set()).add(q)
    round_user_pool.setdefault(0, set()).add(user_query)

    for idx, call in enumerate(oracle_calls):
        if call.action != "tool_call":
            continue
        call_round = round_indices[idx] if idx < len(round_indices) else 0
        tool_name = call.tool_name
        contract = contracts.get(tool_name)
        if contract is None:
            continue
        sensitive_params = set(contract.sensitive_params or [])
        sensitive_params.update(
            p for p in (call.arguments or {}).keys()
            if p in _SECURITY_RELEVANT_PARAMS
        )
        if not sensitive_params:
            # Add observation to pool only after processing the call
            if idx < len(aligned_observations):
                obs = aligned_observations[idx]
                if isinstance(obs, dict):
                    observation_pool.append(_json.dumps(obs, sort_keys=True, default=str))
            continue
        sources = list(observation_pool)
        sources.extend(round_user_pool.get(call_round, []))
        sources.append(user_query)
        for param_name in sorted(sensitive_params):
            param_value = (call.arguments or {}).get(param_name)
            if param_value is None:
                continue
            traceable = _traceable(param_value, sources)
            if not traceable:
                violations.append({
                    "param": param_name,
                    "value": str(param_value),
                    "tool": tool_name,
                    "reason": f"parameter '{param_name}' with value "
                              f"'{param_value}' not traceable to prior "
                              f"user turns or tool outputs",
                })
        if idx < len(aligned_observations):
            obs = aligned_observations[idx]
            if isinstance(obs, dict):
                observation_pool.append(_json.dumps(obs, sort_keys=True, default=str))

    return len(violations) == 0, violations
