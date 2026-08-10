"""Batch scheduling and robustness-plan sampling for generation."""

from __future__ import annotations

import os
import math
import random
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from loguru import logger

from src.live_mcp.domain_allocation import (
    capacity_weighted_domain_quotas,
    jaccard_unique_sequence_count,
)
from src.live_mcp.fsm import RobustnessPlan
from src.live_mcp.types import LiveTask


def largest_remainder_mix_quotas(
    target: int,
    weights: dict[str, float],
) -> dict[str, int]:
    """Allocate an exact accepted-row target across configured mix weights."""
    if target < 0:
        raise ValueError("target must be non-negative")
    if not weights:
        raise ValueError("mix weights must not be empty")
    normalized = {str(key): float(value) for key, value in weights.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("mix weights must be non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("mix weights must sum to a positive value")
    exact = {
        key: target * value / total
        for key, value in normalized.items()
    }
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        normalized,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _difficulty_attempt_schedule(
    quotas: dict[str, int],
) -> list[str]:
    """Interleave quota-specific attempts without allowing stratum substitution.

    Each difficulty bucket receives extra attempts proportional to its quota
    (capped at 50%) to absorb per-candidate failures without letting one
    stratum silently consume another's allocation.
    """
    import math as _math
    attempts = {
        difficulty: quota + max(_math.ceil(quota * 0.5), 2)
        for difficulty, quota in quotas.items()
        if quota > 0
    }
    schedule: list[str] = []
    ordered = sorted(attempts)
    for attempt_index in range(max(attempts.values(), default=0)):
        for difficulty in ordered:
            if attempt_index < attempts[difficulty]:
                schedule.append(difficulty)
    return schedule


class BatchGenerationMixin:
    def generate_many(
        self,
        server_name: str,
        count: int,
        seed: int,
        difficulty_mix: dict[str, float] | None = None,
        irrelevance_ratio: float = 0.05,
        irrelevance_count: int | None = None,
        distractor_rate: float = 0.40,
        missing_function_rate: float = 1500 / (10895 + 1500),
        progress_callback: Callable[[list[LiveTask]], None] | None = None,
    ) -> list[LiveTask]:
        tasks: list[LiveTask] = []
        if server_name == "all":
            servers = self.manager.server_names
        elif "," in server_name:
            servers = [s.strip() for s in server_name.split(",") if s.strip()]
        else:
            servers = [server_name]
        if not servers:
            raise ValueError("no enabled Live MCP servers available")
        unknown = [s for s in servers if s not in self.manager.server_names]
        if unknown:
            raise ValueError(f"unknown servers: {unknown}")
        # Small shards may have fewer rows than domains. Rotate which domains
        # receive the remainder using the launcher client seed stride so the
        # merged candidate pool remains domain-balanced.
        if server_name == "all" and len(servers) > 1:
            stride_raw = os.environ.get("GENERATION_CLIENT_SEED_STRIDE", "1000000")
            try:
                stride = max(1, int(stride_raw))
            except ValueError:
                stride = 1000000
            rotation = (seed // stride) % len(servers)
            servers = servers[rotation:] + servers[:rotation]

        effective_mix = difficulty_mix or {"complete": 0.6, "missing": 0.2, "minimal": 0.2}

        if irrelevance_count is not None:
            if not 0 <= irrelevance_count <= count:
                raise ValueError(
                    "irrelevance_count must be between 0 and count, got "
                    f"{irrelevance_count} for count={count}"
                )
            n_irrelevant = int(irrelevance_count)
        else:
            # Direct callers retain probabilistic sampling. Multi-client
            # production launchers pass an exact globally allocated quota so
            # shard boundaries cannot inflate or erase the 5% bucket.
            n_irrelevant = sum(
                random.Random(seed + i).random() < irrelevance_ratio
                for i in range(count)
            ) if irrelevance_ratio > 0 else 0
        n_normal = count - n_irrelevant

        if len(servers) == 1:
            chain_capacities = {servers[0]: 1}
            domain_quotas = {servers[0]: n_normal}
        else:
            chain_capacities = {
                current_server: jaccard_unique_sequence_count(
                    self._get_chains(current_server),
                    threshold=self.CHAIN_SAMPLING_JACCARD_THRESHOLD,
                )
                for current_server in servers
            }
            domain_quotas = capacity_weighted_domain_quotas(
                n_normal,
                servers,
                chain_capacities,
            )
        logger.info(
            "generate_many domain allocation: target={} capacities={} quotas={}",
            n_normal,
            chain_capacities,
            domain_quotas,
        )
        global_seed_offset = 0
        failed = 0
        # ── tqdm progress bar ──
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=n_normal, desc="[generate_many]", unit="task",
                         dynamic_ncols=True, mininterval=1.0, miniters=1,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        except ImportError:
            pbar = None
        _gen_start = time.time()
        _last_log = 0  # last logged count for rate calc

        # ── Pre-compute task specs for parallel generation ──
        # Build per-domain task specifications with round-robin interleaving
        # so that every domain gets workers immediately, avoiding starvation
        # of later domains by a burst of same-domain tasks.
        # Each spec: (server_name, generation_seed, state_seed, difficulty).
        # Consecutive groups of k candidates clone the same deterministic state
        # into isolated sessions so the per-environment sampling context is
        # reusable without sharing mutations.
        per_domain_specs: dict[str, list[tuple[str, int, int, str]]] = {}
        difficulty_quotas_by_domain: dict[str, dict[str, int]] = {}
        domain_max_failures: dict[str, int] = {}
        max_specs_per_domain = 0
        for current_server in servers:
            domain_target = domain_quotas[current_server]
            difficulty_quotas = largest_remainder_mix_quotas(
                domain_target, effective_mix,
            )
            difficulty_quotas_by_domain[current_server] = difficulty_quotas
            difficulty_schedule = _difficulty_attempt_schedule(difficulty_quotas)
            domain_max_failures[current_server] = (
                len(difficulty_schedule) - domain_target
            )
            specs = []
            domain_state_seed_base = seed + global_seed_offset
            for domain_candidate_index, difficulty in enumerate(
                difficulty_schedule
            ):
                task_seed = seed + global_seed_offset
                global_seed_offset += 1
                state_seed = (
                    domain_state_seed_base
                    + domain_candidate_index // self.SAMPLING_CONTEXT_REFRESH_K
                )
                specs.append(
                    (current_server, task_seed, state_seed, difficulty)
                )
            per_domain_specs[current_server] = specs
            max_specs_per_domain = max(max_specs_per_domain, len(specs))

        # Round-robin interleave so workers pick up tasks from different domains
        task_specs: list[tuple[str, int, int, str]] = []
        for i in range(max_specs_per_domain):
            for s in servers:
                if i < len(per_domain_specs[s]):
                    task_specs.append(per_domain_specs[s][i])

        # ── Parallel generation with ThreadPoolExecutor ──
        configured_workers_raw = os.environ.get("LIVEMCP_GENERATION_MAX_WORKERS", "8")
        try:
            configured_workers = int(configured_workers_raw)
        except ValueError as exc:
            raise ValueError(
                "LIVEMCP_GENERATION_MAX_WORKERS must be an integer, got "
                f"{configured_workers_raw!r}"
            ) from exc
        if configured_workers < 1:
            raise ValueError(
                "LIVEMCP_GENERATION_MAX_WORKERS must be >= 1, got "
                f"{configured_workers}"
            )
        # Sessions are isolated and both transports support concurrent request
        # demultiplexing.  Limiting workers to the number of domains made every
        # serial per-domain production run single-threaded, starving a batched
        # four-GPU Teacher endpoint.  Bound by candidate work, not domain count.
        max_workers = min(configured_workers, max(1, len(task_specs)))
        domain_ok: dict[str, int] = {s: 0 for s in servers}
        difficulty_ok: dict[str, dict[str, int]] = {
            server: {difficulty: 0 for difficulty in quotas}
            for server, quotas in difficulty_quotas_by_domain.items()
        }
        domain_failed_count: dict[str, int] = {s: 0 for s in servers}
        submitted_futures = 0
        completed_futures = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Any, tuple[str, int, str]] = {}
            spec_index = 0

            def submit_until_full() -> None:
                nonlocal spec_index, submitted_futures
                while len(futures) < max_workers and spec_index < len(task_specs):
                    (
                        current_server,
                        task_seed,
                        state_seed,
                        difficulty,
                    ) = task_specs[spec_index]
                    spec_index += 1
                    if domain_ok[current_server] >= domain_quotas[current_server]:
                        continue
                    if (
                        difficulty_ok[current_server][difficulty]
                        >= difficulty_quotas_by_domain[current_server][difficulty]
                    ):
                        continue
                    if domain_failed_count[current_server] >= domain_max_failures[current_server]:
                        continue
                    fut = executor.submit(
                        self._generate_task_with_postprocess,
                        current_server, task_seed, state_seed, difficulty,
                        distractor_rate, missing_function_rate,
                    )
                    futures[fut] = (current_server, task_seed, difficulty)
                    submitted_futures += 1

            submit_until_full()
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for fut in done:
                    completed_futures += 1
                    current_server, task_seed, difficulty = futures.pop(fut)
                    try:
                        task = fut.result()
                    except Exception as e:
                        failed += 1
                        domain_failed_count[current_server] += 1
                        logger.warning(
                            f"generate failed for {current_server} "
                            f"(seed={task_seed}, {domain_failed_count[current_server]}x): {e}"
                        )
                        continue
                    if task is None:
                        failed += 1
                        domain_failed_count[current_server] += 1
                        continue
                    if domain_ok[current_server] >= domain_quotas[current_server]:
                        continue
                    if (
                        difficulty_ok[current_server][difficulty]
                        >= difficulty_quotas_by_domain[current_server][difficulty]
                    ):
                        continue
                    tasks.append(task)
                    domain_ok[current_server] += 1
                    difficulty_ok[current_server][difficulty] += 1
                    if progress_callback is not None:
                        # Called synchronously from the coordinator thread; the
                        # callback must treat this list as read-only.
                        progress_callback(tasks)
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"fail={failed}")
                    elapsed = time.time() - _gen_start
                    pct = len(tasks) * 100.0 / n_normal if n_normal > 0 else 0
                    if len(tasks) - _last_log >= 1:
                        _last_log = len(tasks)
                        logger.info(
                            f"[generate_many] {len(tasks)}/{n_normal} ({pct:.0f}%) "
                            f"| submitted={submitted_futures} completed={completed_futures} "
                            f"| {failed} fail | elapsed={elapsed:.0f}s "
                            f"| rate={len(tasks)/elapsed:.2f} task/s"
                        )
                submit_until_full()

        if pbar:
            pbar.close()

        # ── Warn about domains that fell short ──
        for s in servers:
            shortfall = domain_quotas[s] - domain_ok[s]
            if shortfall > 0:
                logger.warning(
                    f"{s}: fell short by {shortfall} tasks "
                    f"(got {domain_ok[s]}/{domain_quotas[s]}, "
                    f"{domain_failed_count[s]} failures)"
                )
            difficulty_shortfalls = {
                difficulty: quota - difficulty_ok[s][difficulty]
                for difficulty, quota in difficulty_quotas_by_domain[s].items()
                if difficulty_ok[s][difficulty] < quota
            }
            if difficulty_shortfalls:
                # A generation call is one candidate-pool round, not the
                # owner of final shard completeness.  Return every candidate
                # that passed the hard gates so corpus/shard.py can retain it
                # and issue a measured recovery request.  Raising here threw
                # away the whole successful prefix and made the outer
                # max_recovery_rounds contract unreachable.
                logger.warning(
                    "{}: accepted difficulty quota shortfall {}; "
                    "accepted={} target={}; returning the accepted prefix "
                    "to the shard recovery loop",
                    s,
                    difficulty_shortfalls,
                    difficulty_ok[s],
                    difficulty_quotas_by_domain[s],
                )

        # ── irrelevance tasks (5%) ──
        irr = self._generate_irrelevant_tasks(n_irrelevant, seed + 9999, servers)
        tasks.extend(irr)

        # ── Dedup is deferred to the corpus shard split stage ──
        # which applies Jaccard 0.70 after _filter_training_eligible_tasks.
        # Calling it here is redundant and wasteful O(n²).
        removed = 0
        before = len(tasks)

        # Surface low yield to the caller. With irrelevance_ratio<1, the
        # contractual target is `count` rows; falling far short usually means
        # the teacher LLM/MCP server pipeline is broken. Warn at <50% and
        # raise at 0% so callers do not silently write empty Parquet files.
        # Skip the guard entirely when the caller explicitly asked for 0
        # tasks (e.g. val-only or train-only generation).
        if count > 0 and not tasks:
            raise RuntimeError(
                f"generate_many produced 0 tasks (target {count}, "
                f"failures={failed}, dedup_removed={removed}). "
                f"Check teacher LLM connectivity and MCP servers."
            )
        if count > 0 and len(tasks) < max(1, count // 2):
            logger.error(
                f"generate_many SEVERE under-yield: got {len(tasks)}/{count} "
                f"({failed} failures, {removed} dedup_removed). "
                f"Inspect logs for repeated teacher errors."
            )

        # ── Data quality statistics (P0-3) ──
        n_paper_invalid = sum(
            1 for t in tasks
            if not t.metadata.get("paper_replay_valid", True)
        )
        n_outcome_invalid = sum(
            1 for t in tasks
            if not t.metadata.get("project_outcome_valid", True)
        )
        if n_paper_invalid or n_outcome_invalid:
            logger.warning(
                f"Data quality: {n_paper_invalid} tasks failed replay, "
                f"{n_outcome_invalid} tasks failed outcome criteria "
                f"(out of {len(tasks)} total). "
                f"Filter by paper_replay_valid before export; "
                f"outcome-invalid tasks should be isolated for analysis."
            )

        logger.info(
            f"LLM teacher: {len(tasks)} tasks (target {count}, {failed} failures, "
            f"submitted={submitted_futures}, completed={completed_futures}, "
            f"{removed} dedup removed)"
        )
        logger.info(
            "accepted difficulty quotas: target={} actual={}",
            difficulty_quotas_by_domain,
            difficulty_ok,
        )
        if hasattr(self, "_chain_sampling_stats"):
            for sampled_server in servers:
                logger.info(
                    f"dependency-chain sampling [{sampled_server}]: "
                    f"{self._chain_sampling_summary(sampled_server)}"
                )
        return tasks

    def _generate_task_with_postprocess(
        self, server_name: str, seed: int, state_seed: int, difficulty: str,
        distractor_rate: float, missing_function_rate: float,
    ) -> LiveTask | None:
        """Thread-safe single-task generation.

        Samples a robustness plan from seed, then passes it to generate_one
        so all perturbations are applied before Teacher processing and replay.

        Returns None if generate_one raises, so the caller can count failures
        and retry with a new seed.
        """
        all_tools_pool = self.manager.registry.all_tools_with_servers()
        domain_tools = self.manager.registry.server_tools(server_name)

        plan = RobustnessPlan.sample(
            seed=seed,
            all_tools_pool=all_tools_pool,
            domain_tools=domain_tools,
            distractor_rate=distractor_rate,
            strip_enums_rate=0.30,
            missing_function_rate=missing_function_rate,
        )
        return self.generate_one(
            server_name, seed=seed, difficulty=difficulty,
            robustness_plan=plan, state_seed=state_seed,
        )

    @staticmethod
    def _pick_difficulty(seed: int, difficulty_mix: dict[str, float]) -> str:
        if not difficulty_mix:
            return "complete"
        rng = random.Random(seed)
        threshold = rng.random()
        cumulative = 0.0
        for name, weight in sorted(difficulty_mix.items()):
            cumulative += weight
            if threshold <= cumulative:
                return name
        return next(iter(sorted(difficulty_mix)))



