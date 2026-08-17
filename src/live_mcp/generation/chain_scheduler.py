"""Process-local dependency-chain scheduling and diagnostics."""

from __future__ import annotations

import hashlib
import json
import random

from src.live_mcp.domain_allocation import position_aware_jaccard


def chain_fingerprint(server_name: str, chain: list[str]) -> str:
    """Return the canonical stable identity for an ordered chain."""
    payload = json.dumps(
        {"server": server_name, "chain": list(chain)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChainSchedulerMixin:
    @staticmethod
    def _chain_fingerprint(server_name: str, chain: list[str]) -> str:
        """Return a stable identity for one ordered dependency-chain seed."""
        return chain_fingerprint(server_name, chain)

    def _uses_paper_baseline(self) -> bool:
        return bool(
            getattr(getattr(self, "prompt_profile", None), "paper_baseline", False)
        )

    def _chain_lock(self):
        lock = getattr(self, "_chain_sampling_lock", None)
        if lock is None:
            lock = getattr(self, "_dependency_graph_lock")
        return lock

    def _select_feasible_chain(
        self,
        server_name: str,
        feasible_chains: list[list[str]],
        rng: random.Random,
    ) -> tuple[list[str], str, int, bool]:
        """Select a least-attempted feasible chain and record the attempt.

        The seed-local RNG only breaks ties. Failed candidates remain attempts,
        so a repeatedly failing chain cannot monopolise generation. Replay,
        provenance, and deduplication remain downstream eligibility gates.
        """
        if not feasible_chains:
            raise ValueError("cannot select from an empty feasible-chain set")

        if self._uses_paper_baseline():
            # The paper's 0.70 Jaccard threshold belongs to completed-corpus
            # deduplication, not pre-generation chain scheduling.  Scheduling
            # therefore uses no similarity filter, but it does exhaust the
            # least-attempted live-feasible paths before repeating one.
            ordered = sorted(
                (list(chain) for chain in feasible_chains),
                key=lambda chain: self._chain_fingerprint(server_name, chain),
            )
            with self._chain_lock():
                domain_stats = self._chain_sampling_stats.setdefault(
                    server_name, {},
                )
                entries = [
                    (
                        chain,
                        self._chain_fingerprint(server_name, chain),
                        domain_stats.get(
                            self._chain_fingerprint(server_name, chain), {}
                        ).get("attempted", 0),
                    )
                    for chain in ordered
                ]
                minimum = min(item[2] for item in entries)
                candidates = [item for item in entries if item[2] == minimum]
                chain, fingerprint, _ = rng.choice(candidates)
                counters = domain_stats.setdefault(
                    fingerprint,
                    {"attempted": 0, "accepted": 0, "rejected_goal": 0},
                )
                self._chain_sampling_sequences.setdefault(
                    server_name, {},
                )[fingerprint] = tuple(chain)
                counters["attempted"] += 1
                attempt_number = counters["attempted"]
            return chain, fingerprint, attempt_number, False

        with self._chain_lock():
            stats_by_domain = getattr(self, "_chain_sampling_stats", None)
            if stats_by_domain is None:
                stats_by_domain = {}
                self._chain_sampling_stats = stats_by_domain
            domain_stats = stats_by_domain.setdefault(server_name, {})
            sequence_by_domain = getattr(self, "_chain_sampling_sequences", None)
            if sequence_by_domain is None:
                sequence_by_domain = {}
                self._chain_sampling_sequences = sequence_by_domain
            domain_sequences = sequence_by_domain.setdefault(server_name, {})
            entries: list[tuple[list[str], str, int]] = []
            attempted_sequences = [
                list(domain_sequences[fingerprint])
                for fingerprint, counters in domain_stats.items()
                if counters.get("attempted", 0) > 0
                and fingerprint in domain_sequences
            ]
            for chain in feasible_chains:
                fingerprint = self._chain_fingerprint(server_name, chain)
                attempts = domain_stats.get(fingerprint, {}).get("attempted", 0)
                entries.append((chain, fingerprint, attempts))

            novel_entries = [
                entry for entry in entries
                if entry[2] == 0 and all(
                    self._position_aware_chain_jaccard(entry[0], prior)
                    < self.CHAIN_SAMPLING_JACCARD_THRESHOLD
                    for prior in attempted_sequences
                )
            ]
            jaccard_novel = bool(novel_entries)
            selection_pool = novel_entries or entries
            candidates: list[tuple[list[str], str, int]] = []
            min_attempts: int | None = None
            for chain, fingerprint, attempts in selection_pool:
                if min_attempts is None or attempts < min_attempts:
                    min_attempts = attempts
                    candidates = [(chain, fingerprint, attempts)]
                elif attempts == min_attempts:
                    candidates.append((chain, fingerprint, attempts))

            # Sorting makes seeded tie-breaking independent of graph/list order.
            candidates.sort(key=lambda item: item[1])
            chain, fingerprint, previous_attempts = rng.choice(candidates)
            counters = domain_stats.setdefault(
                fingerprint, {"attempted": 0, "accepted": 0},
            )
            domain_sequences[fingerprint] = tuple(chain)
            counters["attempted"] += 1
            return (
                list(chain), fingerprint, previous_attempts + 1, jaccard_novel,
            )

    @staticmethod
    def _position_aware_chain_jaccard(a: list[str], b: list[str]) -> float:
        """Local scheduling heuristic; global PROVE merge uses plain Jaccard."""
        return position_aware_jaccard(a, b)

    def _record_chain_accepted(self, server_name: str, fingerprint: str) -> None:
        """Record that an attempted chain survived candidate-level validation."""
        with self._chain_lock():
            stats_by_domain = getattr(self, "_chain_sampling_stats", None)
            if stats_by_domain is None:
                stats_by_domain = {}
                self._chain_sampling_stats = stats_by_domain
            counters = stats_by_domain.setdefault(server_name, {}).setdefault(
                fingerprint, {"attempted": 0, "accepted": 0},
            )
            counters["accepted"] += 1

    def _record_chain_rejected(
        self, server_name: str, fingerprint: str, reason: str,
    ) -> None:
        """Record a structured chain/state failure for scheduling diagnostics."""
        if not fingerprint:
            return
        with self._chain_lock():
            counters = self._chain_sampling_stats.setdefault(
                server_name, {},
            ).setdefault(
                fingerprint,
                {"attempted": 0, "accepted": 0, "rejected_goal": 0},
            )
            if reason in {
                "goal_unsat", "goal_incoherent_mutation_set",
            }:
                counters["rejected_goal"] = counters.get("rejected_goal", 0) + 1

    def _chain_sampling_summary(self, server_name: str) -> dict[str, int]:
        """Return process-local scheduling counters for diagnostics."""
        with self._chain_lock():
            domain_stats = getattr(self, "_chain_sampling_stats", {}).get(
                server_name, {},
            )
            return {
                "attempted_total": sum(v.get("attempted", 0) for v in domain_stats.values()),
                "attempted_unique": sum(
                    1 for v in domain_stats.values() if v.get("attempted", 0) > 0
                ),
                "accepted_total": sum(v.get("accepted", 0) for v in domain_stats.values()),
                "accepted_unique": sum(
                    1 for v in domain_stats.values() if v.get("accepted", 0) > 0
                ),
                "rejected_goal_total": sum(
                    v.get("rejected_goal", 0) for v in domain_stats.values()
                ),
            }
