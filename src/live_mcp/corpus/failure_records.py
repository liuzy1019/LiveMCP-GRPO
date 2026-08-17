"""Durable append-only evidence for rejected generation candidates."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


class GenerationFailureWriter:
    """Append deterministic failure records without duplicating resumed attempts."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._known_ids = self._load_known_ids()

    def _load_known_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        known_ids: set[str] = set()
        for line_number, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid generation failure JSONL at "
                    f"{self.path}:{line_number}"
                ) from exc
            failure_id = record.get("failure_id")
            if not isinstance(failure_id, str) or not failure_id:
                raise ValueError(
                    f"generation failure record has no failure_id at "
                    f"{self.path}:{line_number}"
                )
            known_ids.add(failure_id)
        return known_ids

    @staticmethod
    def _failure_id(record: dict[str, Any]) -> str:
        identity_fields = (
            "schema_version",
            "candidate_kind",
            "stage",
            "reason_code",
            "domain",
            "generation_seed",
            "state_seed",
            "difficulty",
            "task_id",
            "recovery_round",
            "request_index",
            "request_domain",
            "request_seed",
        )
        identity = {
            key: record[key] for key in identity_fields if key in record
        }
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, record: dict[str, Any]) -> bool:
        """Append one record; return False when the same attempt already exists."""
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "classification": "unclassified",
            **record,
        }
        if payload["classification"] != "unclassified":
            raise ValueError(
                "generation failures must remain unclassified at capture time"
            )
        failure_id = self._failure_id(payload)
        payload["failure_id"] = failure_id
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            if failure_id in self._known_ids:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError(
                            "generation failure record write made no progress"
                        )
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._known_ids.add(failure_id)
        return True
