#!/usr/bin/env python3
"""Wait until an explicitly selected GPU set has released training memory."""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Callable, Sequence


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("GPU IDs must be comma-separated non-negative integers")
    gpu_ids = tuple(int(part) for part in parts)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must not contain duplicates")
    return gpu_ids


def read_memory_used_mib(gpu_id: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def wait_for_gpu_quiescence(
    gpu_ids: Sequence[int],
    *,
    memory_threshold_mib: int,
    timeout_s: float,
    poll_interval_s: float,
    read_memory: Callable[[int], int] = read_memory_used_mib,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[int, int]:
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    if memory_threshold_mib < 0:
        raise ValueError("memory threshold must be non-negative")
    if timeout_s < 0:
        raise ValueError("timeout must be non-negative")
    if poll_interval_s <= 0:
        raise ValueError("poll interval must be positive")

    deadline = monotonic() + timeout_s
    while True:
        observed = {gpu_id: read_memory(gpu_id) for gpu_id in gpu_ids}
        if all(
            used_mib <= memory_threshold_mib
            for used_mib in observed.values()
        ):
            return observed
        if monotonic() >= deadline:
            details = ", ".join(
                f"GPU {gpu_id}={used_mib} MiB"
                for gpu_id, used_mib in observed.items()
            )
            raise TimeoutError(
                "selected GPUs did not become quiescent before timeout: "
                + details
            )
        sleep(poll_interval_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--memory-threshold-mib", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    args = parser.parse_args()

    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        observed = wait_for_gpu_quiescence(
            gpu_ids,
            memory_threshold_mib=args.memory_threshold_mib,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
        )
    except (ValueError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f"[gpu-quiescence] ERROR: {exc}")
        return 1

    details = ", ".join(
        f"GPU {gpu_id}={used_mib} MiB"
        for gpu_id, used_mib in observed.items()
    )
    print(f"[gpu-quiescence] ready: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
