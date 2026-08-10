from __future__ import annotations

import pytest

from scripts.wait_for_gpu_quiescence import (
    parse_gpu_ids,
    wait_for_gpu_quiescence,
)


def test_parse_gpu_ids_is_fail_closed() -> None:
    assert parse_gpu_ids("4,5,6,7") == (4, 5, 6, 7)
    with pytest.raises(ValueError, match="duplicates"):
        parse_gpu_ids("4,4")
    with pytest.raises(ValueError, match="non-negative integers"):
        parse_gpu_ids("4,-1")


def test_waits_for_every_selected_gpu_to_release() -> None:
    snapshots = iter(
        [
            {4: 20_000, 5: 20_000},
            {4: 4, 5: 20_000},
            {4: 4, 5: 4},
        ]
    )
    current: dict[int, int] = {}
    sleeps: list[float] = []

    def read_memory(gpu_id: int) -> int:
        nonlocal current
        if gpu_id == 4:
            current = next(snapshots)
        return current[gpu_id]

    result = wait_for_gpu_quiescence(
        (4, 5),
        memory_threshold_mib=256,
        timeout_s=10,
        poll_interval_s=0.1,
        read_memory=read_memory,
        monotonic=lambda: 0.0,
        sleep=sleeps.append,
    )

    assert result == {4: 4, 5: 4}
    assert sleeps == [0.1, 0.1]


def test_timeout_reports_each_busy_gpu() -> None:
    times = iter([0.0, 0.0, 2.0])

    with pytest.raises(
        TimeoutError,
        match=r"GPU 4=20000 MiB, GPU 5=19900 MiB",
    ):
        wait_for_gpu_quiescence(
            (4, 5),
            memory_threshold_mib=256,
            timeout_s=1,
            poll_interval_s=0.1,
            read_memory=lambda gpu_id: {4: 20_000, 5: 19_900}[gpu_id],
            monotonic=lambda: next(times),
            sleep=lambda _: None,
        )
