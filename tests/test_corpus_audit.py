import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_empty_parquet_audit_exits_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(path, index=False)

    completed = subprocess.run(
        [sys.executable, "-m", "src.live_mcp.corpus.audit", str(path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "'rows': 0" in completed.stdout
