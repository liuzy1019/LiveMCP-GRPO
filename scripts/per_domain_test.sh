#!/usr/bin/env bash
# Compatibility wrapper for the YAML-driven public corpus test runner.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data2/liuzhanyi/envs/arl/bin/python}"
exec "${PYTHON_BIN}" scripts/per_domain_test_runner.py "$@"
