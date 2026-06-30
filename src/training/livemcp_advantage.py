"""向后兼容的 re-export 层。

所有分层 advantage 实现在 advantage_core.py 中。
此文件保留原有 import 路径，避免破坏测试和外部调用。
"""

from src.training.advantage_core import (
    compute_livemcp_advantages,
    compute_per_group_stratified_advantages,
    compute_standard_grpo_advantages,
    compute_stratified_advantage,
)

__all__ = [
    "compute_livemcp_advantages",
    "compute_standard_grpo_advantages",
    "compute_per_group_stratified_advantages",
    "compute_stratified_advantage",
]
