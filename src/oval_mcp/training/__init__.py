"""OVAL-MCP training layer.

Constrained GRPO (§9), Lambda update (§9), Group saturation diagnostics (§9.2-9.3),
LATA: Length-Aware Token Allocation (§9.1).
"""

from src.oval_mcp.training.lambda_state import LambdaState, DEFAULT_STATE_PATH
from src.oval_mcp.training.saturation import (
    GroupSaturation,
    SaturationDiagnostics,
    SaturationSummary,
)
from src.oval_mcp.training.lata import LATAAllocator, LATAConfig, LATAResult

__all__ = [
    "GroupSaturation",
    "SaturationDiagnostics",
    "SaturationSummary",
    "LambdaState",
    "DEFAULT_STATE_PATH",
    "LATAAllocator",
    "LATAConfig",
    "LATAResult",
]
