from .schemashift_replay_loop import SchemaShiftReplayLoop

# BFCLAgentLoop 依赖 loguru 等外部包，按需导入避免 side effects
try:
    from .bfcl_agent_loop import BFCLAgentLoop
except ImportError:
    BFCLAgentLoop = None  # type: ignore[assignment,misc]

__all__ = ["BFCLAgentLoop", "SchemaShiftReplayLoop"]
