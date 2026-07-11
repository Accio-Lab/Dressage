"""Proxy components for Dressage."""

from .last_step import (
    ModelMaskTemplateRegistry,
    PromptAssistantMaskBuilder,
    create_default_mask_template_registry,
)
from .proxy_client import ProxyClient
from .reasoning_parser import (
    ProxyReasoningParser,
    ReasoningParseResult,
    canonicalize_reasoning_content,
    parse_qwen3_reasoning,
)
from .server import create_app
from .session_manager import (
    Lineage,
    Route,
    Session,
    SessionManager,
    SessionPrefixTree,
    StepRecord,
    TurnRecord,
)
from .sglang_client import SGLangResponse, SGLangRouterClient
from .tool_call_parser import (
    ModelToolCallParserRegistry,
    ProxyToolCallParser,
    ToolCallParserSpec,
    create_default_tool_call_parser_registry,
)
from .trajectory_store import TrajectoryItem, TrajectorySegment, TrajectoryStore

__all__ = [
    "ModelMaskTemplateRegistry",
    "ModelToolCallParserRegistry",
    "PromptAssistantMaskBuilder",
    "ProxyToolCallParser",
    "ProxyReasoningParser",
    "ProxyClient",
    "ReasoningParseResult",
    "Lineage",
    "Route",
    "Session",
    "SessionManager",
    "SessionPrefixTree",
    "SGLangResponse",
    "SGLangRouterClient",
    "StepRecord",
    "ToolCallParserSpec",
    "TrajectoryItem",
    "TrajectorySegment",
    "TrajectoryStore",
    "TurnRecord",
    "create_default_mask_template_registry",
    "create_default_tool_call_parser_registry",
    "create_app",
    "canonicalize_reasoning_content",
    "parse_qwen3_reasoning",
]
