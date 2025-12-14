"""
Agent Laboratory Core v2

Next-generation core components for Agent Laboratory:
- Event streaming for real-time progress
- Experiment branching and comparison
- Research knowledge graph
- Agent swarm for parallel execution
- Intelligent code sandbox

Usage:
    from core_v2 import (
        EventStream,
        ExperimentTree,
        KnowledgeGraph,
        AgentSwarm,
        CodeSandbox,
        ResearchSession
    )
"""

from .events import EventStream, Event, EventType
from .experiment_tree import ExperimentTree, ExperimentBranch, BranchConfig
from .knowledge import KnowledgeGraph, Paper, CodePattern
from .swarm import AgentSwarm, SwarmTask, SwarmResult
from .sandbox import CodeSandbox, AnalysisResult, ExecutionResult
from .session import ResearchSession, SessionConfig, SessionState

__all__ = [
    # Events
    "EventStream",
    "Event",
    "EventType",

    # Experiments
    "ExperimentTree",
    "ExperimentBranch",
    "BranchConfig",

    # Knowledge
    "KnowledgeGraph",
    "Paper",
    "CodePattern",

    # Swarm
    "AgentSwarm",
    "SwarmTask",
    "SwarmResult",

    # Sandbox
    "CodeSandbox",
    "AnalysisResult",
    "ExecutionResult",

    # Session
    "ResearchSession",
    "SessionConfig",
    "SessionState",
]
