"""
agentcompiler/graph.py
======================
Core Intermediate Representation (IR).

An agent's execution is modelled as a Directed Acyclic Graph (DAG):
  - Nodes  → operations (LLM calls, tools, conditional branches)
  - Edges  → data-flow dependencies

Compiler passes analyse and transform this graph.
The async runtime then executes the optimised graph.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Node types
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(Enum):
    LLM_CALL  = auto()   # Call to a language model
    TOOL      = auto()   # Deterministic function / external tool
    CONDITION = auto()   # Branch: evaluates a predicate → picks true/false branch


# ─────────────────────────────────────────────────────────────────────────────
# LLM configuration (determines merge-eligibility)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    model: str           = "claude-3-haiku"
    temperature: float   = 0.0
    max_tokens: int      = 512
    sim_latency_s: float = 1.0   # Simulated wall-clock latency for benchmarks

    def mergeable_with(self, other: "LLMConfig") -> bool:
        """Two calls are merge-eligible iff they share model + temperature."""
        return self.model == other.model and self.temperature == other.temperature


# ─────────────────────────────────────────────────────────────────────────────
# Graph node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    node_type: NodeType
    fn: Callable[..., Coroutine]          # async (context: dict) -> Any
    dependencies: List[str] = field(default_factory=list)
    llm_config: Optional[LLMConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Condition-specific ────────────────────────────────────────────────────
    true_branch:  Optional[str] = None    # node_id to execute when predicate=True
    false_branch: Optional[str] = None   # node_id to execute when predicate=False
    p_true: float = 0.5                  # prior P(true) used by speculative pass


# ─────────────────────────────────────────────────────────────────────────────
# Execution graph
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionGraph:
    """
    DAG of agent operations.  Compiler passes transform this in-place or
    return a modified copy.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.metadata: Dict[str, Any] = {}

    # ── Construction ──────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> "ExecutionGraph":
        self.nodes[node.id] = node
        return self

    def add_edge(self, src: str, dst: str) -> "ExecutionGraph":
        """dst depends on src (src must finish before dst starts)."""
        if src not in self.nodes:
            raise KeyError(f"Source node '{src}' not in graph.")
        if dst not in self.nodes:
            raise KeyError(f"Destination node '{dst}' not in graph.")
        if src not in self.nodes[dst].dependencies:
            self.nodes[dst].dependencies.append(src)
        return self

    def copy(self) -> "ExecutionGraph":
        return copy.deepcopy(self)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def compute_levels(self) -> Dict[str, int]:
        """
        Assign each node its level = length of longest path from any root.
        Nodes at the same level have NO data dependency between them and can
        therefore run concurrently.
        """
        in_deg: Dict[str, int] = {
            nid: len(n.dependencies) for nid, n in self.nodes.items()
        }
        children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in children:
                    children[dep].append(nid)

        levels: Dict[str, int] = {}
        queue = [nid for nid, d in in_deg.items() if d == 0]
        lvl = 0

        while queue:
            for nid in queue:
                levels[nid] = lvl
            nxt: List[str] = []
            for nid in queue:
                for child in children[nid]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        nxt.append(child)
            queue = nxt
            lvl += 1

        return levels

    def nodes_at_level(self, level: int, levels: Dict[str, int]) -> List[str]:
        return [nid for nid, l in levels.items() if l == level]

    def get_ready(self, completed: set) -> List[str]:
        """Return node IDs whose every dependency has already completed."""
        return [
            nid
            for nid, node in self.nodes.items()
            if nid not in completed
            and all(dep in completed for dep in node.dependencies)
        ]

    def sequential_latency(self) -> float:
        """Upper bound: sum of all LLM latencies as if fully sequential."""
        return sum(
            n.llm_config.sim_latency_s
            for n in self.nodes.values()
            if n.llm_config
        )
