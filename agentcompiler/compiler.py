"""
agentcompiler/compiler.py
==========================
AgentCompiler — applies passes then executes the optimised graph.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from agentcompiler.graph import ExecutionGraph
from agentcompiler.passes.base import CompilerPass
from agentcompiler.passes.merging import LLMCallMergingPass
from agentcompiler.passes.parallelism import ParallelismExtractionPass
from agentcompiler.passes.speculative import SpeculativeBranchPass
from agentcompiler.runtime.executor import AgentExecutor


class AgentCompiler:
    """
    Two-phase compiler for agentic workloads.

    Phase 1 – Compilation
        Applies a sequence of optimisation passes to the ExecutionGraph.

    Phase 2 – Execution
        Runs the transformed graph via the async AgentExecutor.

    Usage
    -----
        compiler = AgentCompiler()
        optimised = compiler.compile(graph)
        result    = compiler.run(optimised, input_data={"query": "..."})

        # or in one step:
        result = compiler.compile_and_run(graph, input_data={...})
    """

    DEFAULT_PASSES: List[CompilerPass] = [
        ParallelismExtractionPass(),
        LLMCallMergingPass(),
        SpeculativeBranchPass(),
    ]

    def __init__(self, passes: Optional[List[CompilerPass]] = None) -> None:
        self.passes   = passes if passes is not None else list(self.DEFAULT_PASSES)
        self.executor = AgentExecutor()

    # ── Compilation ──────────────────────────────────────────────────────────

    def compile(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Apply all passes to a deep copy of the graph."""
        optimised = graph.copy()
        for p in self.passes:
            optimised = p.apply(optimised)
        return optimised

    # ── Execution ────────────────────────────────────────────────────────────

    async def run_async(
        self,
        graph: ExecutionGraph,
        input_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return await self.executor.run(graph, input_data or {})

    def run(
        self,
        graph: ExecutionGraph,
        input_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return asyncio.run(self.run_async(graph, input_data or {}))

    # ── Combined compile + run ────────────────────────────────────────────────

    async def compile_and_run_async(
        self,
        graph: ExecutionGraph,
        input_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        optimised = self.compile(graph)
        return await self.run_async(optimised, input_data or {})

    def compile_and_run(
        self,
        graph: ExecutionGraph,
        input_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return asyncio.run(self.compile_and_run_async(graph, input_data or {}))
