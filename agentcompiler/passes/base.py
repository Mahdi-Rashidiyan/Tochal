"""agentcompiler/passes/base.py — Abstract base for all compiler passes."""

from abc import ABC, abstractmethod
from agentcompiler.graph import ExecutionGraph


class CompilerPass(ABC):
    """
    A compiler pass analyses and/or transforms an ExecutionGraph.
    Passes are composable; the Compiler applies them sequentially.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Transform graph and return it (may mutate in-place)."""
        ...

    def __repr__(self) -> str:
        return f"<Pass: {self.name}>"
