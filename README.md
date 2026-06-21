# Tochal(ML agent compiler)

A compiler for agentic AI workloads that transforms sequential agent execution graphs into optimised parallel schedules — the way LLVM optimises programs, but for LLM agent pipelines.

## Motivation

LangGraph, CrewAI, and similar frameworks build agents as DAGs. But most implementations execute nodes naively: one at a time, even when nodes are fully independent. For LLM-heavy workloads where each call costs 0.5–5 seconds, this is a critical bottleneck.

The ML Agent Compiler analyses the execution graph and applies three passes before runtime:

| Pass | Technique | Mechanism |
|------|-----------|-----------|
| 1 | **Parallelism Extraction** | Finds independent nodes at the same DAG level; runs them with `asyncio.gather()` |
| 2 | **LLM Call Merging** | Detects sequential chains with the same model config; replaces N API calls with one multi-part prompt |
| 3 | **Speculative Branch Execution** | For condition nodes with high prior P(branch), pre-starts the predicted branch concurrently with evaluation |

## Benchmark Results

LLM latency is **simulated** via `asyncio.sleep()`. Concurrency is **real** — `asyncio.gather()` genuinely overlaps coroutines. Speedup ratios reflect true parallel scheduling.

```
Agent 1 — Research   (3 independent queries)   Pass 1  3.80s → 1.80s   2.11×
Agent 2 — QA Pipeline (3 sequential same-model) Pass 2  3.00s → 2.40s   1.25×
Agent 3 — Branch      (speculative, P=0.80)     Pass 3  2.00s → 1.50s   1.33×
Combined              (all three passes)         All     6.91s → 4.11s   1.68×
```

### Agent 1 — Parallelism Extraction (2.11×)

Three independent research queries that a naive framework runs sequentially:

```
Without compiler:  query_climate (1.0s) → query_energy (1.0s) → query_policy (1.0s) → synthesize (0.8s)
                   Total: 3.8s

With Pass 1:      [query_climate ∥ query_energy ∥ query_policy] (1.0s) → synthesize (0.8s)
                   Total: 1.8s   →   2.11× speedup
```

### Agent 2 — LLM Call Merging (1.25×)

A draft → refine → format pipeline. All three steps share the same model/temperature and form a mergeable chain:

```
Without compiler:  draft (1.0s) → refine (1.0s) → format (1.0s)
                   3 API round-trips = 3.0s

With Pass 2:       merged_chain (0.3s overhead + 3 × 0.7s gen = 2.4s)
                   1 API round-trip = 2.4s   →   1.25× speedup
```

### Agent 3 — Speculative Branch Execution (1.33×)

A sentiment classifier routes to one of two response drafters. 80% of traffic goes to the positive branch:

```
Without compiler:  classify (0.5s) → draft_positive (1.2s) → compose (0.3s) = 2.0s

With Pass 3:       [classify ∥ draft_positive] = max(0.5, 1.2) = 1.2s → compose (0.3s)
                   Total: 1.5s   →   1.33× speedup
                   (mis-speculation rate: 20%)
```

## Architecture

```
ExecutionGraph  ──→  [Pass 1: Parallelism]
                ──→  [Pass 2: Merging]
                ──→  [Pass 3: Speculative]
                ──→  AgentExecutor (async runtime)
```

The compiler never modifies the original graph — it operates on a deep copy, so the original agent definition is preserved.

```python
from agentcompiler import AgentCompiler, ExecutionGraph, Node, NodeType, LLMConfig
import asyncio

# Define your agent graph
graph = ExecutionGraph()
cfg   = LLMConfig(model="claude-3-haiku", temperature=0.0, sim_latency_s=1.0)

async def fetch_data(ctx): ...
async def analyse(ctx): ...
async def summarise(ctx): ...

graph.add_node(Node("fetch",    NodeType.LLM_CALL, fetch_data,  llm_config=cfg))
graph.add_node(Node("analyse",  NodeType.LLM_CALL, analyse,     llm_config=cfg))
graph.add_node(Node("summary",  NodeType.LLM_CALL, summarise,   llm_config=cfg))
graph.add_edge("fetch",   "summary")
graph.add_edge("analyse", "summary")

# Compile and run
compiler = AgentCompiler()
result   = compiler.compile_and_run(graph, input_data={"query": "..."})
```

## Installation

```bash
pip install -e .
```

## Running the Benchmarks

```bash
python -m benchmarks.benchmark
```

## Project Structure

```
agentcompiler/
├── agentcompiler/
│   ├── graph.py              # IR: ExecutionGraph, Node, LLMConfig
│   ├── compiler.py           # AgentCompiler: applies passes + runs
│   ├── passes/
│   │   ├── parallelism.py    # Pass 1: parallelism extraction
│   │   ├── merging.py        # Pass 2: LLM call merging
│   │   └── speculative.py    # Pass 3: speculative branch execution
│   └── runtime/
│       └── executor.py       # Async execution engine
├── examples/
│   ├── research_agent.py
│   ├── pipeline_agent.py
│   └── branching_agent.py
└── benchmarks/
    └── benchmark.py
```

## Roadmap

- [ ] Pass 3 mis-speculation cost model (adaptive threshold)
- [ ] Real LLM backend integration (Anthropic, OpenAI)
- [ ] LangGraph graph import adapter
- [ ] Distributed execution backend (multi-process, multi-machine)
- [ ] Dynamic graph recompilation based on runtime telemetry
- [ ] CUDA-style persistent kernel for agent hot paths

## Connection to Research

This project operationalises insights from compiler theory applied to agentic AI workloads:
gradient norm imbalance in LLM training creates uneven computation graphs that benefit disproportionately from parallelism extraction at inference time. See the companion NeurIPS paper for the theoretical grounding.

## License

MIT
