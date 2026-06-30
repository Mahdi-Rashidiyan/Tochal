"""
test_fix.py
============
Two checks:

1. REGRESSION: simulated pipeline_agent.py (no merge_fn) must still produce
   the same numbers as before the fix (~3.0s -> ~2.4s, 1.25x). Confirms the
   fallback path is untouched.

2. REAL-MERGE ARCHITECTURE: a mock chain with a merge_fn now actually calls
   that function once instead of substituting a fake sleep formula. We test
   it in two scenarios — one where fusing genuinely helps (proving the
   mechanism works), and one where fusing genuinely hurts (proving the
   reported number now reflects reality instead of a miscalibrated guess,
   which is exactly the bug we are fixing).
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from agentcompiler.compiler import AgentCompiler
from agentcompiler.graph import ExecutionGraph, Node, NodeType, LLMConfig
from agentcompiler.passes.merging import LLMCallMergingPass
from agentcompiler.runtime.executor import AgentExecutor

import examples.pipeline_agent as sim_pipeline


async def timed(graph):
    t0 = time.perf_counter()
    result = await AgentExecutor().run(graph)
    return time.perf_counter() - t0, result


# ── Check 1: regression ───────────────────────────────────────────────────────

async def check_regression():
    print("CHECK 1 — Regression (simulated fallback path, no merge_fn)")
    t_raw, _ = await timed(sim_pipeline.build())
    compiler = AgentCompiler(passes=[LLMCallMergingPass()])
    opt_graph = compiler.compile(sim_pipeline.build())
    t_opt, _ = await timed(opt_graph)
    speedup = t_raw / t_opt
    print(f"  Unoptimised: {t_raw:.2f}s   Optimised: {t_opt:.2f}s   Speedup: {speedup:.2f}x")
    ok = 2.0 <= t_opt <= 2.8 and 1.1 <= speedup <= 1.4
    print(f"  Expected ~3.0s -> ~2.4s, ~1.25x  ...  {'PASS' if ok else 'FAIL'}\n")
    return ok


# ── Check 2a: real merge_fn that genuinely HELPS ──────────────────────────────

CALL_COUNT = {"individual": 0, "merged": 0}

async def mock_individual_call(label, latency):
    CALL_COUNT["individual"] += 1
    await asyncio.sleep(latency)
    return f"[real-call:{label}]"

async def mock_merge_call_fast(ctx):
    """Simulates: one fused completion that's faster than 3 separate calls."""
    CALL_COUNT["merged"] += 1
    await asyncio.sleep(0.9)   # one combined call: cheaper than 3x0.7=2.1s + overhead
    return "[real-merged-output-fast]"

async def mock_merge_call_slow(ctx):
    """Simulates: fusing 3 steps into 1 prompt makes the single completion
    longer than the sum of 3 separate fast calls (mirrors the real Agent 2 result)."""
    CALL_COUNT["merged"] += 1
    await asyncio.sleep(3.5)
    return "[real-merged-output-slow]"


def build_mock_chain(merge_fn):
    cfg = LLMConfig("mock-model", 0.0, 512, 0.7)

    async def step_a(ctx): return await mock_individual_call("a", 0.7)
    async def step_b(ctx): return await mock_individual_call("b", 0.7)
    async def step_c(ctx): return await mock_individual_call("c", 0.7)

    g = ExecutionGraph()
    g.add_node(Node("step_a", NodeType.LLM_CALL, step_a, llm_config=cfg,
                     metadata={"merge_fn": merge_fn}))
    g.add_node(Node("step_b", NodeType.LLM_CALL, step_b, dependencies=["step_a"], llm_config=cfg))
    g.add_node(Node("step_c", NodeType.LLM_CALL, step_c, dependencies=["step_b"], llm_config=cfg))
    return g


async def check_real_merge(label, merge_fn, expect_speedup):
    print(f"CHECK 2 — Real merge_fn path ({label})")
    CALL_COUNT["individual"] = 0
    CALL_COUNT["merged"] = 0

    t_raw, raw_result = await timed(build_mock_chain(merge_fn))
    calls_unopt = CALL_COUNT["individual"]

    CALL_COUNT["individual"] = 0
    CALL_COUNT["merged"] = 0
    compiler = AgentCompiler(passes=[LLMCallMergingPass()])
    opt_graph = compiler.compile(build_mock_chain(merge_fn))
    t_opt, opt_result = await timed(opt_graph)
    calls_merged = CALL_COUNT["merged"]
    calls_individual_in_opt = CALL_COUNT["individual"]

    speedup = t_raw / t_opt
    print(f"  Unoptimised: {t_raw:.2f}s  ({calls_unopt} real calls)")
    print(f"  Optimised:   {t_opt:.2f}s  ({calls_merged} real merged call, {calls_individual_in_opt} leftover individual calls)")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Sample optimised output (step_c): {opt_result.get('step_c')}")

    correct_call_count = (calls_merged == 1 and calls_individual_in_opt == 0)
    direction_correct = (speedup > 1.0) == expect_speedup
    ok = correct_call_count and direction_correct
    print(f"  Real API called exactly once, no placeholder fallback used  ...  {'PASS' if correct_call_count else 'FAIL'}")
    print(f"  Speedup direction matches reality (not a hardcoded guess)   ...  {'PASS' if direction_correct else 'FAIL'}\n")
    return ok


async def main():
    print("=" * 70)
    print("  Validating merging.py fix")
    print("=" * 70 + "\n")

    r1 = await check_regression()
    r2 = await check_real_merge("fusion genuinely helps", mock_merge_call_fast, expect_speedup=True)
    r3 = await check_real_merge("fusion genuinely hurts (mirrors real Agent 2 bug)", mock_merge_call_slow, expect_speedup=False)

    print("=" * 70)
    all_pass = r1 and r2 and r3
    print(f"  ALL CHECKS {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())