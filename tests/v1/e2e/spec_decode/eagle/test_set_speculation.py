# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime speculation switch: switching off must stop drafting without changing
outputs; switching back on must resume drafting."""

import os

from vllm import LLM, SamplingParams

from ..utils import get_test_prompts

MODEL = os.environ.get("VLLM_TEST_TARGET", "Qwen/Qwen3-8B")
DRAFT = os.environ.get("VLLM_TEST_DRAFT", "AngelSlim/Qwen3-8B_eagle3")


def _drafts(llm: LLM) -> float:
    total = 0.0
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            total += float(getattr(metric, "value", 0.0))
    return total


def test_set_speculation_is_lossless_and_stops_drafting(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    llm = LLM(
        model=MODEL,
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
        },
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        seed=0,
        disable_log_stats=False,
    )
    prompts = get_test_prompts(mm_enabled=False)[:8]
    params = SamplingParams(temperature=0, max_tokens=48)

    on = [o.outputs[0].token_ids for o in llm.chat(prompts, params)]
    drafts_on = _drafts(llm)
    assert drafts_on > 0

    llm.collective_rpc("set_speculation", kwargs={"enabled": "0"})
    off = [o.outputs[0].token_ids for o in llm.chat(prompts, params)]
    assert _drafts(llm) == drafts_on, "drafts were proposed while speculation was off"
    assert off == on, "switching speculation off changed the outputs"

    llm.collective_rpc("set_speculation", kwargs={"enabled": "1"})
    again = [o.outputs[0].token_ids for o in llm.chat(prompts, params)]
    assert _drafts(llm) > drafts_on, "drafting did not resume"
    assert again == on
