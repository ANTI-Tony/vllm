# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of the live training-data tap: every committed token of every
request must be reconstructable from the ring, in order, with hidden states of
the drafter's aux width."""

import os

import numpy as np
import pytest

from vllm import LLM, SamplingParams
from vllm.v1.spec_decode.live_tap import (
    K_CHUNK,
    K_FINISH,
    K_GAP,
    Ring,
    decode_chunk,
    decode_id,
)

from ..utils import get_test_prompts

MODEL = os.environ.get("VLLM_TEST_TARGET", "Qwen/Qwen3-8B")
DRAFT = os.environ.get("VLLM_TEST_DRAFT", "AngelSlim/Qwen3-8B_eagle3")


@pytest.mark.parametrize("padded_batch", [True, False])
def test_live_tap_reconstructs_committed_tokens(tmp_path, monkeypatch, padded_batch):
    tap = str(tmp_path / "tap")
    monkeypatch.setenv("VLLM_SPEC_LIVE_TAP", tap)
    monkeypatch.setenv("VLLM_SPEC_LIVE_TAP_GB", "0.5")
    llm = LLM(
        model=MODEL,
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
            "disable_padded_drafter_batch": not padded_batch,
        },
        max_model_len=2048,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        seed=0,
    )
    prompts = get_test_prompts(mm_enabled=False)[:12]
    outputs = llm.chat(prompts, SamplingParams(temperature=0, max_tokens=64))
    expected = {
        o.request_id: list(o.prompt_token_ids) + list(o.outputs[0].token_ids)
        for o in outputs
    }
    hidden = llm.llm_engine.model_config.get_hidden_size() * 3

    reader = Ring(tap)
    reader.rewind_to_oldest()
    seqs: dict[str, list[np.ndarray]] = {}
    finished: set[str] = set()
    for kind, payload in reader.read(max_records=1_000_000):
        if kind == K_CHUNK:
            rid, n, start, _plen, hid, tok, aux = decode_chunk(payload)
            assert hid == hidden
            assert aux.shape == (n, hidden)
            got = seqs.setdefault(rid, [])
            assert start == sum(len(c) for c in got), (rid, start)
            got.append(tok.copy())
        elif kind == K_FINISH:
            finished.add(decode_id(payload))
        else:
            assert kind != K_GAP, "a step was dropped"
    assert finished == set(expected)
    for rid, full in expected.items():
        got = np.concatenate(seqs[rid]).tolist()
        # the last sampled tokens (bonus + EOS) never get a hidden state
        assert len(got) >= len(full) - 4
        assert got == full[: len(got)], rid
