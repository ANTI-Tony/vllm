# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the speculative-decoding live training-data tap."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.live_tap import (
    CHUNK_HDR,
    K_CHUNK,
    K_FINISH,
    K_GAP,
    LiveTap,
    Ring,
    _id_payload,
    decode_chunk,
    decode_id,
)


def _chunk(rid: str, start: int, plen: int, tok: np.ndarray, aux: np.ndarray):
    rb = rid.encode()
    hdr = (
        CHUNK_HDR.pack(len(rb), len(tok), start, plen, aux.shape[1], 0)
        + rb
        + bytes((-len(rb)) % 8)
    )
    return [
        np.frombuffer(hdr, dtype=np.uint8),
        tok.view(np.uint8).reshape(-1),
        aux.reshape(-1).view(np.uint8),
    ]


def test_ring_roundtrip_wrap_and_lap(tmp_path):
    path = str(tmp_path / "ring")
    w = Ring(path, capacity=64 * 1024, create=True)
    r = Ring(path)
    hidden = 48
    got = []
    # ~600 KB of records through a 64 KB ring: wraps many times, reader keeps up
    for i in range(300):
        tok = np.arange(i, i + 20, dtype=np.int32)
        aux = np.full((20, hidden), i, dtype=np.int16)
        assert w.write(K_CHUNK, _chunk(f"req-{i}", 0, 5, tok, aux))
        assert w.write(K_FINISH, _id_payload(f"req-{i}"))
        if i % 7 == 0:
            got.extend(r.read())
    got.extend(r.read())
    assert r.lapped == 0
    chunks = [decode_chunk(p) for k, p in got if k == K_CHUNK]
    fins = [decode_id(p) for k, p in got if k == K_FINISH]
    assert len(chunks) == 300 and len(fins) == 300
    for i, (rid, n, start, plen, hid, tok, aux) in enumerate(chunks):
        assert (rid, n, start, plen, hid) == (f"req-{i}", 20, 0, 5, hidden)
        assert tok.tolist() == list(range(i, i + 20))
        assert int(aux[3, 7]) == i
        assert fins[i] == rid
    # a reader that falls behind is lapped and resynchronises at the oldest record
    slow = Ring(path)
    slow.rewind_to_oldest()
    for i in range(300, 600):
        w.write(
            K_CHUNK,
            _chunk(
                f"req-{i}",
                0,
                5,
                np.zeros(20, np.int32),
                np.zeros((20, hidden), np.int16),
            ),
        )
    out = list(slow.read(max_records=10_000))
    assert slow.lapped >= 1 and slow.rseq == w.write_seq
    assert all(k == K_CHUNK for k, _ in out)


def test_from_env_disabled_without_variable():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(LiveTap.ENV, None)
        assert (
            LiveTap.from_env(SimpleNamespace(max_num_tokens=8, max_num_reqs=2)) is None
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_record_exports_committed_positions(tmp_path):
    """Two requests in a padded batch: request A has 4 scheduled tokens of which the
    last 2 are rejected drafts, request B is a 3-token prefill chunk. Only the
    committed positions must reach the ring, in request order, with the right
    start positions; the finish marker follows."""
    path = str(tmp_path / "tap")
    tap = LiveTap(
        path, 1 << 20, 1.0, max_tokens=16, max_reqs=4, pool=2, log=lambda *_: None
    )
    tap._tp = True
    hidden = 6
    tokens = torch.arange(7, dtype=torch.int32, device="cuda")
    hs = (
        torch.arange(7 * hidden, dtype=torch.float32, device="cuda")
        .view(7, hidden)
        .to(torch.bfloat16)
    )
    rejected = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["A", "B"],
        num_computed_tokens_cpu=np.array([10, 0], dtype=np.int32),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"A": 4, "B": 3})
    requests = {
        "A": SimpleNamespace(num_prompt_tokens=5),
        "B": SimpleNamespace(num_prompt_tokens=3),
    }
    tap.record(
        scheduler_output,
        input_batch,
        requests,
        tokens,
        hs,
        num_rejected_gpu=rejected,
        num_draft_tokens=[3, 0],
    )
    tap.finish(["A"])
    tap.q.join() if hasattr(tap.q, "join") else None
    reader = Ring(path)
    reader.rewind_to_oldest()
    deadline = 50
    got = []
    while len(got) < 3 and deadline:
        got.extend(reader.read())
        deadline -= 1
        if len(got) < 3:
            import time

            time.sleep(0.05)
    kinds = [k for k, _ in got]
    assert kinds == [K_CHUNK, K_CHUNK, K_FINISH]
    rid, n, start, plen, hid, tok, aux = decode_chunk(got[0][1])
    assert (rid, n, start, plen, hid) == ("A", 2, 10, 5, hidden)
    assert tok.tolist() == [0, 1]
    assert aux.view(np.int16).shape == (2, hidden)
    expect = hs[:2].view(torch.int16).cpu().numpy()
    assert np.array_equal(aux, expect)
    rid, n, start, plen, hid, tok, aux = decode_chunk(got[1][1])
    assert (rid, n, start, plen) == ("B", 3, 0, 3)
    assert tok.tolist() == [4, 5, 6]
    assert decode_id(got[2][1]) == "A"
    assert tap.dropped == 0
    assert K_GAP not in kinds
