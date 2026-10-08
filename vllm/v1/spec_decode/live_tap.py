# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live training-data tap for speculative decoding (EAGLE-3).

The serving engine streams the (token id, aux hidden state) pairs of every
committed position into a shared-memory ring buffer; a co-located trainer
consumes them as requests complete. One writer (the model runner, TP rank 0),
one reader.

File layout (default /dev/shm):
  header 4096 B : u64 magic, u64 capacity, u64 write_seq, u64 oldest_seq,
                  u32 hidden, u32 dropped_steps
  data capacity : records at offset seq % capacity; a record never straddles
                  the wrap (a PAD record fills the tail)
  record        : u32 kind, u32 payload_len, u64 seq, payload (padded to 16 B)
    CHUNK  : u32 id_len, u32 n, u32 start_pos, u32 prompt_len, u32 hidden,
             u32 flags, req_id (padded to 8), int32 tokens[n],
             int16 aux[n * hidden] (bf16 bit patterns); the rows cover
             positions start_pos .. start_pos + n - 1 of the request
    FINISH : u32 id_len, u32 pad, req_id   -- the request is complete
    GAP    : same as FINISH                -- a step was dropped: discard it
`seq` is a logical byte offset that only grows; `oldest_seq` is the start of
the oldest intact record, so a reader that falls behind it has been lapped and
resynchronises there (newest data wins, by design).
"""

import contextlib
import os
import queue
import struct
import threading
import time
import zlib
from collections import deque

import numpy as np

HDR = 4096
MAGIC = 0x4C49564554415031
REC = 16
K_CHUNK, K_FINISH, K_PAD, K_GAP = 1, 2, 3, 4
CHUNK_HDR = struct.Struct("<IIIIII")
ID_HDR = struct.Struct("<II")


def _r16(x):
    return (x + 15) & ~15


def _id_payload(rid):
    b = rid.encode()
    raw = ID_HDR.pack(len(b), 0) + b + bytes((-len(b)) % 8)
    return [np.frombuffer(raw, dtype=np.uint8)]


def decode_id(payload):
    n, _ = ID_HDR.unpack_from(payload, 0)
    return payload[ID_HDR.size : ID_HDR.size + n].decode()


def decode_chunk(payload):
    """-> (req_id, n, start_pos, prompt_len, hidden, tokens int32[n],
    aux int16[n, hidden]); the arrays are views into payload."""
    id_len, n, start, plen, hidden, _flags = CHUNK_HDR.unpack_from(payload, 0)
    o = CHUNK_HDR.size
    rid = payload[o : o + id_len].decode()
    o += id_len + (-id_len) % 8
    tok = np.frombuffer(payload, dtype=np.int32, count=n, offset=o)
    o += 4 * n
    aux = np.frombuffer(payload, dtype=np.int16, count=n * hidden, offset=o)
    return rid, n, start, plen, hidden, tok, aux.reshape(n, hidden)


class Ring:
    def __init__(self, path, capacity=0, create=False):
        if create:
            cap = _r16(int(capacity))
            with open(path, "wb") as f:
                f.truncate(HDR + cap)
            self.mm = np.memmap(path, dtype=np.uint8, mode="r+")
            self._u64(1, cap)
            self._u64(2, 0)
            self._u64(3, 0)
            self._u64(0, MAGIC)
        else:
            self.mm = np.memmap(path, dtype=np.uint8, mode="r")
            if self._u64(0) != MAGIC:
                raise ValueError(f"{path}: not a live-tap ring")
        self.cap = int(self._u64(1))
        # writer: (seq, length) of the records still intact, oldest first
        self.live = deque()
        # reader: start from "now"
        self.rseq = self._u64(2)
        self.lapped = 0
        self.records = 0
        self.nbytes = 0

    def _u64(self, i, v=None):
        a = self.mm[i * 8 : (i + 1) * 8].view(np.uint64)
        if v is None:
            return int(a[0])
        a[0] = v

    def _u32(self, off, v=None):
        a = self.mm[off : off + 4].view(np.uint32)
        if v is None:
            return int(a[0])
        a[0] = v

    @property
    def write_seq(self):
        return self._u64(2)

    @property
    def oldest_seq(self):
        return self._u64(3)

    # ------------------------------------------------------------- writer
    def _hdr(self, pos, kind, plen, seq):
        raw = struct.pack("<IIQ", kind, plen, seq)
        self.mm[HDR + pos : HDR + pos + REC] = np.frombuffer(raw, dtype=np.uint8)

    def _evict(self, start, length):
        lo = start + length - self.cap
        while self.live and self.live[0][0] < lo:
            self.live.popleft()
        self._u64(3, self.live[0][0] if self.live else start)

    def write(self, kind, parts):
        """parts: list of 1-D np.uint8 arrays. False if the record cannot fit."""
        plen = sum(int(p.nbytes) for p in parts)
        rec = REC + _r16(plen)
        if rec > self.cap:
            return False
        w = self._u64(2)
        pos = w % self.cap
        if self.cap - pos < rec:
            padlen = self.cap - pos
            self._evict(w, padlen)
            self._hdr(pos, K_PAD, padlen - REC, w)
            self.live.append((w, padlen))
            w += padlen
            pos = 0
            self._u64(2, w)
        self._evict(w, rec)
        self._hdr(pos, kind, plen, w)
        o = HDR + pos + REC
        for p in parts:
            n = int(p.nbytes)
            self.mm[o : o + n] = p
            o += n
        self.live.append((w, rec))
        self._u64(2, w + rec)
        self.records += 1
        self.nbytes += rec
        return True

    # ------------------------------------------------------------- reader
    def rewind_to_oldest(self):
        self.rseq = self._u64(3)

    def read(self, max_records=1000):
        """Yield (kind, payload bytes) for the records written since the
        previous call."""
        w = self._u64(2)
        o = self._u64(3)
        if self.rseq < o:
            self.lapped += 1
            self.rseq = o
        n = 0
        while self.rseq < w and n < max_records:
            pos = self.rseq % self.cap
            raw = bytes(self.mm[HDR + pos : HDR + pos + REC])
            kind, plen, seq = struct.unpack("<IIQ", raw)
            if seq != self.rseq:
                self.lapped += 1
                self.rseq = self._u64(3)
                continue
            rec = REC + _r16(plen)
            start = HDR + pos + REC
            payload = bytes(self.mm[start : start + plen]) if kind != K_PAD else b""
            if self._u64(3) > self.rseq:  # overwritten while we were copying it
                self.lapped += 1
                self.rseq = self._u64(3)
                continue
            self.rseq += rec
            n += 1
            if kind != K_PAD:
                self.records += 1
                yield kind, payload


class LiveTap:
    """Runner-side exporter.

    `record` is called once per step from the EAGLE branch with the gathered
    target token ids / hidden states; the D2H copy is issued non-blocking on
    the current stream into a pinned staging buffer, and a background thread
    writes per-request chunks once the copy has landed."""

    ENV = "VLLM_SPEC_LIVE_TAP"

    def __init__(self, path, capacity, rate, max_tokens, max_reqs, pool=3, log=print):
        self.path, self.capacity, self.rate = path, int(capacity), float(rate)
        self.max_tokens, self.max_reqs = int(max_tokens), int(max_reqs)
        self.pool = int(pool)
        self.log = log
        self.ring = None
        self.bufs = []
        self.free = queue.Queue()
        self.q = queue.Queue()
        self.dropped = 0
        self.steps = 0
        self.disabled = False
        self._tp = None
        self._t_log = time.time()
        self.thread = None

    @classmethod
    def from_env(cls, runner, log=print):
        path = os.environ.get(cls.ENV)
        if not path:
            return None
        cap = float(os.environ.get(cls.ENV + "_GB", "4")) * (1 << 30)
        rate = float(os.environ.get(cls.ENV + "_RATE", "1"))
        tap = cls(
            path,
            cap,
            rate,
            getattr(runner, "max_num_tokens", 16384),
            getattr(runner, "max_num_reqs", 1024),
            log=log,
        )
        log(
            f"[live-tap] configured: path={path} "
            f"capacity={cap / 2**30:.1f}GB rate={rate}"
        )
        return tap

    def _is_writer(self):
        if self._tp is None:
            try:
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_rank,
                )

                self._tp = get_tensor_model_parallel_rank() == 0
            except Exception:
                self._tp = True
        return self._tp

    def _ensure(self, hidden):
        import torch

        if self.bufs:
            return
        for i in range(self.pool):
            self.bufs.append(
                dict(
                    tok=torch.empty(
                        self.max_tokens, dtype=torch.int32, pin_memory=True
                    ),
                    aux=torch.empty(
                        self.max_tokens, hidden, dtype=torch.int16, pin_memory=True
                    ),
                    rej=torch.zeros(self.max_reqs, dtype=torch.int32, pin_memory=True),
                )
            )
            self.free.put(i)
        if os.path.exists(self.path):
            os.remove(self.path)
        self.ring = Ring(self.path, self.capacity, create=True)
        self.ring._u32(32, hidden)
        self.thread = threading.Thread(target=self._loop, name="live-tap", daemon=True)
        self.thread.start()
        self.log(
            f"[live-tap] ring created: hidden={hidden} "
            f"pool={self.pool}x{self.max_tokens} tokens"
        )

    def _tapped(self, rid):
        return self.rate >= 1.0 or (zlib.crc32(rid.encode()) % 10000) < (
            self.rate * 10000
        )

    def record(
        self,
        scheduler_output,
        input_batch,
        requests,
        token_ids,
        hidden_states,
        num_rejected_gpu=None,
        num_draft_tokens=None,
        compact_query_start_loc_cpu=None,
    ):
        """token_ids [T], hidden_states [T, hidden] in the batch's token layout
        (the rows of request i are contiguous, in input_batch order). With
        compact_query_start_loc_cpu (padded drafter batch disabled) the layout
        already excludes rejected tokens; otherwise each request's chunk has
        scheduler_output.num_scheduled_tokens rows of which the last
        num_rejected_gpu[i] are rejected drafts."""
        if self.disabled or not self._is_writer():
            return
        try:
            import torch

            T = int(token_ids.shape[0])
            if T == 0:
                return
            if hidden_states.element_size() != 2:
                hidden_states = hidden_states.to(torch.bfloat16)
            self._ensure(int(hidden_states.shape[-1]))
            num_reqs = input_batch.num_reqs
            req_ids = list(input_batch.req_ids[:num_reqs])
            if compact_query_start_loc_cpu is not None:
                qsl = compact_query_start_loc_cpu.numpy()[: num_reqs + 1]
                qlens = np.diff(qsl).astype(np.int64)
                nd = None
            else:
                qlens = np.array(
                    [scheduler_output.num_scheduled_tokens[r] for r in req_ids],
                    dtype=np.int64,
                )
                nd = list(num_draft_tokens) if num_draft_tokens is not None else None
            starts = np.array(
                input_batch.num_computed_tokens_cpu[:num_reqs], dtype=np.int64
            )
            plens = [
                int(getattr(requests[r], "num_prompt_tokens", 0) or 0) for r in req_ids
            ]
            try:
                i = self.free.get_nowait()
            except queue.Empty:
                self.dropped += 1
                self.q.put(("gap", req_ids))
                return
            b = self.bufs[i]
            b["tok"][:T].copy_(token_ids, non_blocking=True)
            b["aux"][:T].copy_(hidden_states.view(torch.int16), non_blocking=True)
            has_rej = (
                num_rejected_gpu is not None and compact_query_start_loc_cpu is None
            )
            if has_rej:
                b["rej"][:num_reqs].copy_(num_rejected_gpu, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self.steps += 1
            self.q.put(("step", i, ev, T, req_ids, qlens, starts, plens, nd, has_rej))
        except Exception as e:  # never let the tap take the server down
            self.disabled = True
            self.log(f"[live-tap] disabled after error: {e!r}")

    def finish(self, req_ids):
        if self.ring is not None and not self.disabled:
            self.q.put(("finish", list(req_ids)))

    def _loop(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            try:
                if it[0] == "finish":
                    for r in it[1]:
                        self.ring.write(K_FINISH, _id_payload(r))
                    continue
                if it[0] == "gap":
                    for r in it[1]:
                        self.ring.write(K_GAP, _id_payload(r))
                    continue
                _, i, ev, T, req_ids, qlens, starts, plens, nd, has_rej = it
                ev.synchronize()
                b = self.bufs[i]
                tok = b["tok"][:T].numpy()
                aux = b["aux"][:T].numpy()
                rej = b["rej"][: len(req_ids)].numpy() if has_rej else None
                hidden = aux.shape[1]
                off = 0
                for j, rid in enumerate(req_ids):
                    q = int(qlens[j])
                    v = q
                    if rej is not None and (nd is None or nd[j] > 0):
                        v = q - int(rej[j])
                    v = max(0, min(q, v))
                    if v > 0 and self._tapped(rid):
                        rb = rid.encode()
                        hdr = (
                            CHUNK_HDR.pack(
                                len(rb), v, int(starts[j]), plens[j], hidden, 0
                            )
                            + rb
                            + bytes((-len(rb)) % 8)
                        )
                        self.ring.write(
                            K_CHUNK,
                            [
                                np.frombuffer(hdr, dtype=np.uint8),
                                tok[off : off + v].view(np.uint8).reshape(-1),
                                aux[off : off + v].reshape(-1).view(np.uint8),
                            ],
                        )
                    off += q
                self.free.put(i)
                now = time.time()
                if now - self._t_log > 30:
                    self._t_log = now
                    self.log(
                        f"[live-tap] steps={self.steps} records={self.ring.records} "
                        f"MB={self.ring.nbytes / 2**20:.0f} "
                        f"dropped_steps={self.dropped} queue={self.q.qsize()}"
                    )
            except Exception as e:
                self.log(f"[live-tap] writer error: {e!r}")
                with contextlib.suppress(Exception):
                    self.free.put(it[1])
