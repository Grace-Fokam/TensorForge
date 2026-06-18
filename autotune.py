"""A small empirical autotuner.

Given a representative problem shape (M, N, K) and an op signature, the tuner
JIT-compiles several :class:`TileConfig` candidates, benchmarks each on real
data, and returns the fastest. This is the same "compile-measure-select" loop
used by autotuning compilers like TVM/Ansor and Triton's autotuner, just with a
hand-written search space instead of a learned cost model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .codegen import TileConfig
from .runtime import CompiledKernel


@dataclass
class TuneResult:
    cfg: TileConfig
    ms: float
    gflops: float


def _search_space():
    space = []
    for MC in (32, 64, 128):
        for NC in (32, 64, 128):
            for KC in (128, 256, 512):
                for MR, NR in ((2, 4), (4, 4), (4, 8)):
                    space.append(TileConfig(MC, NC, KC, MR, NR))
    return space


def _time_kernel(kern, A, B, bias, iters, warmup):
    for _ in range(warmup):
        kern(A, B, bias)
    t0 = time.perf_counter()
    for _ in range(iters):
        kern(A, B, bias)
    return (time.perf_counter() - t0) / iters


def autotune(M, N, K, activation="relu", has_bias=True,
             iters=10, warmup=2, topk=5, verbose=False):
    """Return a sorted list of the ``topk`` fastest configs for (M,N,K)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    bias = rng.standard_normal((N,), dtype=np.float32) if has_bias else None
    flops = 2.0 * M * N * K

    results = []
    for cfg in _search_space():
        try:
            kern = CompiledKernel(cfg, activation, has_bias)
            ms = _time_kernel(kern, A, B, bias, iters, warmup) * 1e3
            results.append(TuneResult(cfg, ms, flops / (ms * 1e-3) / 1e9))
            if verbose:
                print(f"  {cfg.label():38s} {ms:8.3f} ms  "
                      f"{results[-1].gflops:7.2f} GFLOP/s")
        except Exception as e:  # pragma: no cover - compile failure guard
            if verbose:
                print(f"  {cfg.label()} FAILED: {e}")
    results.sort(key=lambda r: r.ms)
    return results[:topk]
