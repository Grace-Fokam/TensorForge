"""End-to-end demo: build an MLP, optimize it, JIT-compile, validate, benchmark.

Run:  python examples/mlp.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tensorforge import placeholder, const, relu, Graph, optimize, run_reference, Engine
from tensorforge.autotune import autotune
from tensorforge.codegen import TileConfig


def build_mlp(batch=256, din=512, hidden=(1024, 1024), dout=256, seed=0):
    rng = np.random.default_rng(seed)
    x = placeholder((batch, din), "x")
    h = x
    d = din
    for i, hdim in enumerate(hidden):
        w = const(rng.standard_normal((d, hdim)).astype(np.float32) * 0.05, f"W{i}")
        b = const(rng.standard_normal((hdim,)).astype(np.float32) * 0.05, f"b{i}")
        h = relu(h @ w + b)
        d = hdim
    w = const(rng.standard_normal((d, dout)).astype(np.float32) * 0.05, "Wout")
    b = const(rng.standard_normal((dout,)).astype(np.float32) * 0.05, "bout")
    out = h @ w + b
    feeds = {"x": rng.standard_normal((batch, din)).astype(np.float32)}
    return Graph([out], [x]), feeds


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn()
    return (time.perf_counter() - t0) / iters * 1e3, r


def main():
    print("=" * 70)
    print("TensorForge :: end-to-end MLP compilation")
    print("=" * 70)

    g, feeds = build_mlp()
    print(f"\nBuilt MLP graph: {g.num_ops()} ops before optimization")
    print(g.pretty())

    print("\n--- optimization transcript ---")
    _, transcript = optimize(g, verbose=True)
    print(f"\nAfter optimization: {g.num_ops()} ops")
    print(g.pretty())

    # Correctness vs numpy reference
    ref = run_reference(g, feeds)

    # Autotune the dominant GEMM shape (first hidden layer).
    M, K = feeds["x"].shape
    N = 1024
    print(f"\n--- autotuning fused-linear for ({M}x{K})@({K}x{N}) ---")
    best = autotune(M, N, K, activation="relu", has_bias=True, verbose=False)
    for r in best:
        print(f"  {r.cfg.label():38s} {r.ms:8.3f} ms  {r.gflops:7.2f} GFLOP/s")
    chosen = best[0].cfg
    print(f"\nchosen config: {chosen.label()}")

    # Run JIT engine with default vs tuned config
    eng_default = Engine(g, TileConfig())
    eng_tuned = Engine(g, chosen)
    out = eng_tuned.run(feeds)
    err = np.abs(ref[0] - out[0]).max()
    print(f"\ncorrectness vs numpy reference: max abs error = {err:.2e}")
    assert err < 1e-2

    t_np, _ = bench(lambda: run_reference(g, feeds))
    t_def, _ = bench(lambda: eng_default.run(feeds))
    t_tun, _ = bench(lambda: eng_tuned.run(feeds))
    print("\n--- latency (lower is better) ---")
    print(f"  numpy reference graph : {t_np:8.3f} ms")
    print(f"  JIT (default tiles)   : {t_def:8.3f} ms")
    print(f"  JIT (autotuned tiles) : {t_tun:8.3f} ms   "
          f"({t_def / t_tun:.2f}x over default tiles)")
    print("\nNote: numpy dispatches to multithreaded BLAS; our single-threaded")
    print("scalar-C kernel is meant to demonstrate the *compiler*, not beat BLAS.")


if __name__ == "__main__":
    main()
