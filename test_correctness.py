"""Correctness tests: optimized + JIT-compiled output must match the spec."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tensorforge import (placeholder, const, relu, sigmoid, Graph, optimize,
                         run_reference, Engine, TileConfig)
from tensorforge.passes import (constant_folding, algebraic_simplify, cse,
                                fuse_linear)


def _mlp():
    rng = np.random.default_rng(1)
    x = placeholder((8, 16), "x")
    w1 = const(rng.standard_normal((16, 32)).astype(np.float32), "W1")
    b1 = const(rng.standard_normal((32,)).astype(np.float32), "b1")
    w2 = const(rng.standard_normal((32, 4)).astype(np.float32), "W2")
    b2 = const(rng.standard_normal((4,)).astype(np.float32), "b2")
    h = relu(x @ w1 + b1)
    out = h @ w2 + b2
    return Graph([out], [x]), {"x": rng.standard_normal((8, 16)).astype(np.float32)}


def test_jit_matches_reference():
    g, feeds = _mlp()
    ref = run_reference(g, feeds)
    optimize(g)
    out = Engine(g, TileConfig(MC=32, NC=32, KC=128)).run(feeds)
    assert np.allclose(ref[0], out[0], atol=1e-3), np.abs(ref[0] - out[0]).max()


def test_fusion_reduces_op_count():
    g, _ = _mlp()
    before = g.num_ops()
    optimize(g)
    after = g.num_ops()
    assert after < before, (before, after)
    # The two (matmul,add,relu)/(matmul,add) chains collapse into linears.
    assert any(n.op == "linear" for n in g.topo())


def test_constant_folding():
    a = const(np.full((4, 4), 2.0, np.float32))
    b = const(np.full((4, 4), 3.0, np.float32))
    g = Graph([(a + b)], [])
    rep = constant_folding(g)
    assert rep.changed >= 1
    node = g.outputs[0]
    assert node.op == "const" and np.allclose(node.data, 5.0)


def test_cse_merges_duplicates():
    x = placeholder((4, 4), "x")
    e1 = x + x
    e2 = x + x  # structurally identical
    g = Graph([e1 + e2], [])
    n0 = g.num_ops()
    cse(g)
    assert g.num_ops() < n0


def test_algebraic_identity():
    x = placeholder((4, 4), "x")
    zero = const(np.zeros((4, 4), np.float32))
    g = Graph([(x + zero)], [])
    algebraic_simplify(g)
    assert g.outputs[0].op == "placeholder"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nall tensorforge tests passed")
