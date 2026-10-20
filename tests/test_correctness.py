"""Correctness tests: optimized + JIT-compiled output must match the spec."""

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


def test_fusion_preserves_intermediate_outputs():
    rng = np.random.default_rng(4)
    x = placeholder((3, 5), "x")
    w = const(rng.standard_normal((5, 7)).astype(np.float32))
    mm = x @ w
    biased = mm + const(np.ones(7, np.float32))
    g = Graph([mm, biased, relu(biased)], [x])
    feeds = {"x": rng.standard_normal((3, 5)).astype(np.float32)}
    expected = run_reference(g, feeds)
    optimize(g)
    for actual, ref in zip(Engine(g).run(feeds), expected):
        np.testing.assert_allclose(actual, ref, rtol=1e-5, atol=1e-5)


def test_near_zero_is_not_an_identity():
    x = placeholder((2, 2), "x")
    g = Graph([x + const(np.full((2, 2), 1e-9, np.float32))], [x])
    feeds = {"x": np.zeros((2, 2), np.float32)}
    expected = run_reference(g, feeds)
    optimize(g)
    np.testing.assert_array_equal(Engine(g).run(feeds)[0], expected[0])


def test_constant_linear_folding():
    from tensorforge.ir import Node
    a = const(np.ones((2, 3), np.float32))
    b = const(np.ones((3, 4), np.float32))
    bias = const(np.ones(4, np.float32))
    linear = Node("linear", (a.node, b.node, bias.node), (2, 4),
                  attrs={"has_bias": True, "activation": "relu"})
    g = Graph([linear], [])
    optimize(g)
    assert g.outputs[0].op == "const"
    np.testing.assert_array_equal(run_reference(g, {})[0], np.full((2, 4), 4))


def test_autotune_validates_edge_tiles():
    import importlib
    tuning = importlib.import_module("tensorforge.autotune")
    configs = [TileConfig(MC=4, NC=4, KC=3, MR=2, NR=2), TileConfig()]
    from unittest.mock import patch
    with patch.object(tuning, "_search_space", return_value=configs):
        results = tuning.autotune(5, 7, 9, activation="tanh", iters=2, warmup=1)
    assert len(results) == 2
    assert all(r.ms > 0 for r in results)
    assert results[0].ms <= results[1].ms


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nall tensorforge tests passed")
