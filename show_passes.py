"""Demonstrate the non-fusion passes on a deliberately messy graph.

Run:  python examples/show_passes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tensorforge import placeholder, const, relu, Graph, optimize, run_reference


def main():
    rng = np.random.default_rng(0)
    x = placeholder((4, 8), "x")

    # Deliberately redundant / foldable expression:
    #  - (2*C) * (3*C) is fully constant  -> constant folding
    #  - (x + 0)                          -> algebraic simplification
    #  - (x @ W) computed twice           -> CSE
    #  - an unused branch                 -> dead code elimination
    C2 = const(np.full((8, 8), 2.0, np.float32))
    C3 = const(np.full((8, 8), 3.0, np.float32))
    folded = C2 @ C3                                  # constant @ constant
    W = const(rng.standard_normal((8, 8)).astype(np.float32), "W")
    zero = const(np.zeros((4, 8), np.float32))

    branch_a = (x + zero) @ W                          # +0 is identity
    branch_b = x @ W                                   # same as branch_a after simplify
    used = relu(branch_a + branch_b)
    _unused = relu(x @ const(rng.standard_normal((8, 8)).astype(np.float32)))  # dead

    out = used @ folded
    g = Graph([out], [x])
    feeds = {"x": rng.standard_normal((4, 8)).astype(np.float32)}

    print("BEFORE  (", g.num_ops(), "ops )")
    print(g.pretty())
    ref = run_reference(g, feeds)

    optimize(g, verbose=True)

    print("\nAFTER  (", g.num_ops(), "ops )")
    print(g.pretty())
    out2 = run_reference(g, feeds)
    print("\nsemantics preserved:",
          np.allclose(ref[0], out2[0], atol=1e-4))


if __name__ == "__main__":
    main()
