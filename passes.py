"""Graph-level optimization passes.

Each pass takes a :class:`~tensorforge.graph.Graph`, mutates it in place, and
returns a small :class:`PassReport` describing what it did. The :func:`optimize`
driver runs them to a fixed point, because passes expose opportunities for one
another (e.g. constant folding creates dead nodes that DCE then removes, and
fusion is only legal once CSE has de-duplicated shared sub-expressions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import Graph
from .ir import Node, ACTIVATIONS, is_const


@dataclass
class PassReport:
    name: str
    changed: int = 0

    def __bool__(self):
        return self.changed > 0


# ---------------------------------------------------------------------------
# 1. Constant folding: evaluate ops whose inputs are all constants.
# ---------------------------------------------------------------------------
def _eval_const(node: Node) -> np.ndarray:
    xs = [i.data for i in node.inputs]
    if node.op == "add":
        return xs[0] + xs[1]
    if node.op == "mul":
        return xs[0] * xs[1]
    if node.op == "neg":
        return -xs[0]
    if node.op == "matmul":
        return xs[0] @ xs[1]
    if node.op == "relu":
        return np.maximum(xs[0], 0)
    if node.op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-xs[0]))
    if node.op == "tanh":
        return np.tanh(xs[0])
    raise NotImplementedError(node.op)


def constant_folding(g: Graph) -> PassReport:
    rep = PassReport("constant_folding")
    for n in g.topo():
        if n.op in ("const", "placeholder"):
            continue
        if all(is_const(i) for i in n.inputs):
            folded = Node("const", (), n.shape,
                          data=_eval_const(n).astype(np.float32))
            g.replace_all_uses(n, folded)
            rep.changed += 1
    return rep


# ---------------------------------------------------------------------------
# 2. Algebraic simplification: x+0, x*1, x*0, --x, etc.
# ---------------------------------------------------------------------------
def _is_all(node: Node, value: float) -> bool:
    return is_const(node) and np.allclose(node.data, value)


def algebraic_simplify(g: Graph) -> PassReport:
    rep = PassReport("algebraic_simplify")
    for n in g.topo():
        if n.op == "add":
            a, b = n.inputs
            if _is_all(b, 0.0) and a.shape == n.shape:
                g.replace_all_uses(n, a); rep.changed += 1
            elif _is_all(a, 0.0) and b.shape == n.shape:
                g.replace_all_uses(n, b); rep.changed += 1
        elif n.op == "mul":
            a, b = n.inputs
            if _is_all(b, 1.0) and a.shape == n.shape:
                g.replace_all_uses(n, a); rep.changed += 1
            elif _is_all(a, 1.0) and b.shape == n.shape:
                g.replace_all_uses(n, b); rep.changed += 1
        elif n.op == "neg" and n.inputs[0].op == "neg":
            g.replace_all_uses(n, n.inputs[0].inputs[0]); rep.changed += 1
    return rep


# ---------------------------------------------------------------------------
# 3. Common subexpression elimination: merge structurally identical nodes.
# ---------------------------------------------------------------------------
def cse(g: Graph) -> PassReport:
    rep = PassReport("cse")
    table: dict = {}
    for n in g.topo():
        key = n.structural_key()
        if key in table:
            g.replace_all_uses(n, table[key]); rep.changed += 1
        else:
            table[key] = n
    return rep


# ---------------------------------------------------------------------------
# 4. Operator fusion: matmul -> (+ bias) -> activation  ==>  fused `linear`.
#    This is the headline optimization: it turns 3 memory-bound kernels into a
#    single compute-bound GEMM with a fused epilogue, eliminating two full
#    round-trips of the activation matrix through memory.
# ---------------------------------------------------------------------------
def fuse_linear(g: Graph) -> PassReport:
    rep = PassReport("fuse_linear")
    users = g.users()

    def single_user(n):
        u = users.get(n.id, [])
        return u[0] if len(u) == 1 else None

    for n in g.topo():
        if n.op != "matmul":
            continue
        epilogue = []
        bias = None
        cur = n
        # Walk the single-use chain: matmul -> add(bias)? -> activation?
        nxt = single_user(cur)
        if nxt is not None and nxt.op == "add":
            a, b = nxt.inputs
            other = b if a is cur else a
            # bias must broadcast over rows: shape (N,) or (1,N)
            if (other.shape == (cur.shape[1],) or
                    other.shape == (1, cur.shape[1])):
                bias = other
                cur = nxt
                nxt = single_user(cur)
        if nxt is not None and nxt.op in ACTIVATIONS:
            epilogue.append(nxt.op)
            cur = nxt

        if bias is None and not epilogue:
            continue  # nothing to fuse

        ins = [n.inputs[0], n.inputs[1]]
        if bias is not None:
            ins.append(bias)
        fused = Node(
            "linear", tuple(ins), cur.shape,
            attrs={"has_bias": bias is not None,
                   "activation": epilogue[0] if epilogue else "none"},
        )
        g.replace_all_uses(cur, fused)
        rep.changed += 1
    return rep


# ---------------------------------------------------------------------------
# 5. Dead code elimination: drop nodes not reachable from outputs.
#    (replace_all_uses already prunes most; topo() is the source of truth.)
# ---------------------------------------------------------------------------
def dce(g: Graph) -> PassReport:
    rep = PassReport("dce")
    before = len(list(g.topo()))
    # Rebuilding topo from outputs is itself a DCE: unreachable nodes vanish.
    reachable = {n.id for n in g.topo()}
    rep.changed = 0  # informational; real pruning is implicit
    after = len(reachable)
    rep.changed = max(0, before - after)
    return rep


# ---------------------------------------------------------------------------
# Driver: run passes to a fixed point and return the transcript.
# ---------------------------------------------------------------------------
PIPELINE = [constant_folding, algebraic_simplify, cse, fuse_linear, dce]


def optimize(g: Graph, max_iters: int = 8, verbose: bool = False):
    transcript = []
    for it in range(max_iters):
        any_change = False
        for p in PIPELINE:
            rep = p(g)
            if rep.changed:
                any_change = True
            transcript.append((it, rep.name, rep.changed))
            if verbose and rep.changed:
                print(f"  [iter {it}] {rep.name}: {rep.changed} rewrites")
        if not any_change:
            break
    return g, transcript
