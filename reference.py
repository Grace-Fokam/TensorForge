"""A straightforward numpy interpreter for the IR.

This is the *reference semantics*: the compiled backend must produce identical
results (up to floating point tolerance). Having an executable spec is how real
compilers gain confidence that an aggressive pass did not change behavior.
"""
from __future__ import annotations

import numpy as np

from .graph import Graph


def _act(name, x):
    if name == "relu":
        return np.maximum(x, 0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "tanh":
        return np.tanh(x)
    if name == "none":
        return x
    raise NotImplementedError(name)


def run_reference(g: Graph, feeds: dict) -> list[np.ndarray]:
    env: dict[int, np.ndarray] = {}
    for n in g.topo():
        if n.op == "placeholder":
            env[n.id] = np.asarray(feeds[n.name], dtype=np.float32)
        elif n.op == "const":
            env[n.id] = n.data
        elif n.op == "matmul":
            env[n.id] = env[n.inputs[0].id] @ env[n.inputs[1].id]
        elif n.op == "add":
            env[n.id] = env[n.inputs[0].id] + env[n.inputs[1].id]
        elif n.op == "mul":
            env[n.id] = env[n.inputs[0].id] * env[n.inputs[1].id]
        elif n.op == "neg":
            env[n.id] = -env[n.inputs[0].id]
        elif n.op in ("relu", "sigmoid", "tanh"):
            env[n.id] = _act(n.op, env[n.inputs[0].id])
        elif n.op == "linear":
            x = env[n.inputs[0].id]
            w = env[n.inputs[1].id]
            y = x @ w
            if n.attrs.get("has_bias"):
                y = y + env[n.inputs[2].id].reshape(1, -1)
            y = _act(n.attrs.get("activation", "none"), y)
            env[n.id] = y
        else:
            raise NotImplementedError(n.op)
    return [env[o.id] for o in g.outputs]
