"""User-facing eDSL frontend.

A :class:`Tensor` is a thin, operator-overloaded handle around an IR
:class:`~tensorforge.ir.Node`. Writing ``relu(x @ w + b)`` builds a graph; it
does not compute anything. This mirrors the define-by-run-then-trace model used
by modern compilers: the Python expression *is* the program.
"""
from __future__ import annotations

import numpy as np

from .ir import Node


class Tensor:
    def __init__(self, node: Node):
        self.node = node

    # --- shape sugar -------------------------------------------------------
    @property
    def shape(self):
        return self.node.shape

    # --- builders ----------------------------------------------------------
    def __matmul__(self, other: "Tensor") -> "Tensor":
        a, b = self.node, other.node
        assert len(a.shape) == 2 and len(b.shape) == 2, "matmul is 2D only"
        assert a.shape[1] == b.shape[0], f"shape mismatch {a.shape} @ {b.shape}"
        out = Node("matmul", (a, b), (a.shape[0], b.shape[1]))
        return Tensor(out)

    def _binop(self, other, op):
        b = other.node if isinstance(other, Tensor) else _const(other).node
        a = self.node
        shape = _broadcast(a.shape, b.shape)
        return Tensor(Node(op, (a, b), shape))

    def __add__(self, other):
        return self._binop(other, "add")

    def __mul__(self, other):
        return self._binop(other, "mul")

    def __neg__(self):
        return Tensor(Node("neg", (self.node,), self.node.shape))


# --- broadcast rule (numpy-style, restricted to bias rows) -----------------
def _broadcast(sa, sb):
    if sa == sb:
        return sa
    # support (M,N) op (N,) bias and (M,N) op (1,N)
    if len(sa) == 2 and len(sb) == 1 and sb[0] == sa[1]:
        return sa
    if len(sa) == 2 and len(sb) == 2 and sb[0] == 1 and sb[1] == sa[1]:
        return sa
    if len(sb) == 2 and len(sa) == 1 and sa[0] == sb[1]:
        return sb
    raise ValueError(f"cannot broadcast {sa} and {sb}")


# --- frontend constructors -------------------------------------------------
def placeholder(shape, name="x") -> Tensor:
    return Tensor(Node("placeholder", (), tuple(shape), name=name))


def _const(value) -> Tensor:
    arr = np.asarray(value, dtype=np.float32)
    return Tensor(Node("const", (), arr.shape, data=arr))


def const(value, name="") -> Tensor:
    t = _const(value)
    if name:
        t.node.name = name
    return t


def relu(x: Tensor) -> Tensor:
    return Tensor(Node("relu", (x.node,), x.shape))


def sigmoid(x: Tensor) -> Tensor:
    return Tensor(Node("sigmoid", (x.node,), x.shape))


def tanh(x: Tensor) -> Tensor:
    return Tensor(Node("tanh", (x.node,), x.shape))
