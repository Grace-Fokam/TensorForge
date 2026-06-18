"""Intermediate representation for TensorForge.

The IR is a small, typed, static-single-assignment (SSA) dataflow graph. Every
value produced in the program is a :class:`Node`. Optimization passes rewrite
this graph; the lowering stage turns it into schedulable kernels.

Design notes
------------
* Shapes are tracked symbolically-but-concretely (tuples of ints). Real
  compilers carry symbolic shapes; we keep it concrete to keep the project
  focused on the optimization/codegen story rather than shape inference.
* Ops are intentionally coarse (``matmul``, ``add``, ``relu``, ...). Fusion
  rewrites several of them into a single ``linear`` op carrying an *epilogue*,
  which is exactly how production ML compilers (XLA, TVM, Inductor) collapse a
  GEMM and its trailing pointwise ops into one kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import itertools

import numpy as np

_uid = itertools.count()


@dataclass(eq=False)
class Node:
    """A single SSA value in the dataflow graph.

    ``op`` is the operation that produced the value. ``inputs`` are the
    producing nodes. ``attrs`` holds op-specific metadata (e.g. the activation
    of a fused ``linear``). ``data`` is set only for ``const`` nodes.
    """

    op: str
    inputs: tuple["Node", ...] = ()
    shape: tuple[int, ...] = ()
    attrs: dict = field(default_factory=dict)
    data: Optional[np.ndarray] = None
    name: str = ""

    def __post_init__(self) -> None:
        self.id = next(_uid)
        if not self.name:
            self.name = f"%{self.op}{self.id}"

    # --- structural identity used by CSE -----------------------------------
    def structural_key(self):
        """A hashable key describing *what this node computes*.

        Two nodes with the same key compute the same value and can be merged
        by common-subexpression elimination. Constants key off their bytes so
        identical literals collapse together.
        """
        if self.op == "const":
            return ("const", self.data.shape, self.data.dtype.str,
                    self.data.tobytes())
        attr_items = tuple(sorted(
            (k, v) for k, v in self.attrs.items()
            if isinstance(v, (int, float, str, bool, tuple))
        ))
        return (self.op, self.shape, tuple(i.id for i in self.inputs), attr_items)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        ins = ", ".join(i.name for i in self.inputs)
        extra = f" {self.attrs}" if self.attrs else ""
        return f"{self.name}: {self.op}({ins}) -> {self.shape}{extra}"


# ---------------------------------------------------------------------------
# Op metadata: which ops are pure pointwise (fusable as epilogues), etc.
# ---------------------------------------------------------------------------
POINTWISE_UNARY = {"relu", "sigmoid", "tanh", "neg"}
POINTWISE_BINARY = {"add", "mul"}
ACTIVATIONS = {"relu", "sigmoid", "tanh"}


def is_pointwise(node: Node) -> bool:
    return node.op in POINTWISE_UNARY or node.op in POINTWISE_BINARY


def is_const(node: Node) -> bool:
    return node.op == "const"
