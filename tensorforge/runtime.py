"""Execution backend: compile generated C with gcc, load via ctypes, run.

The runtime walks the *optimized* graph. Fused ``linear`` nodes dispatch to a
JIT-compiled native kernel; the handful of remaining primitive ops fall back to
numpy. This split mirrors how a real compiler keeps a small library runtime for
the long tail of ops while generating fast code for the hot path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

from .codegen import TileConfig, generate_kernel
from .graph import Graph
from .reference import _act

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "tensorforge_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

CFLAGS = ["-O3", "-march=native", "-funroll-loops", "-ffast-math", "-shared",
          "-fPIC"]


class CompiledKernel:
    """One JIT-compiled fused-linear kernel, callable on numpy arrays."""

    def __init__(self, cfg: TileConfig, activation: str, has_bias: bool):
        self.cfg = cfg
        self.activation = activation
        self.has_bias = has_bias
        src = generate_kernel(cfg, activation, has_bias)
        self._lib = _compile(src)
        self._fn = self._lib.linear_kernel
        argtypes = [ctypes.POINTER(ctypes.c_float)] * (4 if has_bias else 3)
        argtypes += [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._fn.argtypes = argtypes
        self._fn.restype = None

    def __call__(self, A, B, bias=None):
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = np.zeros((M, N), dtype=np.float32)
        fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if self.has_bias:
            bias = np.ascontiguousarray(bias, dtype=np.float32).ravel()
            self._fn(fp(A), fp(B), fp(bias), fp(C), M, N, K)
        else:
            self._fn(fp(A), fp(B), fp(C), M, N, K)
        return C


def _compile(src: str) -> ctypes.CDLL:
    key = hashlib.sha1(src.encode()).hexdigest()[:16]
    so = os.path.join(_CACHE_DIR, f"k_{key}.so")
    if not os.path.exists(so):
        c = os.path.join(_CACHE_DIR, f"k_{key}.c")
        with open(c, "w") as f:
            f.write(src)
        subprocess.run(["gcc", *CFLAGS, c, "-o", so, "-lm"],
                       check=True, capture_output=True)
    return ctypes.CDLL(so)


class Engine:
    """Executes an optimized graph, JIT-compiling fused linears on first use."""

    def __init__(self, g: Graph, cfg: TileConfig | None = None):
        self.g = g
        self.cfg = cfg or TileConfig()
        self._kernels: dict[tuple, CompiledKernel] = {}

    def _kernel_for(self, node):
        key = (node.attrs.get("activation", "none"),
               bool(node.attrs.get("has_bias")), self.cfg.label())
        if key not in self._kernels:
            self._kernels[key] = CompiledKernel(
                self.cfg, node.attrs.get("activation", "none"),
                bool(node.attrs.get("has_bias")))
        return self._kernels[key]

    def run(self, feeds: dict) -> list[np.ndarray]:
        env: dict[int, np.ndarray] = {}
        for n in self.g.topo():
            if n.op == "placeholder":
                env[n.id] = np.ascontiguousarray(feeds[n.name], np.float32)
            elif n.op == "const":
                env[n.id] = n.data
            elif n.op == "linear":
                x, w = env[n.inputs[0].id], env[n.inputs[1].id]
                bias = env[n.inputs[2].id] if n.attrs.get("has_bias") else None
                env[n.id] = self._kernel_for(n)(x, w, bias)
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
            else:
                raise NotImplementedError(n.op)
        return [env[o.id] for o in self.g.outputs]
