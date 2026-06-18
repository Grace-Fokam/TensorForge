# TensorForge

**A tiny optimizing tensor compiler with a JIT backend — built from scratch in Python + C.**

TensorForge takes a neural network written in a small Python eDSL, lowers it to
an SSA dataflow IR, runs a pipeline of classic compiler optimizations, generates
tiled C kernels, JIT-compiles them with `gcc`, autotunes the tile sizes by
measurement, and validates every result against an executable numpy reference.

It is a compact but honest model of what production ML compilers (XLA, TVM,
PyTorch Inductor, Triton) actually do — minus the 200k lines of code.

```
   eDSL          IR            optimizer              backend
  relu(x@W+b) ─► dataflow ─► fold/CSE/DCE/fuse ─► tiled C ─► gcc JIT ─► run
                  graph                            codegen     │
                                                               └─► autotuner
                                                                   (measure & pick)
```

## Why this is interesting

The headline transformation is **operator fusion**. A linear layer naively runs
as three separate kernels — `matmul`, `bias add`, `activation` — each of which
streams the entire activation matrix through memory. TensorForge's `fuse_linear`
pass collapses them into one `linear` op that applies bias + activation as a
**register-resident epilogue**, eliminating two full memory round-trips. This is
the single most important optimization in real ML compilers.

```
$ python examples/mlp.py

Built MLP graph: 8 ops before optimization
  ...matmul, add, relu, matmul, add, relu, matmul, add...

  [iter 0] fuse_linear: 3 rewrites

After optimization: 3 ops
  %linear15 = linear(x, W0, b0)  [has_bias=True, activation=relu]
  %linear16 = linear(%linear15, W1, b1)  [has_bias=True, activation=relu]
  %linear17 = linear(%linear16, Wout, bout)  [has_bias=True, activation=none]

correctness vs numpy reference: max abs error = 5.25e-06
JIT (default tiles)   : 115.330 ms
JIT (autotuned tiles) :  99.881 ms   (1.15x over default tiles)
```

## What's implemented

**Frontend** (`tensor.py`) — operator-overloaded `Tensor` handles build the
graph lazily; `relu(x @ w + b)` *is* the program.

**IR** (`ir.py`, `graph.py`) — typed SSA dataflow graph with structural hashing,
use-def chains, topological traversal, and `replace_all_uses` rewiring.

**Optimization passes** (`passes.py`), run to a fixed point:
| pass | what it does |
|------|--------------|
| constant folding | evaluates all-constant subgraphs at compile time |
| algebraic simplification | `x+0`, `x*1`, `x*0`, `-(-x)` rewrites |
| common subexpression elimination | merges structurally identical nodes |
| **operator fusion** | `matmul → +bias → activation` ⟹ one `linear` |
| dead code elimination | drops nodes unreachable from the outputs |

**Backend** (`codegen.py`, `runtime.py`) — emits cache-blocked
(`MC`/`NC`/`KC`) + register-blocked (`MR`×`NR`) GEMM C with a fused epilogue,
compiles with `gcc -O3 -march=native`, loads via `ctypes`.

**Autotuner** (`autotune.py`) — compiles a search space of tile configs,
benchmarks each on representative data, and selects the fastest. The classic
*compile → measure → select* loop, with a hand-written search space instead of
a learned cost model.

**Reference semantics** (`reference.py`) — a numpy interpreter for the IR that
serves as the ground truth every compiled run is checked against.

## Run it

```bash
python examples/mlp.py          # build → optimize → autotune → benchmark
python examples/show_passes.py  # watch fold/simplify/CSE/DCE clean a messy graph
python tests/test_correctness.py
```

No dependencies beyond `numpy` and a C compiler (`gcc`).

## Honest limitations

This is a portfolio compiler, not a BLAS replacement. The micro-kernel is scalar
C (no hand-vectorized intrinsics, no packing), single-threaded, and 2D-dense
only. numpy's multithreaded BLAS will beat it on raw GEMM — the point is to
demonstrate the **compiler pipeline** (IR, passes, fusion, codegen, autotuning),
which is the part that transfers directly to real systems.

## Roadmap (what a v2 would add)
- Symbolic shapes + a real shape-inference pass
- A packing stage and AVX intrinsics in the micro-kernel
- Conv via implicit-GEMM lowering (im2col-free)
- A learned cost model for the autotuner
- Multi-threaded scheduling of independent tiles
