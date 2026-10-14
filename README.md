# TensorForge

**A tiny optimizing tensor compiler with a JIT backend, built from scratch in Python + C.**

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
as three separate kernels:  `matmul`, `bias add`, `activation`  each of which
streams the entire activation matrix through memory. TensorForge's `fuse_linear`
pass collapses them into one `linear` op that applies bias + activation as a
**register-resident epilogue**, eliminating two full memory round-trips. This is
the single most important optimization in real ML compilers.

The following output was previously reported as an illustrative example; its
1.15x speedup is not a guarantee or a reproduced measurement.

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

**Frontend** (`tensor.py`): operator-overloaded `Tensor` handles build the
graph lazily; `relu(x @ w + b)` *is* the program.

**IR** (`ir.py`, `graph.py`), typed SSA dataflow graph with structural hashing,
use-def chains, topological traversal, and `replace_all_uses` rewiring.

**Optimization passes** (`passes.py`), run to a fixed point:
| pass | what it does |
|------|--------------|
| constant folding | evaluates all-constant subgraphs at compile time |
| algebraic simplification | `x+0`, `x*1`, `-(-x)` rewrites |
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

**Reference semantics** (`reference.py`): a numpy interpreter for the IR that
serves as the ground truth every compiled run is checked against.

## Install and run

Supported and verified: Linux with Python 3.10+ and `gcc` on PATH. On Debian/Ubuntu,
install the native dependencies with `sudo apt-get install gcc python3-venv`.
Windows users can use WSL2 with Ubuntu. Native Windows and macOS are not verified;
the runtime currently uses Linux-style shared libraries and GCC flags.

```bash
git clone https://github.com/Grace-Fokam/TensorForge.git
cd TensorForge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[test]"
python -m pytest -q
python tests/test_correctness.py
python examples/show_passes.py
python examples/mlp.py --quick  # small end-to-end C JIT correctness check
python examples/mlp.py          # full 81-config autotuning search and benchmark
```

Use `python -m pip install -e ".[test]"` for development. The lowercase
`tensorforge/` directory is the installed package; `examples/` and `tests/` contain
standalone scripts. No `sys.path` changes or repository-name assumptions are needed.
Runtime dependencies are NumPy and GCC; pytest is included only in the `test` extra.
The JIT requires a writable temporary directory and permission to load native code.
Generated C and shared libraries are cached in the temporary `tensorforge_cache`
directory; delete that directory to force recompilation after changing compiler
flags or moving to different hardware (`-march=native` specializes to the host).

For a short demonstration, run `show_passes.py` followed by `mlp.py --quick`:
the former asserts preserved semantics after fold/simplify/CSE, while the latter
reduces an eight-op MLP to three fused linears, compiles C, and asserts both default
and selected tile outputs against the original unoptimized NumPy graph.
GitHub Actions runs all nine correctness tests and these lightweight demos on
Ubuntu with Python 3.10 and 3.12. Full autotuning is intentionally outside CI.

The full demo tunes the first layer and applies that configuration to all layers.
It reports warmed execution latency (including allocations and Python dispatch),
excluding compilation. A configuration that wins on the first layer can lose on
the complete MLP. NumPy timing also depends on its BLAS and thread settings.
See [BENCHMARK.md](BENCHMARK.md) for measured results and reproduction details.

## Honest limitations

This is a portfolio compiler, not a BLAS replacement. The micro-kernel is scalar
C (no hand-vectorized intrinsics, no packing), single-threaded, and 2D-dense
only. numpy's multithreaded BLAS will beat it on raw GEMM, the point is to
demonstrate the **compiler pipeline** (IR, passes, fusion, codegen, autotuning),
which is the part that transfers directly to real systems.

## Roadmap (what a v2 would add)
- Symbolic shapes + a real shape-inference pass
- A packing stage and AVX intrinsics in the micro-kernel
- Conv via implicit-GEMM lowering (im2col-free)
- A learned cost model for the autotuner
- Multi-threaded scheduling of independent tiles
