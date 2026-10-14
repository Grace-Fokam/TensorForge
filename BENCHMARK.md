# Measured benchmark

Measured on October 14, 2026 (UTC) in a Linux x86-64 virtual machine:
AMD EPYC 9V74, Python 3.12.14, NumPy 2.3.5, GCC 13.3.0.
Command: `python examples/mlp.py` after installation.

The default seeded MLP uses batch 256, input 512, two 1024-wide hidden
layers, and output 256. Optimization reduced eight ops to three fused linears.
The generated C uses `-O3 -march=native -funroll-loops -ffast-math -shared -fPIC`.
The search evaluates 81 configurations, with two warmups and ten timed calls
per candidate. Whole-graph timing uses three warmups and twenty calls.
Compilation is excluded from the reported warmed latency. BLAS thread settings
were left at the environment defaults.

| Measurement | Result |
| --- | ---: |
| Maximum absolute error vs original NumPy graph | 5.25e-06 |
| NumPy reference graph | 4.580 ms |
| Default JIT tiles | 124.299 ms |
| Selected JIT tiles | 75.565 ms |
| Default / selected latency | 1.64x |

Selected configuration: `MC32_NC128_KC256_MR4_NR8`.
This is one measured run, not a statistical performance guarantee. Timing can
vary with load, CPU, GCC, NumPy/BLAS, thread settings and cache state. The
previous README's 115.330 / 99.881 ms (1.15x) values remain labeled illustrative;
those exact latencies were not reproduced here.

The lightweight `--quick` demo does not autotune or make performance claims.
