"""C code generation for the fused ``linear`` op.

We emit a single tiled, register-blocked GEMM with a fused epilogue (bias +
activation). The interesting compiler content lives here:

* **Cache blocking** along M/N/K (``MC``/``NC``/``KC``) keeps the working set
  resident in cache so the inner kernel streams from L1/L2 rather than RAM.
* **Register blocking** (``MR``x``NR`` micro-tile) keeps a small block of the
  output in registers across the K loop, maximizing arithmetic intensity.
* The **epilogue** (bias add + activation) is applied while the tile is still
  hot in registers, which is the whole point of fusion.

These tile parameters are exactly what the autotuner searches over.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileConfig:
    MC: int = 64    # cache block rows of A / C
    NC: int = 64    # cache block cols of B / C
    KC: int = 256   # cache block of the contraction dim
    MR: int = 4     # register micro-tile rows
    NR: int = 4     # register micro-tile cols

    def label(self):
        return f"MC{self.MC}_NC{self.NC}_KC{self.KC}_MR{self.MR}_NR{self.NR}"


_ACT_EXPR = {
    "none": "{v}",
    "relu": "({v} > 0.0f ? {v} : 0.0f)",
    "sigmoid": "(1.0f / (1.0f + expf(-({v}))))",
    "tanh": "tanhf({v})",
}


def generate_kernel(cfg: TileConfig, activation: str = "relu",
                    has_bias: bool = True) -> str:
    """Return C source for one fused-linear kernel specialized to ``cfg``."""
    act = _ACT_EXPR[activation].format(v="acc")
    bias_decl = "const float* restrict bias," if has_bias else ""
    bias_add = "acc += bias[j + jj];" if has_bias else ""

    return f"""
#include <math.h>
#include <string.h>

/* Fused linear: C = act( A[M,K] @ B[K,N] (+ bias[N]) )
   Tiling: MC={cfg.MC} NC={cfg.NC} KC={cfg.KC}  micro {cfg.MR}x{cfg.NR}
   Config: {cfg.label()} act={activation} bias={int(has_bias)} */
void linear_kernel(const float* restrict A,
                   const float* restrict B,
                   {bias_decl}
                   float* restrict C,
                   int M, int N, int K)
{{
    const int MC = {cfg.MC}, NC = {cfg.NC}, KC = {cfg.KC};
    const int MR = {cfg.MR}, NR = {cfg.NR};

    /* Cache-blocked loops (jc, kc, ic) */
    for (int jc = 0; jc < N; jc += NC) {{
        int nb = (jc + NC <= N) ? NC : (N - jc);
        for (int kc = 0; kc < K; kc += KC) {{
            int kb = (kc + KC <= K) ? KC : (K - kc);
            int first_k = (kc == 0);
            for (int ic = 0; ic < M; ic += MC) {{
                int mb = (ic + MC <= M) ? MC : (M - ic);

                /* Register-blocked micro-kernel over the MCxNC panel */
                for (int i = ic; i < ic + mb; i += MR) {{
                    int mr = (i + MR <= ic + mb) ? MR : (ic + mb - i);
                    for (int j = jc; j < jc + nb; j += NR) {{
                        int nr = (j + NR <= jc + nb) ? NR : (jc + nb - j);

                        float acc_tile[{cfg.MR}][{cfg.NR}];
                        for (int ii = 0; ii < mr; ii++)
                            for (int jj = 0; jj < nr; jj++)
                                acc_tile[ii][jj] = first_k ? 0.0f
                                    : C[(i+ii)*N + (j+jj)];

                        for (int p = kc; p < kc + kb; p++) {{
                            for (int ii = 0; ii < mr; ii++) {{
                                float a = A[(i+ii)*K + p];
                                for (int jj = 0; jj < nr; jj++)
                                    acc_tile[ii][jj] += a * B[p*N + (j+jj)];
                            }}
                        }}

                        int last_k = (kc + kb >= K);
                        for (int ii = 0; ii < mr; ii++) {{
                            for (int jj = 0; jj < nr; jj++) {{
                                float acc = acc_tile[ii][jj];
                                if (last_k) {{
                                    {bias_add}
                                    acc = {act};
                                }}
                                C[(i+ii)*N + (j+jj)] = acc;
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
"""
