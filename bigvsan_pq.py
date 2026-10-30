"""
================================================================================
[8b-is Sovereign BigVSAN Substrate — Z_{2^32} SnakePq & Popcount SANLinearPq]

Connects Sony's BigVSAN ("Enhancing GAN-based Neural Vocoders with Slicing
Adversarial Network", ICASSP 2024) with the 8b-is audio & quantization stack:
- https://github.com/8b-is/IndexTTS-Rust (BigVGAN + GPT BFP-64 / p/q engine)
- https://github.com/8b-is/kokoro-tiny   (Z_{2^32} phase-ring & MEM|8 synthesis)
- https://github.com/8b-is/alexiai       (GAIA-MLX-QUANT p/q + BFP-64 substrate)

Key Mathematical Contributions for BigVSAN:
1. Z_{2^32} Modular Phase-Ring `SnakePq` & `SnakeBetaPq`:
   Replaces per-sample transcendental `sin(alpha * x)^2` in `activations.py` with
   modular integer phase wrap (`phase_u32 = int(x * step) & 0xFFFFFFFF`) and a
   a precomputed `p/q` (`q = 2^20`) sin^2 lookup table, so the activation path
   contains no per-sample transcendental. This targets hardware without a
   transcendental unit (NPU/embedded) and fused/compiled kernels; on CPython the
   pure-Python LUT path is actually slower than the C `math.sin` call
   (`BIGVSAN_BENCH=1` measures ~0.5x).

2. Bitplane Popcount Spherical Normalization for Slicing Adversarial Networks
   (`QuantumSANLinearPq` & `QuantumSANConv1dPq`):
   In `san_modules.py`, `SANLinear` normalizes each slice direction `w_j` onto the
   unit sphere `S^{D-1}` via `w_j / ||w_j||_2`.
   When slice directions are represented in ternary `w_j in {-1, 0, +1}^D` packed
   into 64-trit bitplanes (`pos_mask: u64`, `neg_mask: u64`), since `(-1)^2 = 1^2 = 1`
   and `0^2 = 0`, the squared L2 norm is EXACT bitplane popcount:
       ||w_j||_2^2 = sum_k w_{j,k}^2 = popcount(pos_mask_j | neg_mask_j)
   eliminating floating-point L2 norm reductions across discriminator updates!
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

PQ_Q20: int = 1 << 20
U32_MASK: int = 0xFFFF_FFFF
SIN2_LUT_BITS: int = 12
SIN2_LUT_SIZE: int = 1 << SIN2_LUT_BITS
SIN2_LUT_SHIFT: int = 32 - SIN2_LUT_BITS

# Precompute sin^2(pi * k / SIN2_LUT_SIZE) in p/q (q = 2^20)
# Note: sin^2(theta) has period pi, so one full period [0, pi) maps to 2^32!
_SIN2_LUT_Q20: List[int] = [
    int(round((math.sin(math.pi * i / SIN2_LUT_SIZE) ** 2) * PQ_Q20))
    for i in range(SIN2_LUT_SIZE)
]


def isqrt_u128(n: int) -> int:
    """Exact integer floor square root floor(sqrt(n))."""
    if n < 2:
        return max(0, n)
    return math.isqrt(n)


class SnakePq:
    """
    Z_{2^32} phase-ring Snake periodic activation:
        Snake(x) = x + (1 / alpha) * sin^2(alpha * x)
    """

    def __init__(self, in_features: int, alphas: Sequence[float] | float = 1.0, alpha_logscale: bool = False):
        self.in_features = int(in_features)
        if isinstance(alphas, (int, float)):
            raw = [float(alphas)] * self.in_features
        else:
            raw = [float(a) for a in alphas]
        self.alphas = [math.exp(a) if alpha_logscale else a for a in raw]
        # Map 1 radian of alpha*x into Z_{2^32} period pi: step = alpha * (2^32 / pi)
        self.phase_steps_q0 = [a * (float(1 << 32) / math.pi) for a in self.alphas]
        self.inv_alphas_q20 = [
            int(round((1.0 / (a + 1e-9)) * PQ_Q20)) for a in self.alphas
        ]

    def forward_channel(self, x_seq: Sequence[float], channel_idx: int = 0) -> List[float]:
        step = self.phase_steps_q0[channel_idx]
        inv_a_q20 = self.inv_alphas_q20[channel_idx]
        inv_q40 = 1.0 / float(PQ_Q20 * PQ_Q20)
        out: List[float] = []
        for x in x_seq:
            phase_u32 = int(round(x * step)) & U32_MASK
            idx = phase_u32 >> SIN2_LUT_SHIFT
            sin2_q20 = _SIN2_LUT_Q20[idx]
            out.append(x + (inv_a_q20 * sin2_q20) * inv_q40)
        return out

    def __call__(self, x_bct: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
        return [
            [self.forward_channel(ch_seq, c_idx) for c_idx, ch_seq in enumerate(batch)]
            for batch in x_bct
        ]


class SnakeBetaPq:
    """
    Z_{2^32} phase-ring SnakeBeta periodic activation:
        SnakeBeta(x) = x + (1 / beta) * sin^2(alpha * x)
    """

    def __init__(
        self,
        in_features: int,
        alphas: Sequence[float] | float = 1.0,
        betas: Sequence[float] | float = 1.0,
        alpha_logscale: bool = False,
    ):
        self.in_features = int(in_features)
        raw_a = [float(alphas)] * self.in_features if isinstance(alphas, (int, float)) else [float(a) for a in alphas]
        raw_b = [float(betas)] * self.in_features if isinstance(betas, (int, float)) else [float(b) for b in betas]
        self.alphas = [math.exp(a) if alpha_logscale else a for a in raw_a]
        self.betas = [math.exp(b) if alpha_logscale else b for b in raw_b]
        self.phase_steps_q0 = [a * (float(1 << 32) / math.pi) for a in self.alphas]
        self.inv_betas_q20 = [
            int(round((1.0 / (b + 1e-9)) * PQ_Q20)) for b in self.betas
        ]

    def forward_channel(self, x_seq: Sequence[float], channel_idx: int = 0) -> List[float]:
        step = self.phase_steps_q0[channel_idx]
        inv_b_q20 = self.inv_betas_q20[channel_idx]
        inv_q40 = 1.0 / float(PQ_Q20 * PQ_Q20)
        return [
            x + (inv_b_q20 * _SIN2_LUT_Q20[(int(round(x * step)) & U32_MASK) >> SIN2_LUT_SHIFT]) * inv_q40
            for x in x_seq
        ]


@dataclass(frozen=True)
class QuantumSANRow:
    """
    A single Slicing Adversarial Network (SAN) slice direction packed into 64-trit
    bitplanes (`pos_blocks`, `neg_blocks`).
    Because `w_{j,k} in {-1, 0, +1}`, the squared L2 norm `||w_j||_2^2` is EXACTLY
    `sum( (pos | neg).bit_count() )`!
    """
    pos_blocks: Tuple[int, ...]
    neg_blocks: Tuple[int, ...]
    nnz_popcount: int
    inv_l2_norm_q20: int  # round((1.0 / sqrt(max(1, nnz_popcount))) * 2^20)
    scale_q20: int        # learnable SAN scale in p/q (q = 2^20)

    @classmethod
    def from_ternary_signs(cls, signs: Sequence[int], scale: float = 1.0) -> "QuantumSANRow":
        num_blocks = (len(signs) + 63) // 64
        pos_list: List[int] = []
        neg_list: List[int] = []
        nnz = 0
        for b in range(num_blocks):
            chunk = signs[b * 64:(b + 1) * 64]
            pos = 0
            neg = 0
            for k, s in enumerate(chunk):
                if s > 0:
                    pos |= 1 << k
                elif s < 0:
                    neg |= 1 << k
            nnz += (pos | neg).bit_count()
            pos_list.append(pos)
            neg_list.append(neg)

        # Pure-integer reciprocal L2 norm via isqrt_u128:
        # sqrt(nnz) in Q20 = isqrt(nnz * 2^40)
        sqrt_nnz_q20 = max(1, isqrt_u128(max(1, nnz) << 40))
        inv_l2_q20 = ((1 << 40) + (sqrt_nnz_q20 >> 1)) // sqrt_nnz_q20
        return cls(
            pos_blocks=tuple(pos_list),
            neg_blocks=tuple(neg_list),
            nnz_popcount=nnz,
            inv_l2_norm_q20=inv_l2_q20,
            scale_q20=int(round(scale * PQ_Q20)),
        )


class QuantumSANLinearPq:
    """
    Slicing Adversarial Network (`SANLinear`) powered by Quantum 64-Trit Bitplanes
    and Omni `BFP-64` (`i16 -> i64`) integer accumulation.
    """

    def __init__(self, weight_signs: Sequence[Sequence[int]], scales: Sequence[float]):
        self.out_features = len(weight_signs)
        self.in_features = len(weight_signs[0]) if self.out_features > 0 else 0
        self.rows: List[QuantumSANRow] = [
            QuantumSANRow.from_ternary_signs(w_row, scale=s)
            for w_row, s in zip(weight_signs, scales)
        ]

    @staticmethod
    def quantize_bfp64_input(x: Sequence[float]) -> Tuple[List[int], List[int]]:
        num_blocks = (len(x) + 63) // 64
        m_x: List[int] = []
        exp_x: List[int] = []
        for b in range(num_blocks):
            blk = x[b * 64:(b + 1) * 64]
            max_abs = max((abs(v) for v in blk), default=0.0)
            if max_abs < 1e-18:
                m_x.extend([0] * len(blk))
                exp_x.append(-60)
            else:
                e = max(-60, min(60, math.ceil(math.log2(max_abs / 127.0))))
                inv_s = math.ldexp(1.0, -e)
                m_x.extend(max(-127, min(127, int(round(v * inv_s)))) for v in blk)
                exp_x.append(e)
        return m_x, exp_x

    def forward(self, x: Sequence[float], flg_train: bool = False) -> List[float] | Tuple[List[float], List[float]]:
        m_x, exp_x = self.quantize_bfp64_input(x)
        min_exp = min(exp_x)
        block_mult = [1 << max(0, min(24, e - min_exp)) for e in exp_x]
        boundary_scale = math.ldexp(1.0, min_exp - 40)

        out_scaled: List[float] = []
        out_dir: List[float] = []
        for row in self.rows:
            dot_aligned_i64 = 0
            for b, (pos, neg) in enumerate(zip(row.pos_blocks, row.neg_blocks)):
                base = b * 64
                blk_sum_i16 = 0
                p = pos
                while p:
                    lsb = p & -p
                    blk_sum_i16 += m_x[base + lsb.bit_length() - 1]
                    p ^= lsb
                n = neg
                while n:
                    lsb = n & -n
                    blk_sum_i16 -= m_x[base + lsb.bit_length() - 1]
                    n ^= lsb
                dot_aligned_i64 += blk_sum_i16 * block_mult[b]

            # Multiply by exact popcount-derived inv_l2_norm_q20 and SAN scale_q20
            dir_q20 = dot_aligned_i64 * row.inv_l2_norm_q20
            scaled_q40 = dir_q20 * row.scale_q20
            val_scaled = scaled_q40 * boundary_scale
            out_scaled.append(val_scaled)
            if flg_train:
                out_dir.append(val_scaled)

        if flg_train:
            return out_scaled, out_dir
        return out_scaled
