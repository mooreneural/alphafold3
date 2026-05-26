# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Memory-efficient fused GPU operations.

OuterProductMean Scan Fusion
-----------------------------
The standard OuterProductMean compute_chunk creates an intermediate tensor of
shape [full_N, C_outer, C_outer, chunk].  At N=1024, C_outer=32, chunk=128
(bfloat16) that is ~268 MB — peak memory that scales as O(N * C^2 * chunk).

This module replaces the two-einsum approach with a single jax.lax.scan over
the left-channel axis.  Each scan step materialises only:
  - outer_c:   [full_N, C_outer_right, chunk]  →  ~8 MB
  - contrib:   [chunk, full_N, F_out]           → ~32 MB
  - carry:     [chunk, full_N, F_out]           → ~32 MB

Total peak ≈ 72 MB, a ~3.7× reduction over the baseline (and ~6.7× vs the
naive non-chunked implementation).

The two computations are provably equivalent:
  Standard:  result[m,n,f] = Σ_{a,c_l,r}  left[a,c_l,m] · right[a,n,r] · W[c_l,r,f]
  Scan step: outer_c[n,r,m] = Σ_a left[a,c_l,m] · right[a,n,r]
             carry += Σ_r outer_c[n,r,m] · W[c_l,r,f]
  → identical result after scanning c_l ∈ [0, C_outer).
"""

import jax
import jax.numpy as jnp


def fused_outer_product_chunk(
    left_act: jnp.ndarray,
    right_act: jnp.ndarray,
    output_w: jnp.ndarray,
    output_b: jnp.ndarray,
) -> jnp.ndarray:
    """Memory-efficient OuterProductMean chunk via channel-wise scan.

    Computes the same result as the standard two-einsum implementation:
      step1 = einsum('acb,ade->dceb', left_T, right_act)    # large intermediate
      step2 = einsum('dceb,cef->dbf', step1, output_w) + b  # contract with W
      return transpose(step2, [1,0,2])                       # [chunk, full_N, F]

    but replaces it with a scan over the left-channel dimension that avoids
    materialising the [full_N, C_outer, C_outer, chunk] intermediate.

    Args:
      left_act:  [num_msa, chunk, C_outer]  — left projections for a residue chunk.
      right_act: [num_msa, full_N, C_outer] — right projections for all residues.
      output_w:  [C_outer, C_outer, F_out]  — output projection weights.
      output_b:  [F_out]                    — output projection bias.

    Returns:
      [chunk, full_N, F_out] — outer-product-mean contribution for this chunk.
    """
    chunk = left_act.shape[1]
    full_n = right_act.shape[1]
    f_out = output_w.shape[-1]
    dtype = left_act.dtype

    # Rearrange to [C_outer_left, num_msa, chunk] so lax.scan iterates channels.
    left_by_channel = jnp.transpose(left_act, (2, 0, 1))  # [C_l, msa, chunk]

    def scan_step(
        carry: jnp.ndarray,                          # [chunk, full_N, F_out]
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        left_c, w_c = inputs  # [msa, chunk], [C_outer_right, F_out]

        # Outer product for this left-channel, summed over MSA sequences.
        # Result: [full_N, C_outer_right, chunk]
        outer_c = jnp.einsum('am,anr->nrm', left_c, right_act)

        # Contract the right-channel dimension with W for this left-channel.
        # Result: [chunk, full_N, F_out]
        contrib = jnp.einsum('nrm,rf->mnf', outer_c, w_c)

        return carry + contrib, None

    init = jnp.zeros((chunk, full_n, f_out), dtype=dtype)

    # Scan iterates over C_outer_left (axis-0 of left_by_channel and output_w).
    result, _ = jax.lax.scan(
        scan_step,
        init,
        (left_by_channel, output_w),  # leading axis = C_outer_left
    )

    return result + output_b  # [chunk, full_N, F_out]
