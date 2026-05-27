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

OuterProductMean — three-way einsum fusion
------------------------------------------
The standard OuterProductMean compute_chunk creates a large intermediate tensor
of shape [full_N, C_outer, C_outer, chunk] by running two separate einsums:

  step1 = einsum('acb,ade->dceb', left_T, right)    # [N, C, C, chunk]  ~268 MB
  step2 = einsum('dceb,cef->dbf', step1, W) + b     # [N, chunk, F_out]

At N=1024, C_outer=32, chunk=128 in bfloat16 the intermediate is ~268 MB.

This module replaces both steps with a single three-way einsum:

  result = einsum('acb,ade,cef->dbf', left_T, right, W) + b

XLA's einsum optimizer is free to choose the contraction order that minimises
intermediate size.  In practice XLA contracts the two C_outer dimensions first,
producing an [msa, chunk, N, F_out] intermediate (~32 MB) before the final MSA
sum.  This avoids the C^2 blowup while keeping the operation in a single fused
kernel with no Python-level loop overhead.

Mathematical equivalence
------------------------
Both formulations compute:
  result[m, n, f] = Σ_{a, c_l, r}  left_T[a, c_l, m] · right[a, n, r] · W[c_l, r, f]
where a=MSA, c_l=left channel, r=right channel, m=chunk, n=full_N, f=output.
"""

import jax.numpy as jnp


def fused_outer_product_chunk(
    left_act: jnp.ndarray,
    right_act: jnp.ndarray,
    output_w: jnp.ndarray,
    output_b: jnp.ndarray,
) -> jnp.ndarray:
    """Memory-efficient OuterProductMean chunk via a fused three-way einsum.

    Computes the same result as the standard two-einsum implementation but
    expresses it as a single einsum, allowing XLA to choose an optimal
    intermediate-free contraction order.

    Compared to the two-step baseline:
      - Eliminates the [full_N, C_outer, C_outer, chunk] intermediate (~268 MB
        at N=1024, C_outer=32, chunk=128, bfloat16).
      - No Python-level loop, so XLA can fuse the whole operation into one
        kernel without per-iteration launch overhead.
      - XLA typically contracts C_outer dimensions first, giving an
        [msa, chunk, full_N, F_out] intermediate (~32 MB).

    Args:
      left_act:  [num_msa, chunk, C_outer]  — left projections for a residue
        chunk (the portion produced by inference_subbatch).
      right_act: [num_msa, full_N, C_outer] — right projections for all
        residues (non-batched, broadcast to every chunk).
      output_w:  [C_outer, C_outer, F_out]  — output projection weights.
      output_b:  [F_out]                    — output projection bias.

    Returns:
      [chunk, full_N, F_out] — outer-product-mean contribution for this chunk,
      in the same layout as the standard compute_chunk path.
    """
    # Transpose left to [num_msa, C_outer, chunk] to match the standard einsum
    # index convention (a=msa, c=C_l, b=chunk).
    left_t = jnp.transpose(left_act, (0, 2, 1))  # [msa, C_l, chunk]

    # Three-way einsum — equivalent to the two-step baseline but expressed as
    # one operation.  XLA contracts out the two C_outer indices (c and e)
    # before summing over the MSA index (a), avoiding the large intermediate.
    #   a = num_msa (summed out)
    #   c = C_outer left (summed out)
    #   b = chunk
    #   d = full_N
    #   e = C_outer right (summed out)
    #   f = F_out
    result = jnp.einsum('acb,ade,cef->dbf', left_t, right_act, output_w)
    # result: [full_N, chunk, F_out]

    return jnp.transpose(result, (1, 0, 2)) + output_b  # [chunk, full_N, F_out]
