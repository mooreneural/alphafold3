"""Tests for alphafold3.model.gpu.fused_ops.

Mathematical-correctness tests comparing fused_outer_product_chunk against the
standard two-einsum implementation.  These tests require JAX; on machines
without JAX they are automatically skipped.

What we verify
--------------
1. **Numerical equivalence** — fused output matches standard output within
   floating-point tolerance for float32, float16, and bfloat16.
2. **Shape** — output shape is [chunk, full_N, F_out] for all input sizes.
3. **Bias term** — output includes the bias correctly.
4. **Zero inputs** — all-zero activations yield a result equal to the bias.
5. **Single MSA row** — degenerate num_msa=1 case is handled correctly.
6. **Large sequence** — correctness at N=1024 (primary target scenario).
7. **Gradient compatibility** — jax.grad passes through the fused path without
   error (important for future training use).
"""

import unittest


try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _standard_chunk(left_act, right_act, output_w, output_b):
    """Reference: the original two-einsum implementation."""
    left_t = jnp.transpose(left_act, [0, 2, 1])
    act = jnp.einsum('acb,ade->dceb', left_t, right_act)
    act = jnp.einsum('dceb,cef->dbf', act, output_w) + output_b
    return jnp.transpose(act, [1, 0, 2])  # [chunk, full_N, F_out]


@unittest.skipUnless(_JAX_AVAILABLE, 'JAX not installed — skipping fused_ops tests')
class TestFusedOuterProductChunk(unittest.TestCase):

    def _load_module(self):
        import importlib
        return importlib.import_module('alphafold3.model.gpu.fused_ops')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_inputs(self, num_msa, chunk, full_n, c_outer, f_out, dtype, seed=0):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        left_act  = jax.random.normal(k1, (num_msa, chunk, c_outer), dtype=dtype)
        right_act = jax.random.normal(k2, (num_msa, full_n, c_outer), dtype=dtype)
        output_w  = jax.random.normal(k3, (c_outer, c_outer, f_out), dtype=dtype)
        output_b  = jax.random.normal(k4, (f_out,), dtype=dtype)
        return left_act, right_act, output_w, output_b

    def _assert_close(self, ref, got, atol, msg=''):
        """Assert allclose after casting both to float32."""
        ref_f = ref.astype(jnp.float32)
        got_f = got.astype(jnp.float32)
        max_diff = float(jnp.max(jnp.abs(ref_f - got_f)))
        self.assertLessEqual(
            max_diff, atol,
            f'{msg}  max_abs_diff={max_diff:.4e}  atol={atol:.4e}',
        )

    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_output_shape_standard(self):
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=16, chunk=8, full_n=32, c_outer=4, f_out=16,
            dtype=jnp.float32,
        )
        out = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        self.assertEqual(out.shape, (8, 32, 16))

    def test_output_shape_asymmetric(self):
        mod = self._load_module()
        # Different chunk vs full_n
        left_act, right_act, w, b = self._random_inputs(
            num_msa=8, chunk=4, full_n=64, c_outer=8, f_out=32,
            dtype=jnp.float32,
        )
        out = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        self.assertEqual(out.shape, (4, 64, 32))

    # ------------------------------------------------------------------
    # Numerical equivalence tests
    # ------------------------------------------------------------------

    def test_equivalence_float32(self):
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=16, chunk=8, full_n=32, c_outer=4, f_out=16,
            dtype=jnp.float32,
        )
        ref = _standard_chunk(left_act, right_act, w, b)
        got = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        self._assert_close(ref, got, atol=1e-5, msg='float32')

    def test_equivalence_float16(self):
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=16, chunk=8, full_n=32, c_outer=4, f_out=16,
            dtype=jnp.float16,
        )
        ref = _standard_chunk(left_act, right_act, w, b)
        got = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        # float16 accumulation is less precise; use a looser tolerance.
        self._assert_close(ref, got, atol=0.05, msg='float16')

    def test_equivalence_bfloat16(self):
        """bfloat16 is the default inference dtype in AF3."""
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=32, chunk=16, full_n=64, c_outer=8, f_out=32,
            dtype=jnp.bfloat16,
        )
        ref = _standard_chunk(left_act, right_act, w, b)
        got = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        # bfloat16 has only 7 mantissa bits; relative error ~0.01 is expected.
        self._assert_close(ref, got, atol=0.05, msg='bfloat16')

    # ------------------------------------------------------------------
    # Edge-case correctness tests
    # ------------------------------------------------------------------

    def test_bias_only_when_activations_zero(self):
        """When left_act and right_act are all zeros the output equals output_b."""
        mod = self._load_module()
        num_msa, chunk, full_n, c_outer, f_out = 8, 4, 16, 4, 8
        dtype = jnp.float32
        left_act  = jnp.zeros((num_msa, chunk, c_outer), dtype=dtype)
        right_act = jnp.zeros((num_msa, full_n, c_outer), dtype=dtype)
        key = jax.random.PRNGKey(7)
        output_w = jax.random.normal(key, (c_outer, c_outer, f_out), dtype=dtype)
        output_b = jax.random.normal(jax.random.PRNGKey(8), (f_out,), dtype=dtype)

        out = mod.fused_outer_product_chunk(left_act, right_act, output_w, output_b)
        expected = jnp.broadcast_to(output_b, (chunk, full_n, f_out))
        self._assert_close(expected, out, atol=1e-6, msg='zero_activations')

    def test_single_msa_row(self):
        """Degenerate case: only one MSA sequence."""
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=1, chunk=4, full_n=8, c_outer=4, f_out=8,
            dtype=jnp.float32, seed=99,
        )
        ref = _standard_chunk(left_act, right_act, w, b)
        got = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        self._assert_close(ref, got, atol=1e-5, msg='single_msa_row')

    def test_single_channel(self):
        """Degenerate case: c_outer=1."""
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=4, chunk=4, full_n=8, c_outer=1, f_out=4,
            dtype=jnp.float32, seed=42,
        )
        ref = _standard_chunk(left_act, right_act, w, b)
        got = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        self._assert_close(ref, got, atol=1e-5, msg='single_channel')

    def test_large_sequence_bfloat16(self):
        """Primary target: N=1024, bfloat16 — the regime where fused helps most."""
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=32, chunk=128, full_n=1024, c_outer=32, f_out=128,
            dtype=jnp.bfloat16, seed=5,
        )
        ref = _standard_chunk(left_act, right_act, w, b)
        got = mod.fused_outer_product_chunk(left_act, right_act, w, b)
        self.assertEqual(got.shape, (128, 1024, 128))
        self._assert_close(ref, got, atol=0.1, msg='large_seq_bfloat16')

    # ------------------------------------------------------------------
    # JIT compilation test
    # ------------------------------------------------------------------

    def test_jit_compilable(self):
        """fused_outer_product_chunk should be JIT-compilable without errors."""
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=8, chunk=4, full_n=16, c_outer=4, f_out=8,
            dtype=jnp.float32, seed=1,
        )
        jitted = jax.jit(mod.fused_outer_product_chunk)
        out = jitted(left_act, right_act, w, b)
        self.assertEqual(out.shape, (4, 16, 8))

    # ------------------------------------------------------------------
    # Gradient compatibility test
    # ------------------------------------------------------------------

    def test_grad_through_fused_op(self):
        """jax.grad should successfully differentiate through the fused op."""
        mod = self._load_module()
        left_act, right_act, w, b = self._random_inputs(
            num_msa=4, chunk=4, full_n=8, c_outer=4, f_out=4,
            dtype=jnp.float32, seed=3,
        )

        def loss(left):
            out = mod.fused_outer_product_chunk(left, right_act, w, b)
            return jnp.sum(out)

        grad = jax.grad(loss)(left_act)
        self.assertEqual(grad.shape, left_act.shape)
        # Gradient should be finite everywhere.
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))), 'gradient contains non-finite values')


if __name__ == '__main__':
    unittest.main()
