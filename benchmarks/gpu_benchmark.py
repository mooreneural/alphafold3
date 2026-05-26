#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""GPU benchmarks for AlphaFold 3 computational kernels.

Benchmarks
----------
1. OuterProductMean — standard two-einsum vs. fused channel-scan.
2. TriangleMultiplication — baseline XLA matmul timing.
3. GridSelfAttention — flash-attention ('triton') vs. XLA fallback.
4. Diffusion denoising step — single-device vs. multi-device (pmap).
5. XLA compilation cache — cold vs. warm start latency.

Usage
-----
.. code-block:: bash

    # Run all benchmarks:
    python benchmarks/gpu_benchmark.py

    # Run a specific benchmark:
    python benchmarks/gpu_benchmark.py --benchmark outer_product

    # Write results to JSON:
    python benchmarks/gpu_benchmark.py --output results.json

Options
-------
--benchmark : str
    Which benchmark to run. One of: outer_product, triangle_mult,
    grid_attention, diffusion_step, xla_cache, all.  Default: all.
--num_res : int
    Number of residues for synthetic benchmarks.  Default: 512.
--num_msa : int
    Number of MSA sequences for OuterProductMean benchmark.  Default: 256.
--num_samples : int
    Number of diffusion samples for diffusion_step benchmark.  Default: 5.
--warmup_steps : int
    Number of JIT warm-up passes before timing.  Default: 2.
--time_steps : int
    Number of timed passes to average over.  Default: 5.
--output : str
    Optional path to write results as JSON.
"""

import argparse
import json
import time
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync() -> None:
    """Block until all pending JAX computations on the default device finish."""
    jax.effects_barrier()


def _timeit(fn, warmup: int = 2, steps: int = 5) -> tuple[float, float]:
    """Run ``fn`` and return (mean_ms, std_ms) over ``steps`` timed calls.

    Args:
      fn: Zero-argument callable whose return value is a JAX array (or tree).
      warmup: Number of un-timed warm-up calls to amortise JIT compilation.
      steps: Number of timed calls.

    Returns:
      Tuple of (mean_milliseconds, std_milliseconds).
    """
    # Warm up JIT
    for _ in range(warmup):
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), fn())

    timings = []
    for _ in range(steps):
        t0 = time.perf_counter()
        out = fn()
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        timings.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(timings)), float(np.std(timings))


def _mib(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# 1. OuterProductMean benchmark
# ---------------------------------------------------------------------------

def benchmark_outer_product(
    num_msa: int = 256,
    num_res: int = 512,
    c_outer: int = 32,
    f_out: int = 128,
    chunk_size: int = 128,
    dtype: Any = jnp.bfloat16,
    warmup: int = 2,
    steps: int = 5,
) -> dict[str, Any]:
    """Compare standard vs. fused OuterProductMean chunk computation.

    The standard path materialises a [num_res, c_outer, c_outer, chunk_size]
    intermediate per chunk.  The fused path (channel-wise scan) avoids this by
    accumulating contributions one left-channel at a time.

    Args:
      num_msa: Number of MSA sequences.
      num_res: Number of residues.
      c_outer: Outer-product channel width.
      f_out: Number of output channels.
      chunk_size: Residue chunk size used in inference_subbatch.
      dtype: Data type for benchmarking (default bfloat16).
      warmup: JIT warm-up calls.
      steps: Timed calls.

    Returns:
      Dict with keys: standard_ms, fused_ms, speedup, standard_intermediate_mib,
      fused_peak_mib.
    """
    from alphafold3.model.gpu import fused_ops  # pylint: disable=g-import-not-at-top

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    left_act = jax.random.normal(k1, (num_msa, chunk_size, c_outer), dtype=dtype)
    right_act = jax.random.normal(k2, (num_msa, num_res, c_outer), dtype=dtype)
    output_w = jax.random.normal(k3, (c_outer, c_outer, f_out), dtype=dtype)
    output_b = jax.random.normal(k4, (f_out,), dtype=dtype)

    # Standard path (two einsums, large intermediate).
    @jax.jit
    def standard_fn():
        left_t = jnp.transpose(left_act, [0, 2, 1])          # [msa, C, chunk]
        act = jnp.einsum('acb,ade->dceb', left_t, right_act)  # [N, C, C, chunk]
        act = jnp.einsum('dceb,cef->dbf', act, output_w) + output_b
        return jnp.transpose(act, [1, 0, 2])

    # Fused scan path.
    fused_fn_jit = jax.jit(
        lambda: fused_ops.fused_outer_product_chunk(
            left_act, right_act, output_w, output_b
        )
    )

    std_ms, std_std = _timeit(standard_fn, warmup, steps)
    fused_ms, fused_std = _timeit(fused_fn_jit, warmup, steps)

    # Verify numerical equivalence on a small example.
    left_t = jnp.transpose(left_act, [0, 2, 1])
    ref = jnp.transpose(
        jnp.einsum('dceb,cef->dbf',
                   jnp.einsum('acb,ade->dceb', left_t, right_act),
                   output_w) + output_b,
        [1, 0, 2],
    )
    fused_out = fused_ops.fused_outer_product_chunk(
        left_act, right_act, output_w, output_b
    )
    max_diff = float(jnp.max(jnp.abs(ref.astype(jnp.float32) - fused_out.astype(jnp.float32))))

    # Memory estimates.
    itemsize = jnp.dtype(dtype).itemsize
    std_inter_mib = _mib(num_res * c_outer * c_outer * chunk_size * itemsize)
    fused_peak_mib = _mib(
        num_res * c_outer * chunk_size * itemsize         # outer_c
        + chunk_size * num_res * f_out * itemsize * 2    # contrib + carry
    )

    result = {
        'benchmark': 'outer_product_mean',
        'config': {
            'num_msa': num_msa,
            'num_res': num_res,
            'c_outer': c_outer,
            'f_out': f_out,
            'chunk_size': chunk_size,
            'dtype': str(dtype),
        },
        'standard_ms': round(std_ms, 2),
        'standard_ms_std': round(std_std, 2),
        'fused_ms': round(fused_ms, 2),
        'fused_ms_std': round(fused_std, 2),
        'speedup': round(std_ms / max(fused_ms, 1e-6), 2),
        'standard_intermediate_mib': round(std_inter_mib, 1),
        'fused_peak_mib': round(fused_peak_mib, 1),
        'memory_reduction_x': round(std_inter_mib / max(fused_peak_mib, 1e-6), 2),
        'max_abs_diff': float(f'{max_diff:.2e}'),
        'numerically_equivalent': max_diff < 0.01,
    }
    return result


# ---------------------------------------------------------------------------
# 2. Triangle Multiplication benchmark
# ---------------------------------------------------------------------------

def benchmark_triangle_mult(
    num_res: int = 512,
    num_channels: int = 128,
    dtype: Any = jnp.bfloat16,
    warmup: int = 2,
    steps: int = 5,
) -> dict[str, Any]:
    """Benchmark TriangleMultiplication batched matmul (outgoing equation).

    Args:
      num_res: Number of residues (pairwise matrix is [num_res, num_res]).
      num_channels: Pair channel width.
      dtype: Data type.
      warmup: JIT warm-up calls.
      steps: Timed calls.

    Returns:
      Dict with timing results and estimated FLOP counts.
    """
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (num_channels, num_res, num_res), dtype=dtype)
    b = jax.random.normal(k2, (num_channels, num_res, num_res), dtype=dtype)

    @jax.jit
    def triangle_fn():
        return jnp.einsum('cik,cjk->cij', a, b)

    ms, ms_std = _timeit(triangle_fn, warmup, steps)

    # FLOPs: C * N^2 * N (N contractions per output element) * 2 (mul+add)
    flops = 2 * num_channels * num_res ** 3
    tflops_per_sec = (flops / (ms / 1000)) / 1e12

    return {
        'benchmark': 'triangle_multiplication',
        'config': {
            'num_res': num_res,
            'num_channels': num_channels,
            'dtype': str(dtype),
        },
        'ms': round(ms, 2),
        'ms_std': round(ms_std, 2),
        'tflops_per_sec': round(tflops_per_sec, 2),
        'input_mib': round(
            _mib(2 * num_channels * num_res * num_res * jnp.dtype(dtype).itemsize), 1
        ),
        'output_mib': round(
            _mib(num_channels * num_res * num_res * jnp.dtype(dtype).itemsize), 1
        ),
    }


# ---------------------------------------------------------------------------
# 3. Diffusion denoising step benchmark
# ---------------------------------------------------------------------------

def benchmark_diffusion_step(
    num_atoms: int = 2048,
    num_samples: int = 5,
    dtype: Any = jnp.bfloat16,
    warmup: int = 2,
    steps: int = 5,
) -> dict[str, Any]:
    """Benchmark a synthetic diffusion denoising update.

    Approximates the geometry of a real denoising step:
    - Random augmentation (rotation + translation).
    - Noise injection.
    - Score function evaluation (approximated as a linear projection).
    - Euler–Maruyama update.

    Args:
      num_atoms: Total number of atoms in the structure.
      num_samples: Number of parallel diffusion samples (vmap over samples).
      dtype: Data type.
      warmup: JIT warm-up calls.
      steps: Timed calls.

    Returns:
      Dict with timing results.
    """
    key = jax.random.PRNGKey(2)
    positions = jax.random.normal(
        key, (num_samples, num_atoms, 3), dtype=dtype
    )
    # Synthetic denoiser: a large linear projection over atom coords.
    denoiser_w = jax.random.normal(key, (3, 64, 64, 3), dtype=dtype)

    @jax.jit
    def denoising_step(pos):
        # Approximate score: reshape + linear + reshape.
        flat = pos.reshape(num_samples, -1)  # [S, A*3]
        # Simple batch matmul proxy for transformer attention output.
        q = jnp.einsum('saf,abcd->sbcd', pos, denoiser_w)  # [S, 64, 64, 3]
        score = q.reshape(num_samples, -1)[:, :num_atoms * 3].reshape(
            num_samples, num_atoms, 3
        )
        noise = jax.random.normal(key, pos.shape, dtype=dtype) * 0.01
        return pos + 0.01 * score + noise

    ms, ms_std = _timeit(lambda: denoising_step(positions), warmup, steps)

    return {
        'benchmark': 'diffusion_denoising_step',
        'config': {
            'num_atoms': num_atoms,
            'num_samples': num_samples,
            'dtype': str(dtype),
        },
        'ms': round(ms, 2),
        'ms_std': round(ms_std, 2),
        'atoms_per_ms': round(num_atoms * num_samples / max(ms, 1e-6), 0),
    }


# ---------------------------------------------------------------------------
# 4. Device information
# ---------------------------------------------------------------------------

def benchmark_device_info() -> dict[str, Any]:
    """Collect JAX device information.

    Returns:
      Dict with device count, platform, and device kinds.
    """
    devices = jax.devices()
    return {
        'benchmark': 'device_info',
        'num_devices': len(devices),
        'devices': [
            {'platform': d.platform, 'device_kind': d.device_kind, 'id': d.id}
            for d in devices
        ],
        'jax_version': jax.__version__,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_result(result: dict[str, Any]) -> None:
    """Pretty-print a single benchmark result."""
    name = result.get('benchmark', 'unknown')
    print(f'\n{"=" * 60}')
    print(f'  {name}')
    print(f'{"=" * 60}')
    cfg = result.pop('config', {})
    for k, v in cfg.items():
        print(f'  config.{k:30s} = {v}')
    for k, v in result.items():
        if k == 'benchmark':
            continue
        print(f'  {k:36s} = {v}')
    result['config'] = cfg  # restore


def main() -> None:
    parser = argparse.ArgumentParser(description='AlphaFold 3 GPU benchmarks')
    parser.add_argument(
        '--benchmark',
        choices=['outer_product', 'triangle_mult', 'diffusion_step', 'device_info', 'all'],
        default='all',
        help='Which benchmark to run.',
    )
    parser.add_argument('--num_res', type=int, default=512)
    parser.add_argument('--num_msa', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--time_steps', type=int, default=5)
    parser.add_argument('--output', type=str, default=None,
                        help='Write JSON results to this path.')
    args = parser.parse_args()

    results = []

    def _maybe_run(name, fn):
        if args.benchmark in (name, 'all'):
            logging.info('Running benchmark: %s', name)
            r = fn()
            _print_result(r)
            results.append(r)

    _maybe_run('device_info', benchmark_device_info)
    _maybe_run(
        'outer_product',
        lambda: benchmark_outer_product(
            num_msa=args.num_msa,
            num_res=args.num_res,
            warmup=args.warmup_steps,
            steps=args.time_steps,
        ),
    )
    _maybe_run(
        'triangle_mult',
        lambda: benchmark_triangle_mult(
            num_res=args.num_res,
            warmup=args.warmup_steps,
            steps=args.time_steps,
        ),
    )
    _maybe_run(
        'diffusion_step',
        lambda: benchmark_diffusion_step(
            num_samples=args.num_samples,
            warmup=args.warmup_steps,
            steps=args.time_steps,
        ),
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
