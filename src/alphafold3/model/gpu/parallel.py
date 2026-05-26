# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Multi-GPU inference utilities for AlphaFold 3.

Overview
--------
AlphaFold 3 generates ``num_samples`` independent structure predictions via its
diffusion head.  By default all samples run on a single device (vmapped over the
sample axis).  On machines with N GPUs we can split those samples across devices
using ``jax.pmap``, yielding a near-linear speed-up for the diffusion phase.

The Evoformer trunk (pairformer stack) runs once and its ``embeddings`` dict is
broadcast to all devices; only the diffusion denoising loop and confidence head
are parallelised.

Usage example
-------------
.. code-block:: python

    import jax
    from alphafold3.model.gpu import parallel

    num_devices = jax.device_count()
    # Ensure num_samples is divisible by the number of devices.
    num_samples = parallel.round_samples_to_devices(20, num_devices)

    sampler = parallel.MultiDeviceDiffusionSampler(model_runner, num_devices)
    all_positions = sampler.sample(batch, embeddings, key, num_samples)
"""

import math
from collections.abc import Callable
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np


PyTree = Any


def round_samples_to_devices(num_samples: int, num_devices: int) -> int:
    """Round ``num_samples`` up so it is evenly divisible by ``num_devices``.

    Args:
      num_samples: Desired number of diffusion samples.
      num_devices: Number of accelerator devices available.

    Returns:
      The smallest integer >= ``num_samples`` that is divisible by
      ``num_devices``.
    """
    return math.ceil(num_samples / num_devices) * num_devices


def _split_along_leading_axis(
    tree: PyTree,
    num_devices: int,
) -> PyTree:
    """Reshape leading axis from [N, ...] to [num_devices, N//num_devices, ...].

    Args:
      tree: A JAX pytree whose arrays all share the same leading dimension N.
      num_devices: Number of devices to split across.  N must be divisible.

    Returns:
      A pytree with the leading axis split into [num_devices, N//num_devices].

    Raises:
      ValueError: If the leading dimension is not divisible by ``num_devices``.
    """
    def _split(arr: jnp.ndarray) -> jnp.ndarray:
        n = arr.shape[0]
        if n % num_devices != 0:
            raise ValueError(
                f'Leading dimension {n} is not divisible by num_devices'
                f' {num_devices}.'
            )
        return arr.reshape((num_devices, n // num_devices) + arr.shape[1:])

    return jax.tree_util.tree_map(_split, tree)


def _concat_along_leading_axis(tree: PyTree) -> PyTree:
    """Merge [num_devices, shard_size, ...] back to [N, ...].

    Args:
      tree: A JAX pytree whose arrays all have shape
        [num_devices, shard_size, ...].

    Returns:
      A pytree with arrays of shape [N, ...] where N = num_devices * shard_size.
    """
    def _concat(arr: jnp.ndarray) -> jnp.ndarray:
        return arr.reshape((-1,) + arr.shape[2:])

    return jax.tree_util.tree_map(_concat, tree)


def make_parallel_diffusion_fn(
    single_device_fn: Callable[..., PyTree],
) -> Callable[..., PyTree]:
    """Wrap a single-device diffusion sampling function for multi-GPU execution.

    The wrapped function:

    1. Expects ``positions`` with a leading ``num_samples`` axis.
    2. Reshapes that axis into ``[num_devices, num_samples // num_devices, ...]``.
    3. Broadcasts non-sample arguments (keys, masks, embeddings) to all devices.
    4. Calls ``jax.pmap`` of ``single_device_fn`` across devices.
    5. Concatenates per-device outputs back into a single leading ``num_samples``
       axis and returns the result on the first device.

    Args:
      single_device_fn: A function that runs diffusion sampling for a sub-batch
        of samples on a single device.  Its first argument must be an array with
        a leading sample axis.

    Returns:
      A ``pmap``-wrapped version of ``single_device_fn``.
    """
    pmapped = jax.pmap(single_device_fn, axis_name='devices')

    def parallel_fn(positions: jnp.ndarray, *args: Any, **kwargs: Any) -> PyTree:
        num_devices = jax.device_count()
        num_samples = positions.shape[0]

        if num_samples % num_devices != 0:
            raise ValueError(
                f'num_samples ({num_samples}) must be divisible by the number of'
                f' available devices ({num_devices}).  Use'
                f' round_samples_to_devices() to pad.'
            )

        # Split sample axis across devices: [N, ...] -> [D, N/D, ...]
        positions_split = _split_along_leading_axis(positions, num_devices)

        logging.info(
            'Multi-GPU diffusion: %d samples across %d devices (%d per device)',
            num_samples,
            num_devices,
            num_samples // num_devices,
        )

        result_split = pmapped(positions_split, *args, **kwargs)

        # Gather results: [D, N/D, ...] -> [N, ...]
        return _concat_along_leading_axis(result_split)

    return parallel_fn


def get_num_devices() -> int:
    """Return the number of JAX-visible accelerator devices.

    Returns:
      Integer count of available GPUs (or TPUs).  Falls back to 1 if only a
      CPU backend is present.
    """
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform in ('gpu', 'tpu')]
    n = len(gpu_devices) if gpu_devices else 1
    if n > 1:
        logging.info('Multi-GPU mode: %d devices available.', n)
    return n


def log_device_info() -> None:
    """Log available JAX devices and their memory capacities."""
    devices = jax.devices()
    logging.info('JAX device count: %d', len(devices))
    for i, dev in enumerate(devices):
        logging.info(
            '  Device %d: platform=%s, device_kind=%s',
            i,
            dev.platform,
            dev.device_kind,
        )
