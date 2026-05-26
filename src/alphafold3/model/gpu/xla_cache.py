# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Persistent XLA compilation cache for AlphaFold 3.

Overview
--------
AlphaFold 3 JIT-compiles its model graph on first use.  Compilation can take
5–15 minutes on a large GPU because the pairformer and diffusion graphs are
very deep.  JAX supports a disk-based compilation cache (introduced in JAX
0.4.1) that serialises compiled XLA executables to a directory so that
subsequent runs with the same input shapes and GPU architecture skip
recompilation entirely.

This module provides a thin wrapper around JAX's compilation-cache API with
sensible defaults and logging.  Call ``setup_compilation_cache()`` once at
process start, before any JAX operations, to enable the cache.

Typical speed-up
----------------
* First run (cache miss): same as baseline — full compilation.
* Subsequent runs (cache hit): model ready in ~30 seconds instead of 5–15 min.

Example
-------
.. code-block:: python

    from alphafold3.model.gpu import xla_cache

    # Call once, early in your script / before loading model params.
    xla_cache.setup_compilation_cache(cache_dir='/tmp/af3_xla_cache')
"""

import os
import pathlib

from absl import logging
import jax


# Default cache directory — respects $AF3_XLA_CACHE_DIR if set.
_DEFAULT_CACHE_DIR = os.environ.get(
    'AF3_XLA_CACHE_DIR',
    str(pathlib.Path.home() / '.cache' / 'alphafold3' / 'xla_compilation'),
)


def setup_compilation_cache(
    cache_dir: str | None = None,
    min_compile_time_secs: float = 5.0,
) -> str:
    """Enable JAX's persistent XLA compilation cache.

    Must be called **before** any JAX computations (ideally right after imports).

    The cache is keyed by:
    * The XLA computation graph (i.e. model architecture + input shapes).
    * The CUDA/ROCm toolkit version.
    * The GPU architecture (compute capability).

    Changing any of these automatically produces a new cache entry, so stale
    caches are never silently used.

    Args:
      cache_dir: Directory in which to store compiled executables.  Defaults to
        ``~/.cache/alphafold3/xla_compilation`` or the value of the environment
        variable ``AF3_XLA_CACHE_DIR``.
      min_compile_time_secs: Only cache compilations that took longer than this
        many seconds, to avoid cluttering the cache with cheap operations.
        Defaults to 5.0 s.

    Returns:
      The resolved cache directory path (after creation).

    Raises:
      RuntimeError: If JAX does not expose a compilation-cache API (JAX < 0.4.1).
    """
    resolved = pathlib.Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(resolved)

    if not hasattr(jax, 'config'):
        raise RuntimeError(
            'jax.config not available — cannot configure XLA compilation cache.'
        )

    try:
        jax.config.update('jax_compilation_cache_dir', cache_dir_str)
        jax.config.update(
            'jax_persistent_cache_min_compile_time_secs', min_compile_time_secs
        )
        logging.info(
            'XLA compilation cache enabled: %s (min_compile_time=%.1fs)',
            cache_dir_str,
            min_compile_time_secs,
        )
    except AttributeError as exc:
        raise RuntimeError(
            'JAX version does not support the persistent compilation cache API.'
            ' Upgrade to JAX >= 0.4.1 to use this feature.'
        ) from exc

    return cache_dir_str


def clear_compilation_cache(cache_dir: str | None = None) -> None:
    """Delete all cached executables in ``cache_dir``.

    Useful after upgrading JAX, CUDA, or the model architecture.

    Args:
      cache_dir: Directory to clear.  Defaults to the same path used by
        ``setup_compilation_cache()``.
    """
    import shutil  # pylint: disable=g-import-not-at-top

    resolved = pathlib.Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser().resolve()
    if resolved.exists():
        shutil.rmtree(resolved)
        logging.info('XLA compilation cache cleared: %s', resolved)
    else:
        logging.info('XLA compilation cache directory not found: %s', resolved)


def get_cache_size_bytes(cache_dir: str | None = None) -> int:
    """Return the total size of the compilation cache in bytes.

    Args:
      cache_dir: Directory to measure.  Defaults to the standard location.

    Returns:
      Total size in bytes, or 0 if the directory does not exist.
    """
    resolved = pathlib.Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser().resolve()
    if not resolved.exists():
        return 0
    return sum(f.stat().st_size for f in resolved.rglob('*') if f.is_file())


def log_cache_stats(cache_dir: str | None = None) -> None:
    """Log the number of cache entries and their total disk usage.

    Args:
      cache_dir: Directory to inspect.  Defaults to the standard location.
    """
    resolved = pathlib.Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser().resolve()
    if not resolved.exists():
        logging.info('XLA cache directory does not exist: %s', resolved)
        return

    files = list(resolved.rglob('*'))
    cache_files = [f for f in files if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in cache_files)
    logging.info(
        'XLA cache stats: %d entries, %.1f MB on disk (%s)',
        len(cache_files),
        total_bytes / (1024 ** 2),
        resolved,
    )
