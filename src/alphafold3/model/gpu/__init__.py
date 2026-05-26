# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""GPU acceleration utilities for AlphaFold 3.

Modules
-------
fused_ops
    Memory-efficient fused operations (OuterProductMean scan fusion).
parallel
    Multi-GPU inference via jax.pmap for diffusion sample parallelism.
xla_cache
    Persistent XLA compilation cache setup and management.
"""

from alphafold3.model.gpu import fused_ops
from alphafold3.model.gpu import parallel
from alphafold3.model.gpu import xla_cache
