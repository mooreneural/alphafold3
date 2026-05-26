# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Global config for the model."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from alphafold3.common import base_config
import tokamax

_Shape2DType: TypeAlias = tuple[int | None, int | None]


class GlobalConfig(base_config.BaseConfig):
  """Global configuration for the AlphaFold3 model."""

  bfloat16: Literal['all', 'none', 'intermediate'] = 'all'
  final_init: Literal['zeros', 'linear'] = 'zeros'
  pair_attention_chunk_size: Sequence[_Shape2DType] = ((1536, 128), (None, 32))
  pair_transition_shard_spec: Sequence[_Shape2DType] = (
      (2048, None),
      (None, 1024),
  )
  # Note: flash_attention_implementation = 'xla' means no flash attention.
  flash_attention_implementation: tokamax.DotProductAttentionImplementation = (
      'triton'
  )

  # ---------------------------------------------------------------------------
  # GPU acceleration options (alphafold3.model.gpu)
  # ---------------------------------------------------------------------------

  # When True, enables the fused channel-scan OuterProductMean that avoids
  # materialising the large [N, C_outer, C_outer, chunk] intermediate tensor.
  # Reduces peak memory ~6x for that operation at a small scan-loop overhead.
  # Recommended for sequences > 512 residues on GPU with limited HBM.
  # See alphafold3.model.gpu.fused_ops for derivation and memory analysis.
  use_fused_outer_product_scan: bool = False

  # When True, emit an info-level log of available JAX devices at model init.
  log_device_info: bool = False
