"""Tests for alphafold3.model.gpu.parallel.

Pure-Python / NumPy tests covering:
  - round_samples_to_devices()
  - _split_along_leading_axis() / _concat_along_leading_axis()

JAX-dependent tests (make_parallel_diffusion_fn, get_num_devices) are
marked with @pytest.mark.skipif and require JAX + at least one GPU.
"""

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Helper: load the parallel module with a minimal numpy-backed JAX stub.
# ---------------------------------------------------------------------------

def _load_parallel_module():
    """Import parallel.py with a numpy-backed JAX stub."""
    # Stub absl.logging if not installed.
    if 'absl' not in sys.modules:
        absl_stub = types.ModuleType('absl')
        logging_stub = types.ModuleType('absl.logging')
        logging_stub.info = lambda *a, **kw: None
        logging_stub.warning = lambda *a, **kw: None
        absl_stub.logging = logging_stub
        sys.modules['absl'] = absl_stub
        sys.modules['absl.logging'] = logging_stub

    # Clean up any previous import.
    for key in list(sys.modules.keys()):
        if 'alphafold3' in key or (key == 'jax' and not isinstance(sys.modules[key], types.ModuleType)):
            pass  # leave real jax if present

    # Only stub if JAX is absent.
    real_jax = sys.modules.get('jax')
    if real_jax is None:
        jax_stub = types.ModuleType('jax')

        # Minimal tree_util stub.
        tree_util = types.ModuleType('jax.tree_util')

        def tree_map(fn, *trees):
            if len(trees) == 1:
                return fn(trees[0]) if not isinstance(trees[0], (list, tuple, dict)) else type(trees[0])(fn(x) for x in trees[0])
            return fn(*[t for t in trees])

        tree_util.tree_map = tree_map
        jax_stub.tree_util = tree_util

        # Stub devices().
        DeviceStub = types.SimpleNamespace(platform='gpu', device_kind='Tesla V100', id=0)
        jax_stub.devices = lambda: [DeviceStub]
        jax_stub.device_count = lambda: 1
        jax_stub.pmap = lambda fn, axis_name=None: fn  # identity for tests

        sys.modules['jax'] = jax_stub
        sys.modules['jax.tree_util'] = tree_util
        sys.modules['jax.numpy'] = types.ModuleType('jax.numpy')

    for pkg in ('alphafold3', 'alphafold3.model', 'alphafold3.model.gpu'):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    sys.modules.pop('alphafold3.model.gpu.parallel', None)

    spec = importlib.util.spec_from_file_location(
        'alphafold3.model.gpu.parallel',
        pathlib.Path(__file__).parent.parent / 'src/alphafold3/model/gpu/parallel.py',
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['alphafold3.model.gpu.parallel'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests: round_samples_to_devices
# ---------------------------------------------------------------------------

class TestRoundSamplesToDevices(unittest.TestCase):

    def setUp(self):
        self.mod = _load_parallel_module()

    def test_already_divisible(self):
        self.assertEqual(self.mod.round_samples_to_devices(8, 4), 8)
        self.assertEqual(self.mod.round_samples_to_devices(1, 1), 1)
        self.assertEqual(self.mod.round_samples_to_devices(20, 5), 20)

    def test_rounds_up(self):
        # 5 samples, 4 devices -> needs 8 (2 per device)
        self.assertEqual(self.mod.round_samples_to_devices(5, 4), 8)
        # 1 sample, 3 devices -> needs 3
        self.assertEqual(self.mod.round_samples_to_devices(1, 3), 3)
        # 7 samples, 4 devices -> needs 8
        self.assertEqual(self.mod.round_samples_to_devices(7, 4), 8)

    def test_single_device_unchanged(self):
        for n in (1, 5, 13, 100):
            self.assertEqual(self.mod.round_samples_to_devices(n, 1), n)

    def test_result_is_divisible(self):
        for n in range(1, 17):
            for d in (1, 2, 3, 4, 8):
                result = self.mod.round_samples_to_devices(n, d)
                self.assertEqual(
                    result % d, 0,
                    f'round_samples_to_devices({n}, {d}) = {result} not divisible by {d}',
                )
                self.assertGreaterEqual(result, n)


# ---------------------------------------------------------------------------
# Tests: _split_along_leading_axis / _concat_along_leading_axis
# ---------------------------------------------------------------------------

class TestSplitConcatAxis(unittest.TestCase):

    def setUp(self):
        self.mod = _load_parallel_module()

    def _split(self, arr, n):
        """Call _split_along_leading_axis on a single numpy array."""
        # The function operates on JAX pytrees but our stub tree_map works
        # on plain arrays too.
        return self.mod._split_along_leading_axis(arr, n)

    def _concat(self, arr):
        return self.mod._concat_along_leading_axis(arr)

    def test_split_shape_2d(self):
        arr = np.zeros((8, 16))
        result = self._split(arr, 4)
        self.assertEqual(result.shape, (4, 2, 16))

    def test_split_shape_3d(self):
        arr = np.zeros((12, 5, 3))
        result = self._split(arr, 3)
        self.assertEqual(result.shape, (3, 4, 5, 3))

    def test_split_raises_on_non_divisible(self):
        arr = np.zeros((7, 3))
        with self.assertRaises(ValueError):
            self._split(arr, 4)

    def test_concat_inverts_split_2d(self):
        original = np.arange(24).reshape(8, 3)
        split = self._split(original, 4)
        restored = self._concat(split)
        np.testing.assert_array_equal(original, restored)

    def test_concat_inverts_split_3d(self):
        original = np.arange(60).reshape(12, 5, 1)
        split = self._split(original, 4)
        restored = self._concat(split)
        np.testing.assert_array_equal(original, restored)

    def test_split_single_device(self):
        arr = np.ones((5, 7))
        result = self._split(arr, 1)
        self.assertEqual(result.shape, (1, 5, 7))

    def test_roundtrip_preserves_values(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((16, 32))
        split = self._split(arr, 8)
        restored = self._concat(split)
        np.testing.assert_allclose(arr, restored)


# ---------------------------------------------------------------------------
# JAX-dependent tests (skipped when JAX is unavailable)
# ---------------------------------------------------------------------------

try:
    import jax as _real_jax
    _JAX_AVAILABLE = hasattr(_real_jax, 'lax')  # real JAX has lax
except ImportError:
    _JAX_AVAILABLE = False

import unittest


@unittest.skipUnless(_JAX_AVAILABLE, 'JAX not installed')
class TestGetNumDevices(unittest.TestCase):

    def test_returns_positive_int(self):
        from alphafold3.model.gpu import parallel
        n = parallel.get_num_devices()
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)


if __name__ == '__main__':
    unittest.main()
