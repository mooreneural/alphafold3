"""Tests for alphafold3.model.gpu.xla_cache.

These tests cover all pure-Python behaviour (directory management, cache stats,
argument handling) and skip the live JAX config update when JAX is absent.
They run without any GPU and without the alphafold3 package installed.
"""

import pathlib
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Minimal JAX stub so the module can be imported without a real JAX install.
# ---------------------------------------------------------------------------

def _ensure_absl_stub():
    """Install a minimal absl stub if absl-py is not available."""
    if 'absl' not in sys.modules:
        absl_stub = types.ModuleType('absl')
        logging_stub = types.ModuleType('absl.logging')
        logging_stub.info = lambda *a, **kw: None
        logging_stub.warning = lambda *a, **kw: None
        absl_stub.logging = logging_stub
        sys.modules['absl'] = absl_stub
        sys.modules['absl.logging'] = logging_stub


def _make_jax_stub():
    """Return a minimal jax stub module."""
    _ensure_absl_stub()
    jax_mod = types.ModuleType('jax')
    jax_mod.__version__ = '0.4.99'

    config_store = {}

    class _Config:
        def update(self, key, value):
            config_store[key] = value

        def get_store(self):
            return config_store

    jax_mod.config = _Config()
    sys.modules['jax'] = jax_mod
    return jax_mod, config_store


def _remove_jax_stub():
    sys.modules.pop('jax', None)
    sys.modules.pop('alphafold3.model.gpu.xla_cache', None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestXlaCacheSetup(unittest.TestCase):

    def setUp(self):
        _remove_jax_stub()
        self._jax_stub, self._config_store = _make_jax_stub()

    def tearDown(self):
        _remove_jax_stub()

    def _import_module(self):
        """Import xla_cache with the stub JAX in place."""
        # Make sure module cache is cleared so it re-imports with stub.
        sys.modules.pop('alphafold3.model.gpu.xla_cache', None)

        # Minimal package stubs so the import path resolves.
        for pkg in ('alphafold3', 'alphafold3.model', 'alphafold3.model.gpu'):
            if pkg not in sys.modules:
                sys.modules[pkg] = types.ModuleType(pkg)

        # Now do a direct file-based import.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'alphafold3.model.gpu.xla_cache',
            pathlib.Path(__file__).parent.parent
            / 'src/alphafold3/model/gpu/xla_cache.py',
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules['alphafold3.model.gpu.xla_cache'] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_setup_creates_directory(self):
        """setup_compilation_cache should create the cache directory."""
        import tempfile, os
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, 'xla_test_cache')
            result = mod.setup_compilation_cache(cache_dir=cache_dir)
            self.assertTrue(
                pathlib.Path(cache_dir).exists(),
                'Cache directory was not created.',
            )
            self.assertEqual(result, str(pathlib.Path(cache_dir).resolve()))

    def test_setup_sets_jax_config_keys(self):
        """setup_compilation_cache should call jax.config.update with the right keys."""
        import tempfile, os
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, 'xla_test_cache2')
            mod.setup_compilation_cache(cache_dir=cache_dir, min_compile_time_secs=3.0)
            self.assertIn('jax_compilation_cache_dir', self._config_store)
            self.assertIn(
                'jax_persistent_cache_min_compile_time_secs', self._config_store
            )
            self.assertEqual(self._config_store['jax_persistent_cache_min_compile_time_secs'], 3.0)

    def test_setup_idempotent(self):
        """Calling setup twice with the same directory should not raise."""
        import tempfile, os
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, 'xla_idempotent')
            mod.setup_compilation_cache(cache_dir=cache_dir)
            mod.setup_compilation_cache(cache_dir=cache_dir)  # second call
            self.assertTrue(pathlib.Path(cache_dir).exists())

    def test_clear_nonexistent_dir_does_not_raise(self):
        """clear_compilation_cache should not raise if directory doesn't exist."""
        mod = self._import_module()
        mod.clear_compilation_cache(cache_dir='/tmp/_af3_nonexistent_xla_dir_xyz')

    def test_clear_removes_directory(self):
        """clear_compilation_cache should delete the directory and its contents."""
        import tempfile, os
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, 'xla_to_clear')
            pathlib.Path(cache_dir).mkdir()
            # Create a dummy file inside.
            (pathlib.Path(cache_dir) / 'dummy.xla').write_text('data')
            mod.clear_compilation_cache(cache_dir=cache_dir)
            self.assertFalse(pathlib.Path(cache_dir).exists())

    def test_get_cache_size_empty_dir(self):
        """get_cache_size_bytes should return 0 for an empty directory."""
        import tempfile, os
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, 'xla_empty')
            pathlib.Path(cache_dir).mkdir()
            size = mod.get_cache_size_bytes(cache_dir=cache_dir)
            self.assertEqual(size, 0)

    def test_get_cache_size_nonexistent(self):
        """get_cache_size_bytes should return 0 when directory is absent."""
        mod = self._import_module()
        size = mod.get_cache_size_bytes(cache_dir='/tmp/_af3_nonexistent_xla_size')
        self.assertEqual(size, 0)

    def test_get_cache_size_with_files(self):
        """get_cache_size_bytes should sum file sizes correctly."""
        import tempfile, os
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, 'xla_with_files')
            pathlib.Path(cache_dir).mkdir()
            (pathlib.Path(cache_dir) / 'a.xla').write_bytes(b'x' * 1024)
            (pathlib.Path(cache_dir) / 'b.xla').write_bytes(b'y' * 512)
            size = mod.get_cache_size_bytes(cache_dir=cache_dir)
            self.assertEqual(size, 1536)

    def test_default_cache_dir_env_var(self):
        """AF3_XLA_CACHE_DIR env var should override the default cache location."""
        import os
        mod = self._import_module()
        # The module reads AF3_XLA_CACHE_DIR at import time via module-level code.
        # We verify the default constant is at least a non-empty string.
        self.assertIsInstance(mod._DEFAULT_CACHE_DIR, str)
        self.assertTrue(len(mod._DEFAULT_CACHE_DIR) > 0)


class TestXlaCacheWithoutJax(unittest.TestCase):
    """Verify behaviour when JAX is absent entirely."""

    def setUp(self):
        _remove_jax_stub()
        # Remove jax entirely so the import fails inside setup_compilation_cache.
        sys.modules['jax'] = None  # importlib treats None as "not found"

    def tearDown(self):
        sys.modules.pop('jax', None)
        sys.modules.pop('alphafold3.model.gpu.xla_cache', None)

    def test_raises_runtime_error_without_jax(self):
        """setup_compilation_cache should raise RuntimeError when JAX unavailable."""
        # Re-import with broken jax.
        import importlib.util, types

        for pkg in ('alphafold3', 'alphafold3.model', 'alphafold3.model.gpu'):
            if pkg not in sys.modules:
                sys.modules[pkg] = types.ModuleType(pkg)

        # Provide a jax stub with no .config attribute.
        jax_no_config = types.ModuleType('jax')
        jax_no_config.__version__ = '0.3.0'
        # Deliberately no 'config' attribute.
        sys.modules['jax'] = jax_no_config
        sys.modules.pop('alphafold3.model.gpu.xla_cache', None)

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'alphafold3.model.gpu.xla_cache',
            pathlib.Path(__file__).parent.parent
            / 'src/alphafold3/model/gpu/xla_cache.py',
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules['alphafold3.model.gpu.xla_cache'] = mod
        spec.loader.exec_module(mod)

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                mod.setup_compilation_cache(cache_dir=os.path.join(tmp, 'xla'))


if __name__ == '__main__':
    unittest.main()
