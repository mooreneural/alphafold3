"""Tests that GlobalConfig exposes the new GPU acceleration fields correctly.

These tests work without JAX — they only check that the config fields
exist with the right names, types, and default values.
"""

import ast
import pathlib
import unittest


_CONFIG_PATH = (
    pathlib.Path(__file__).parent.parent
    / 'src/alphafold3/model/model_config.py'
)


class TestModelConfigGpuFieldsInSource(unittest.TestCase):
    """Parse model_config.py with ast to verify fields without importing it."""

    def _parse_global_config_body(self):
        src = _CONFIG_PATH.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'GlobalConfig':
                return node
        self.fail('GlobalConfig class not found in model_config.py')

    def _get_annotated_defaults(self, cls_node):
        """Return dict of {name: default_repr} for AnnAssign nodes in the class."""
        result = {}
        for stmt in cls_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                default = ast.unparse(stmt.value) if stmt.value else None
                result[name] = default
        return result

    def test_use_fused_outer_product_scan_field_exists(self):
        cls = self._parse_global_config_body()
        defaults = self._get_annotated_defaults(cls)
        self.assertIn(
            'use_fused_outer_product_scan', defaults,
            'GlobalConfig is missing the use_fused_outer_product_scan field',
        )

    def test_use_fused_outer_product_scan_default_is_false(self):
        cls = self._parse_global_config_body()
        defaults = self._get_annotated_defaults(cls)
        self.assertEqual(
            defaults.get('use_fused_outer_product_scan'), 'False',
            'use_fused_outer_product_scan should default to False (opt-in)',
        )

    def test_log_device_info_field_exists(self):
        cls = self._parse_global_config_body()
        defaults = self._get_annotated_defaults(cls)
        self.assertIn(
            'log_device_info', defaults,
            'GlobalConfig is missing the log_device_info field',
        )

    def test_log_device_info_default_is_false(self):
        cls = self._parse_global_config_body()
        defaults = self._get_annotated_defaults(cls)
        self.assertEqual(
            defaults.get('log_device_info'), 'False',
            'log_device_info should default to False',
        )

    def test_pre_existing_fields_unchanged(self):
        """Existing GlobalConfig fields must still be present and unmodified."""
        cls = self._parse_global_config_body()
        defaults = self._get_annotated_defaults(cls)
        # These must not have been accidentally removed or renamed.
        required_existing = [
            'bfloat16',
            'final_init',
            'pair_attention_chunk_size',
            'pair_transition_shard_spec',
            'flash_attention_implementation',
        ]
        for field in required_existing:
            self.assertIn(
                field, defaults,
                f'Pre-existing GlobalConfig field "{field}" was removed or renamed',
            )


class TestModulesImportsFusedOps(unittest.TestCase):
    """Check that modules.py imports from fused_ops (AST-based, no JAX)."""

    def _parse_modules(self):
        path = (
            pathlib.Path(__file__).parent.parent
            / 'src/alphafold3/model/network/modules.py'
        )
        return ast.parse(path.read_text())

    def test_fused_ops_imported(self):
        """Check for: from alphafold3.model.gpu import fused_ops."""
        tree = self._parse_modules()
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Module is 'alphafold3.model.gpu', name is 'fused_ops'
                if node.module and 'gpu' in node.module:
                    if any(alias.name == 'fused_ops' for alias in node.names):
                        found = True
                        break
        self.assertTrue(
            found,
            'modules.py does not contain "from alphafold3.model.gpu import fused_ops"',
        )

    def test_fused_outer_product_chunk_referenced(self):
        """modules.py should reference the fused function by name."""
        path = (
            pathlib.Path(__file__).parent.parent
            / 'src/alphafold3/model/network/modules.py'
        )
        source = path.read_text()
        self.assertIn(
            'fused_outer_product_chunk',
            source,
            'modules.py does not call fused_ops.fused_outer_product_chunk',
        )

    def test_global_config_flag_used(self):
        """The dispatch must read from global_config, not a local variable."""
        path = (
            pathlib.Path(__file__).parent.parent
            / 'src/alphafold3/model/network/modules.py'
        )
        source = path.read_text()
        self.assertIn(
            'global_config.use_fused_outer_product_scan',
            source,
            'OuterProductMean should check global_config.use_fused_outer_product_scan',
        )


class TestGpuModuleFilesExist(unittest.TestCase):
    """Verify all expected files of the gpu module are present."""

    _BASE = pathlib.Path(__file__).parent.parent / 'src/alphafold3/model/gpu'

    def _expect(self, filename):
        path = self._BASE / filename
        self.assertTrue(
            path.exists(),
            f'Expected file not found: {path}',
        )

    def test_init_exists(self):
        self._expect('__init__.py')

    def test_fused_ops_exists(self):
        self._expect('fused_ops.py')

    def test_parallel_exists(self):
        self._expect('parallel.py')

    def test_xla_cache_exists(self):
        self._expect('xla_cache.py')

    def test_benchmark_exists(self):
        bench = pathlib.Path(__file__).parent.parent / 'benchmarks/gpu_benchmark.py'
        self.assertTrue(bench.exists(), f'Benchmark script not found: {bench}')

    def test_all_gpu_files_are_valid_python(self):
        """Every .py file in the gpu module must parse without SyntaxError."""
        for py_file in self._BASE.glob('*.py'):
            with self.subTest(file=py_file.name):
                try:
                    ast.parse(py_file.read_text())
                except SyntaxError as e:
                    self.fail(f'SyntaxError in {py_file}: {e}')


if __name__ == '__main__':
    unittest.main()
