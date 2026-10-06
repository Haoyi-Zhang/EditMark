"""Synthetic asset paths and the current source-derived test count."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import build_contract_artifacts as builder


class AssetManifestTests(unittest.TestCase):
    def test_sibling_paper_asset_uses_project_relative_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'artifact'
            root.mkdir()
            asset = root.parent / 'paper/tosem/generated/row.tex'
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b'synthetic row\n')
            with patch.object(builder, 'ROOT', root):
                builder.manifest([asset])
            value = json.loads((asset.parent / 'asset_manifest.json').read_text())
            self.assertEqual(value['assets'], [{
                'path': 'paper/tosem/generated/row.tex', 'bytes': 14,
                'sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
            }])

    def test_partial_asset_refresh_retains_other_entries_and_basis(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'artifact'
            root.mkdir()
            asset = root.parent / 'paper/tosem/generated/row.tex'
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b'synthetic row\n')
            manifest = asset.parent / 'asset_manifest.json'
            retained = {'path': 'paper/tosem/figures/retained.pdf', 'bytes': 10, 'sha256': 'a'*64}
            manifest.write_text(json.dumps({'basis': ['supplied-summary.json'], 'assets': [retained]}))
            with patch.object(builder, 'ROOT', root):
                builder.manifest([asset])
            value = json.loads(manifest.read_text())
            self.assertEqual(value['basis'], ['supplied-summary.json'])
            self.assertIn(retained, value['assets'])

    def test_paper_test_count_matches_current_source_methods(self):
        value = json.loads((builder.ROOT / 'analysis/release_facts.json').read_text())
        self.assertEqual(value['contract_test_count'], builder.count_test_methods())


if __name__ == '__main__':
    unittest.main()
