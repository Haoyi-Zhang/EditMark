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
            value = json.loads((root / 'analysis/manuscript_assets.json').read_text())
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
            manifest = root / 'analysis/manuscript_assets.json'
            retained = {'path': 'paper/tosem/figures/retained.pdf', 'bytes': 10, 'sha256': 'a'*64}
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({'basis': ['supplied-summary.json'], 'assets': [retained]}))
            with patch.object(builder, 'ROOT', root):
                builder.manifest([asset])
            value = json.loads(manifest.read_text())
            self.assertEqual(value['basis'], ['supplied-summary.json'])
            self.assertIn(retained, value['assets'])

    def test_paper_test_count_matches_current_source_methods(self):
        value = json.loads((builder.ROOT / 'analysis/release_facts.json').read_text())
        self.assertEqual(value['contract_test_count'], builder.count_test_methods())

    def test_supplementary_count_matches_current_discovery_selection(self):
        value = json.loads((builder.ROOT / 'analysis/release_facts.json').read_text())
        self.assertEqual(value['supplementary_test_count'], builder.count_test_methods('regressions'))

    def test_both_generated_counts_are_consumed_without_relabeling_history(self):
        root = builder.ROOT
        facts = json.loads((root / 'analysis/release_facts.json').read_text())
        macros = (root.parent / 'paper/tosem/generated/release_facts.tex')
        if macros.is_file():
            text = macros.read_text(encoding='utf-8')
            for command, key in (('ContractTestCount', 'contract_test_count'),
                                 ('SupplementaryTestCount', 'supplementary_test_count')):
                self.assertIn('\\newcommand{\\' + command + '}{' + str(facts[key]) + '}', text)
        readme = (root / 'README.md').read_text(encoding='utf-8')
        self.assertIn(str(facts['contract_test_count']) + ' contract', readme)
        self.assertIn(str(facts['supplementary_test_count']) + ' supplementary', readme)
        self.assertIn('203 contract test methods', readme)


if __name__ == '__main__':
    unittest.main()
