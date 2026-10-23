"""Optional plotting tests on tiny private copies of retained exports only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FigureConsumerTests(unittest.TestCase):
    def fixture(self, temporary):
        root=Path(temporary)/'artifact'
        (root/'scripts').mkdir(parents=True)
        for name in ('build_paper_assets.py','plot_figures.py'):
            shutil.copyfile(ROOT/'scripts'/name,root/'scripts'/name)
        shutil.copytree(ROOT/'fonts',root/'fonts')
        (root/'analysis').mkdir()
        for name in ('source_inventory.json','method_profiles.json','edit_component_audit.json'):
            shutil.copyfile(ROOT/'analysis'/name,root/'analysis'/name)
        table=root/'results/tables/suite_all_models_methods/method_attack_summary.csv'
        table.parent.mkdir(parents=True)
        shutil.copyfile(ROOT/'results/tables/suite_all_models_methods'/table.name,table)
        generated=root.parent/'paper/tosem/generated';generated.mkdir(parents=True)
        for name in ('release_facts','mutation_rows','source_rows','profile_rows','witness_rows'):
            (generated/(name+'.tex')).write_text('% owned non-empirical test fragment\n')
        return root,table

    def build(self,root,no_dependencies=False):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(root.parent/'mpl-cache'))
        if no_dependencies:env.pop('PYTHONPATH',None)
        return subprocess.run([sys.executable,'-B',*(['-S'] if no_dependencies else []),
            str(root/'scripts/build_paper_assets.py')],cwd=root,env=env,
            capture_output=True,text=True,timeout=45)

    def change_csv(self,path,change):
        with path.open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        change(rows)
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def test_clean_cache_rebuilds_every_actual_pdf_png_consumer(self):
        # The full-packet standalone entry must refresh measured plots too.
        standalone = ROOT.parent / 'paper/tosem/Makefile'
        if standalone.is_file():
            recipe = standalone.read_text(encoding='utf-8')
            self.assertIn('paper: assets', recipe)
            self.assertIn('$(PYTHON) ../../artifact/scripts/build_paper_assets.py', recipe)
            self.assertNotIn('--contract-only', recipe)
        with tempfile.TemporaryDirectory() as td:
            root,_=self.fixture(td)
            result=self.build(root);self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((root/'analysis/manuscript_assets.json').read_text())
            for name in ('evidence_contract','edit_functionality','application_coverage'):
                for suffix in ('pdf','png'):
                    path=root/'figures'/f'{name}.{suffix}'
                    paper=root.parent/'paper/tosem/figures'/path.name
                    self.assertGreater(path.stat().st_size,1000)
                    self.assertEqual(path.read_bytes(),paper.read_bytes())
                    item=next(r for r in manifest['assets'] if r['path']==paper.relative_to(root.parent).as_posix())
                    self.assertEqual(item['sha256'],hashlib.sha256(paper.read_bytes()).hexdigest())

    def test_one_owned_csv_cell_refreshes_both_formats_not_old_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root,table=self.fixture(td)
            # Deliberately stale output caches must be replaced, never consumed.
            paper=root.parent/'paper/tosem/figures';paper.mkdir()
            for name in ('edit_functionality','application_coverage'):
                (paper/(name+'.png')).write_bytes(b'old cached PNG')
            first=self.build(root);self.assertEqual(first.returncode,0,first.stderr)
            before={ext:(paper/('edit_functionality.'+ext)).read_bytes() for ext in ('pdf','png')}
            def update(rows):
                next(r for r in rows if r['method']=='KGW' and r['attack']=='whitespace_normalize')['attacked_test_pass_rate']='0.08'
            self.change_csv(table,update)
            second=self.build(root);self.assertEqual(second.returncode,0,second.stderr)
            for ext in ('pdf','png'):
                self.assertNotEqual(before[ext],(paper/('edit_functionality.'+ext)).read_bytes())

    def test_bad_consumed_rates_fail_even_with_png_cache(self):
        for bad in ('NaN','inf','-0.1','1.1','not-a-number'):
            with self.subTest(value=bad),tempfile.TemporaryDirectory() as td:
                root,table=self.fixture(td)
                paper=root.parent/'paper/tosem/figures';paper.mkdir()
                (paper/'edit_functionality.png').write_bytes(b'cached')
                (paper/'application_coverage.png').write_bytes(b'cached')
                self.change_csv(table,lambda rows: rows[0].update(attacked_test_pass_rate=bad))
                result=self.build(root)
                self.assertNotEqual(result.returncode,0)
                self.assertNotIn('PAPER_ASSETS_OK',result.stdout)

    def test_duplicate_method_edit_cells_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root,table=self.fixture(td)
            self.change_csv(table,lambda rows:rows.append(dict(rows[0])))
            result=self.build(root)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Duplicate retained method/edit cell',result.stderr)

    def test_coverage_must_agree_with_retained_integer_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root,_=self.fixture(td)
            path=root/'analysis/edit_component_audit.json'
            rows=json.loads(path.read_text());rows[0]['application_coverage']=0.5
            path.write_text(json.dumps(rows))
            result=self.build(root)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('coverage disagrees',result.stderr)

    def test_missing_csv_is_failure_not_cached_success(self):
        with tempfile.TemporaryDirectory() as td:
            root,table=self.fixture(td);table.unlink()
            paper=root.parent/'paper/tosem/figures';paper.mkdir()
            for name in ('edit_functionality','application_coverage'):
                (paper/(name+'.png')).write_bytes(b'cached')
            result=self.build(root)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('PAPER_ASSETS_OK',result.stdout)

    def test_missing_optional_dependency_fails_without_cached_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root,_=self.fixture(td)
            paper=root.parent/'paper/tosem/figures';paper.mkdir()
            for name in ('edit_functionality','application_coverage'):
                (paper/(name+'.png')).write_bytes(b'cached')
            result=self.build(root,no_dependencies=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Plotting requires matplotlib',result.stderr)
            self.assertNotIn('PAPER_ASSETS_OK',result.stdout)
