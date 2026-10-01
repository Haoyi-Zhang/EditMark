"""Regression checks for source-graph and citation extraction."""
import tempfile
import unittest
from pathlib import Path
from scripts.verify_manuscript import citation_keys, format_errors, source_graph

class ManuscriptCitationTests(unittest.TestCase):
    def test_style_is_not_a_citation(self):
        self.assertEqual(citation_keys(r"\citestyle{acmnumeric}"), set())
    def test_multiple_keys(self):
        self.assertEqual(citation_keys(r"\cite{kgw, sweet,ewd}"), {"kgw","sweet","ewd"})
    def test_optional_arguments(self):
        self.assertEqual(citation_keys(r"\citep[see][p. 4]{stone}"), {"stone"})
    def test_starred_command(self):
        self.assertEqual(citation_keys(r"\citet*{dodge} \bibliography{references}"), {"dodge"})

class SourceGraphTests(unittest.TestCase):
    def check_graph(self, files):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,text in files.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            return source_graph(root)
    def test_recursive_generated_fragment(self):
        sources,errors=self.check_graph({'main.tex':r'\input{section}', 'section.tex':r'\input{generated/rows}', 'generated/rows.tex':'row'})
        self.assertFalse(errors);self.assertEqual(len(sources),3)
    def test_unreferenced_draft_ignored(self):
        sources,errors=self.check_graph({'main.tex':'current','obsolete.tex':r'\cite{wrong}'})
        self.assertFalse(errors);self.assertNotIn('obsolete.tex',[p.name for p in sources])
    def test_cycle_fails(self):
        _,errors=self.check_graph({'main.tex':r'\input{main}'})
        self.assertIn('input cycle',errors[0])
    def test_missing_fails(self):
        _,errors=self.check_graph({'main.tex':r'\input{missing}'})
        self.assertIn('missing input',errors[0])
    def test_escape_fails(self):
        _,errors=self.check_graph({'main.tex':r'\input{../outside}'})
        self.assertIn('escapes',errors[0])
    def test_comment_input_ignored(self):
        _,errors=self.check_graph({'main.tex':'% '+r'\input{missing}'+'\nbody'})
        self.assertFalse(errors)


class FormatTests(unittest.TestCase):
    BASE = r'''\documentclass[review,manuscript]{acmart}
\settopmatter{printacmref=true}
\acmJournal{TOSEM}
\begin{figure}x\Description{accessible}\end{figure}'''

    def test_official_interface_accepted(self):
        self.assertEqual(format_errors(self.BASE), [])

    def test_layout_override_rejected(self):
        errors = format_errors(self.BASE + r'\usepackage{geometry}')
        self.assertTrue(any('geometry package' in item for item in errors))

    def test_missing_description_rejected(self):
        text = self.BASE.replace(r'x\Description{accessible}', 'x')
        self.assertTrue(any('missing an ACM Description' in item for item in format_errors(text)))
