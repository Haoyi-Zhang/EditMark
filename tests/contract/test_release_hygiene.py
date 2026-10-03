"""Checks for a single distributable manuscript tree and resolved authors."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_release_hygiene import verify


class ReleaseHygieneTests(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        paper = root.parent / "paper" / "tosem"
        (paper / "sections").mkdir(parents=True)
        (paper / "main.tex").write_text(r"\input{authors}\input{sections/body}", encoding="utf-8")
        (paper / "authors.tex").write_text(
            "\\author{Haoyi Zhang}\n\\author{Huaijin Ran}\n\\author{Xunzhu Tang}\n",
            encoding="utf-8",
        )
        (paper / "sections" / "body.tex").write_text("body\n", encoding="utf-8")
        (paper / "main.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    def test_clean_tree_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifact"
            root.mkdir()
            self.fixture(root)
            self.assertEqual(verify(root), [])

    def test_superseded_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifact"
            root.mkdir()
            self.fixture(root)
            (root / "history").mkdir()
            (root / "history" / "paper.tex").write_text("old", encoding="utf-8")
            self.assertTrue(any("superseded-version path" in item for item in verify(root)))

    def test_inactive_manuscript_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifact"
            root.mkdir()
            self.fixture(root)
            (root.parent / "paper" / "tosem" / "unused.tex").write_text("unused", encoding="utf-8")
            self.assertTrue(any("inactive manuscript sources" in item for item in verify(root)))
