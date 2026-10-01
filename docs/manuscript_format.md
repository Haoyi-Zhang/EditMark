# Manuscript format

The active manuscript is `paper/tosem/main.tex`. It uses the ACM Primary Article interface for journal review:

```latex
\documentclass[manuscript,review]{acmart}
\acmJournal{TOSEM}
\bibliographystyle{ACM-Reference-Format}
```

The source does not override margins, text width, section spacing, line spacing, or the ACM reference block. Every figure has an ACM `\Description{}` for accessibility. Each author has a separate `\author`, `\affiliation`, and `\email` block.

Before submission, compile with an up-to-date ACM/TeX distribution. The repository linter checks the stable official interface and rejects layout overrides, missing figure descriptions, inactive manuscript source files, undefined references, missing citations, and superseded manuscript trees.
