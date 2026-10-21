# Manuscript format

The active manuscript is `paper/tosem/main.tex`. It uses the ACM Primary Article interface for journal review:

```latex
\documentclass[manuscript,review]{acmart}
\acmJournal{TOSEM}
\bibliographystyle{ACM-Reference-Format}
```

The source does not override margins, text width, section spacing, or line spacing.
This nonanonymous review copy retains the approved author blocks. It has no assigned
DOI, ISBN, volume, issue, article number or publication date and uses
`\setcopyright{none}` with `printacmref=false`; this avoids presenting the class's
default volume/issue/date as an actual publication citation. The linter permits
that setting only for `manuscript,review` with explicitly empty DOI/ISBN and no
assigned publication fields. A production-metadata copy must retain the ACM
reference block. Every figure has an ACM `\Description{}` for accessibility. Each
author has a separate `\author`, `\affiliation`, and `\email` block.

Before submission, compile with an up-to-date ACM/TeX distribution. The repository linter checks the stable official interface and rejects layout overrides, missing figure descriptions, inactive manuscript source files, undefined references, missing citations, and superseded manuscript trees.
