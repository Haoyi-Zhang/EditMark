# Figure reproduction

From the artifact directory, run:

```sh
python scripts/plot_figures.py
```

Plotting requires Matplotlib (with its NumPy dependency); it does not require
TeX or the scientific experiment dependencies. The two dot plots read the retained method/edit CSV and applicability audit JSON. They keep the reported descriptive denominators and do not infer a paired effect.

The native outputs are the PDFs in this directory. When the sibling paper
directory is present, the producer copies identical PDF bytes into its figures
directory. Fixed PDF boxes retain the prior layout; no tight-crop rescaling or
rasterized data/text is used. The manuscript captions and references stay in TeX.

Labels use the bundled unmodified Linux Libertine 5.3.0 regular and italic
TrueType fonts, matching the ACM paper's Libertine family. They are redistributed
under the SIL Open Font License in `../fonts/OFL-1.1.txt`. The upstream archive is
[Linux Libertine 5.3.0](https://sourceforge.net/projects/linuxlibertine/files/linuxlibertine/5.3.0/LinLibertineTTF_5.3.0_2012_07_02.tgz/download);
the downloaded archive matched its published SHA-256
`24a593a949808d976850131a953c0c0d7a72299531dfbb348191964cc038d75d`.
The source font files are unchanged.

The bundled fonts provide portable reproduction without system font paths.
`--font-dir DIRECTORY` can supply the same two TTF filenames. If the bundled
files are unavailable, an installed Linux Libertine regular/italic pair is used;
absence of that family raises an error instead of silently substituting another
font. PDFs embed subsets with `pdf.fonttype=42`, without TeX rendering or
unembedded PDF core fonts.
