# Word rebuild of the 1995 report (issue #23)

`docs/report_1995.docx` is rebuilt from the proofread OCR text (`docs/report_text.md`), the listing in
`src/borland/` and new figures. The scan (`docs/report_scan.pdf`) was the layout reference.

## Rebuild

```sh
uv run --no-project --with numpy --with matplotlib python docs/report_1995/plots/make_plots.py [--data DIR] [--impl clang]
uv run --no-project python docs/report_1995/diagrams/make_diagrams.py      # needs draw.io (brew install --cask drawio)
uv run --no-project --with python-docx --with pillow python docs/report_1995/build_docx.py
soffice --headless --convert-to pdf --outdir /tmp/r docs/report_1995.docx
uv run --no-project python docs/report_1995/check_pages.py /tmp/r/report_1995.pdf docs/report_1995/pages.json
uv run --no-project --with python-docx --with pillow python docs/report_1995/build_docx.py --pages docs/report_1995/pages.json
```

For the corrected version, point `make_plots.py --data` at the fixed run's CSV directory (and `--impl`
at the fixed implementation, e.g. `clang_fix`), then rebuild. Figure footnotes are in `FIG` in
`build_docx.py`; several say "unfixed listing" and will need rewording.

The skill's docx-js route was not used (no Node.js on this machine); the document is written with
python-docx plus raw OOXML for fields, OMML equations, footnotes and custom properties. It passes the
docx skill's schema validator.

## Fields

- Custom properties `ReportTitle`, `ReportType`, `ReportAuthor` (= "Author"), `Supervisor`
  (= "Supervisor"), `Degree`, `CourseCode` (IIIE), `ReportDayMonth` (28 April), `Year` (1995), shown
  through DOCPROPERTY fields on the title page and in the header/footer. To fill in the real names:
  File > Properties > Custom, then Ctrl+A, F9 (and update the header/footer).
- Page numbers are PAGE fields: roman for the front matter (ii...), arabic from chapter 1.
- Figure and table numbers: literal chapter + `SEQ Figure`/`SEQ Table` (`\r 1` at each chapter's
  first figure). Every "figure X.Y"/"table X.Y" in the text is a `REF` to the caption bookmark.
  (STYLEREF chapter numbers were tried first, but LibreOffice renders them as the heading text.)
- Contents is a `TOC \o "1-2"` field; its cached page numbers come from the LibreOffice render.
- core.xml creator/lastModifiedBy are "Author"; app.xml has no company/manager; no comments.

## Page-by-page check against the scan (LibreOffice render, 83 pages vs 82)

Checked: title page, summary, contents, notation, chapters 1-10, references, appendix A, appendix B.

- All headings and section numbers match the scan's contents page. Chapter pages shift by up to two
  pages (different fonts, larger figures): e.g. chapter 3 starts on page 7 (scan 6), chapter 9 on 31
  (scan 29). The TOC shows the new page numbers.
- All 34 figure captions (1.1-9.3) and both table captions (3.1, 4.1) are present, numbered as in the
  scan, captions below figures and above tables as in the scan.
- Tables 3.1 and 4.1 rebuilt as Word tables. Notation page as a borderless two-column table.
- Equations (2-1)...(8.2) and (A1)...(A4) are OMML with their numbers right-aligned; unnumbered
  equations as in the scan. Multi-line derivations use equation arrays (rows centred, not aligned on
  "=": LibreOffice shows OMML alignment marks literally).
- Footnote [a] (chapter 1, Minsky and Papert) is a real footnote.
- Appendix B is set from `src/borland/` (all 14 files, in the scan's order; RB.C, missing from the
  scan's OCR, is added at the end). The listing takes fewer pages than the scan's.

### Text changes from report_text.md

Verbatim otherwise, including 1995 spellings ("Reciever", "extremley", "Procressing") and slips
(e.g. H[z] = ... + 0.3482z^-1 printed twice, "Figure 7.1 shows the convergence plot").
- The three "[???] (handwritten correction: N)" references are set as [N], as corrected by hand in the scan.
- OCR noise `network!2"!` in chapter 10 is set as `network[21]` (superscript citation in the scan).

### Figures

Diagrams (draw.io sources in `diagrams/`, 300 dpi PNG): 2.1, 2.4, 3.1, 3.2, 4.1-4.5, 5.1, 7.1, 8.1, 8.2.
Sketches 1.1 and 2.2 are redrawn with matplotlib.

Plots (`plots/`, 300 dpi PNG) use the unmodified C build (`clang`, same numbers as src/anneq_orig.py):
- 2.3: from eq. (2-4) with NumPy's generator, 640 bins over -1..1 (matches the ideal peak of ~103).
- 4.6, 4.7, 5.2, 5.6: Bayes decision function computed for 16 dB SNR.
- 5.3, 5.4, 5.5, 5.7, 7.2, 7.3, 8.3, 9.2, 9.3: from results/rebuild. Zero-BER points cannot be shown on
  the log axis and are left out (footnoted).
- 5.4: the eta = 0.0001 run of the unmodified LMS.C hangs; that curve comes from the NumPy port, src/anneq_numpy.py (footnoted).
- 6.2: BPBC only runs with the BP-DEBUG fix (`clang_fix`), learning rate/momentum assumed 0.1/0.5,
  21 dB. Channel 2 stays on its plateau for all 1000 epochs in this run (footnoted).
- 6.3, 6.4: not reproduced (the programs don't save the trained MLP's boundary/surface); a placeholder
  box and caption stand in their place (footnoted).
- 7.3, 8.3: the listed code makes far fewer errors than the original figures (issue #19); plotted
  as the code gives and footnoted.
- 9.2, 9.3: ADALINE with eta = 0.008, because the printed 0.08 diverges and the run hangs. 9.2
  curves at 20 dB SNR (the original doesn't say).
- `plots/substitutions.txt` lists every point taken from a fallback implementation.

## Redaction

Author and supervisor are placeholders in the custom properties and core.xml. The unzipped docx was
grepped for the author's surname, user names and home paths (clean). The supervisor's surname is not
in any source text (the OCR says "[name omitted]"), so it can't have entered the document.

## Corrected edition (`docs/report_1995_corrected.docx`)

The 1995 report re-run with the simulator bugs #10-#18 fixed (branch `pyfix`, `results/RESULTS_FIXED.md`).

```sh
uv run --no-project --with numpy --with matplotlib python docs/report_1995/plots/make_plots.py \
    --data results/fixed --impl pyfix --fallback "" --out docs/report_1995/plots_corrected
uv run --no-project --with python-docx --with pillow --with pyyaml python docs/report_1995/build_docx.py \
    --edition corrected --pages docs/report_1995/pages_corrected.json
```

- Plots: `pyfix` (every fix, random training bits) because RESULTS_FIXED.md quotes those numbers and
  chose the RNN learning rates with them. `make_plots.py` gained `--out` creation, a preference for the
  `<impl>_all` curves in Fig 5.4 (no trial passes at eta 0.1 / 0.0001, so the all-trial average is
  plotted), and lower y-limits for Figs 7.2 / 9.2 when a curve goes below 0.01. The 1995 plots
  regenerate byte-identical and the 1995 docx's XML is unchanged.
- Replaced figures (all data plots): 5.3, 5.4, 5.5, 5.7, 6.2, 7.2, 7.3, 8.3, 9.2, 9.3, footnoted
  "Corrected: ..." (`FIG_CORRECTED`). Analytic figures and diagrams are unchanged.
- Text: `corrections.yaml`, one entry per change with the RESULTS_FIXED.md item; the build asserts each
  applies exactly once. Summary (1), §3.2 retry cap (8), §7.3 sensitivity and eta 0.005 -> 0.12 (2),
  §9 RNN eta 0.003 -> 0.08 (4), §9 Fig 9.2/9.3 statements (5, 6), §10 RNN paragraph and "no advantage" (7).
- A "Note on this edition" page follows the title page. Changed paragraphs and the captions of redrawn
  figures have a grey left-margin bar.
- Left as in 1995 on purpose: §9's ADALINE "learning rate of 0.08" (a misprint, not a results change;
  the figure footnotes say 0.008 was used) and §3.3's PN-sequence description (RESULTS_FIXED.md says
  it stands; the edition note says the rerun used random training bits, #16).
- Checks: LibreOffice render 84 pages (1995 edition 83: the note page); body pages and TOC unchanged
  (`pages_corrected.json` = `pages.json`). Looked at pp. i-ii, 8, 26-27, 31-33: bars, figures, footnotes
  and the new text render correctly. Unzipped XML has creator "Author", supervisor "Supervisor",
  no emails, names or home paths.
