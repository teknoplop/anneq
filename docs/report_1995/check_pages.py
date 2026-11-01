"""Find the page on which each heading lands in a PDF render of report_1995.docx (for the TOC cache).

soffice --headless --convert-to pdf --outdir <dir> docs/report_1995.docx
uv run --no-project python docs/report_1995/check_pages.py <dir>/report_1995.pdf docs/report_1995/pages.json
then rebuild with build_docx.py --pages docs/report_1995/pages.json. (Word users can instead
right-click the contents and choose Update Field.)
"""
import json, pathlib, re, subprocess, sys

pdf, out = sys.argv[1], sys.argv[2]
entries = json.loads((pathlib.Path(__file__).parent / "toc_entries.json").read_text())
pages = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout.split("\f")
res, start = {}, 0
for k, p in enumerate(pages):
    if "NOTATION" in p: start = k + 1
for e in entries:
    for k in range(start, len(pages)):
        lines = [re.sub(r"\s+", " ", l).strip() for l in pages[k].splitlines()]
        if any(l == e or re.fullmatch(r"[\d.]+ " + re.escape(e) + r"( ?\[\d+\])*", l) or l.startswith(e)
               or re.fullmatch(r"[\d.]+ " + re.escape(e) + r".*", l) and len(l) < len(e) + 30 for l in lines):
            m = re.search(r"Page (\S+)\s*$", pages[k].strip())
            res[e] = m.group(1) if m else ""; start = k; break
    else:
        print("not found:", e)
pathlib.Path(out).write_text(json.dumps(res, indent=1))
print(len(res), "of", len(entries))
