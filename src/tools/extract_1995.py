"""Extract the Appendix B listings from docs/report_text.md into src/borland/ verbatim.

Only page markers (<!-- pNNN -->) and the appendix heading are dropped.
Run: uv run --no-project python src/tools/extract_1995.py
"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
lines = (root / "docs/report_text.md").read_text().splitlines()
start = next(i for i, l in enumerate(lines) if l.strip() == "Appendix B - Program listings")
starts = []  # (line index, name)
for i in range(start, len(lines)):
    m = re.match(r"\s*FILENAME:\s*(\S+)", lines[i])
    if m:
        j = i - 1
        while lines[j].strip() != "/*":
            j -= 1
        starts.append((j, m.group(1)))
# RB main program: no header block, starts after the p062 marker
p062 = next(i for i, l in enumerate(lines) if l.strip() == "<!-- p062 -->")
starts.append((p062 + 1, "RB.C"))
starts.sort()
out = root / "src/borland"
out.mkdir(parents=True, exist_ok=True)
for k, (s, name) in enumerate(starts):
    e = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
    body = [l for l in lines[s:e] if not re.match(r"\s*<!-- p\d+ -->\s*$", l)]
    while body and not body[-1].strip():
        body.pop()
    (out / name).write_text("\n".join(body) + "\n")
    print(name, s + 1, e, len(body))
