"""Undo print line-wrapping in the 1995 listings so they compile.

The listing was printed at ~90 columns, so long source lines wrapped. Inside
/* */ comments and between tokens a wrap is harmless C, so it is left alone.
Only two cases change the program and are rejoined here (nothing else is
touched):
  1. a #define whose body wrapped onto the next line (would leave a stray
     expression at file scope);
  2. a string literal that wrapped (an unterminated string is an error).
Usage: prepare.py SRC_DIR OUT_DIR   (writes lower-case filenames, .H mains -> .c)
"""
import pathlib, re, sys

def unterminated_string(line):
    s = re.sub(r"/\*.*?\*/", "", line)
    s = s.split("//")[0] if '"' not in s.split("//")[0][-1:] else s
    s = re.sub(r"\\.", "", s)
    s = re.sub(r"'[^']*'", "", s)
    return s.count('"') % 2 == 1

def fix(text):
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        l = lines[i]
        while i + 1 < len(lines) and (
            (l.lstrip().startswith("#define") and l.rstrip().endswith(("+", "*", "-", ",")))
            or unterminated_string(l)):
            sep = "" if unterminated_string(l) else " "
            l = l.rstrip() + sep + " " + lines[i + 1].lstrip() if sep else l + " " + lines[i + 1].lstrip()
            i += 1
        out.append(l)
        i += 1
    text = "\n".join(out)
    # 3. transcription gap: one semicolon the scan does not show (BP_INIT.H,
    #    marked /*?*/ in docs/report_text.md) is restored.
    return text.replace('errors\\n")/*?*/', 'errors\\n");/*?*/')

src, dst = map(pathlib.Path, sys.argv[1:3])
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.iterdir()):
    if f.suffix.upper() not in (".C", ".H"):
        continue
    (dst / f.name.lower()).write_text(fix(f.read_text()))
