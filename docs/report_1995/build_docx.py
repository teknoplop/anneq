"""Build docs/report_1995.docx from the proofread OCR text, the 1995 listing and the figure exports.

uv run --no-project --with python-docx --with pillow python docs/report_1995/build_docx.py [--pages pages.json]

Text: docs/report_text.md (verbatim). Listings: src/borland/. Figures: docs/report_1995/diagrams/*.png
(draw.io) and docs/report_1995/plots/*.png (make_plots.py). --pages gives the contents page numbers
(written by check_pages.py after a LibreOffice render) for the TOC field's cached result.
Author/supervisor are custom document properties set to the placeholders "Author"/"Supervisor":
change them in File > Properties > Custom, then select all and press F9 to update the fields.
"""
import argparse, copy, json, pathlib, re
from xml.sax.saxutils import escape
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.shared import Pt, Inches, Cm, RGBColor
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pages", default=None)
ap.add_argument("--edition", choices=("1995", "corrected"), default="1995",
                help="corrected: plots from plots_corrected/ (make_plots.py on results/fixed --impl pyfix), "
                     "text substitutions from corrections.yaml, a note on the edition and margin bars on changed passages")
ap.add_argument("--out", default=None)
args = ap.parse_args()
CORR = args.edition == "corrected"
if args.out is None:
    args.out = str(ROOT / "docs" / ("report_1995_corrected.docx" if CORR else "report_1995.docx"))
if CORR:
    import yaml
    CORRECTIONS = yaml.safe_load((HERE / "corrections.yaml").read_text())
    for c in CORRECTIONS: c["used"] = 0
else:
    CORRECTIONS = []
CHANGED_BAR = '<w:pBdr><w:left w:val="single" w:sz="12" w:space="8" w:color="A6A6A6"/></w:pBdr>'

def correct(t):
    """Corrected edition: apply corrections.yaml; returns (text, changed)."""
    changed = False
    for c in CORRECTIONS:
        if c["old"] in t:
            t = t.replace(c["old"], c["new"]); c["used"] += 1; changed = True
    return t, changed


PAGES = json.loads(pathlib.Path(args.pages).read_text()) if args.pages else {}

PROPS = {  # custom document properties (DOCPROPERTY fields)
    "ReportTitle": "Neural Network Based Adaptive Equalisation Techniques",
    "ReportType": "Final Year Project Thesis",
    "ReportAuthor": "Author",
    "Supervisor": "Supervisor",
    "Degree": "Bachelor of Engineering in Electronics Engineering with Communications",
    "CourseCode": "IIIE",
    "ReportDayMonth": "28 April",
    "Year": "1995",
}
M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
W = nsdecls("w")
GREEK = {"mu": "μ", "sigma": "σ", "eta": "η", "varphi": "φ", "phi": "φ", "epsilon": "ε", "varepsilon": "ε",
         "delta": "δ", "Delta": "Δ", "pi": "π", "theta": "θ", "nabla": "∇", "partial": "∂", "approx": "≈",
         "ge": "≥", "le": "≤", "quad": "\u2003", ",": "\u2009", " ": " "}

# ------------------------------------------------------------------ OMML from a small LaTeX subset
class MathParser:
    def __init__(s, src):
        s.s, s.i = src, 0

    def peek(s):
        return s.s[s.i] if s.i < len(s.s) else ""

    def group(s):
        if s.peek() == "{":
            s.i += 1; depth = 1; st = s.i
            while depth:
                c = s.s[s.i]; depth += (c == "{") - (c == "}"); s.i += 1
            return MathParser(s.s[st:s.i - 1]).parse()
        return s.atom()

    def raw_group(s):
        assert s.peek() == "{"; s.i += 1; st = s.i; depth = 1
        while depth:
            c = s.s[s.i]; depth += (c == "{") - (c == "}"); s.i += 1
        return s.s[st:s.i - 1]

    @staticmethod
    def r(text, plain=False):
        sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
        return f'<m:r>{sty}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'

    def atom(s):
        c = s.peek()
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s.s[s.i:]); name = m.group(1); s.i += len(m.group(0))
            if name == "frac":
                a = s.group(); b = s.group()
                return f"<m:f><m:num>{a}</m:num><m:den>{b}</m:den></m:f>"
            if name == "sqrt":
                return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{s.group()}</m:e></m:rad>'
            if name == "sum":
                sub = sup = ""
                if s.peek() == "_": s.i += 1; sub = s.group()
                if s.peek() == "^": s.i += 1; sup = s.group()
                body = s.group()
                hide = "" if sup else '<m:supHide m:val="1"/>'
                return (f'<m:nary><m:naryPr><m:chr m:val="∑"/>{hide}</m:naryPr><m:sub>{sub}</m:sub>'
                        f"<m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>")
            if name == "op":
                return s.r(s.raw_group(), plain=True)
            if name == "cases":
                rows = MathParser.rows(s.raw_group())
                return (f'<m:d><m:dPr><m:begChr m:val="{{"/><m:endChr m:val=""/></m:dPr><m:e><m:eqArr>'
                        + "".join(f"<m:e>{x}</m:e>" for x in rows) + "</m:eqArr></m:e></m:d>")
            if name == "|":
                return s.r("‖")
            return s.r(GREEK.get(name, name), plain=name in ("quad", ",", " "))
        s.i += 1
        if c.isalpha():
            return s.r(c)
        if c.isdigit() or c == ".":
            m = re.match(r"[0-9.]+", s.s[s.i - 1:]); s.i += len(m.group(0)) - 1
            return s.r(m.group(0))
        if c == "&":  # alignment marks: LibreOffice shows them literally, so rows are just centred
            return ""
        return s.r(c, plain=True)

    @staticmethod
    def rows(src):
        return [MathParser(x).parse() for x in src.split("\\\\")]

    def parse(s):
        out = []
        while s.i < len(s.s):
            if s.peek() == " ":
                s.i += 1; continue
            base = s.atom()
            sub = sup = None
            while s.peek() in ("_", "^"):
                k = s.peek(); s.i += 1; g = s.group()
                if k == "_": sub = g
                else: sup = g
            if sub is not None and sup is not None:
                base = f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
            elif sub is not None:
                base = f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
            elif sup is not None:
                base = f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
            out.append(base)
        return "".join(out)


def omml(src):
    if "\\\\" in src.replace("\\cases{", "\x00"):  # multi-line outside cases -> aligned eqArr
        if "\\cases" not in src:
            rows = MathParser.rows(src)
            return f"<m:oMath {M}><m:eqArr>" + "".join(f"<m:e>{x}</m:e>" for x in rows) + "</m:eqArr></m:oMath>"
    return f"<m:oMath {M}>{MathParser(src).parse()}</m:oMath>"


# Equations: first line in report_text.md -> (lines used, LaTeX subset, equation number)
EQ = {
    157: (1, r"x(t)=h[d(t)]+g(t)", "(2-1)"),
    161: (1, r"x(nT)=d(nT)H[z]+g(nT)", "(2-2)"),
    165: (1, r"x(n)=d(n)H[z]+g(n)", "(2-3)"),
    178: (1, r"\op{SNR (dB)}=10\op{log}_{10}(\frac{1}{\sigma^{2}})", ""),
    182: (1, r"\sigma^{2}=10^{-\op{SNR}/10}", ""),
    186: (1, r"g=\mu_{m}+\op{cos}(2\pi u_{1})\sqrt{-2\sigma^{2}\op{ln}u_{2}}", "(2-4)"),
    194: (1, r"H[z]=0.3482+0.8704z^{-1}+0.3482z^{-1}", ""),
    201: (1, r"H[z]=w_{0}+w_{1}z^{-1}+w_{2}z^{-2}+...+w_{M-1}z^{-(M-1)}", ""),
    205: (1, r"c(n)=\sum_{l=0}^{M-1}{w_{l}d(n-l)}", ""),
    222: (1, r"P[d(n)=+1]=P[d(n)=-1]=0.5", "(3-1)"),
    226: (2, r"o(n)=\cases{+1\quad\op{for}\ y(n)\ge 0\\-1\quad\op{for}\ y(n)<0}", "(3-2)"),
    257: (1, r"H[z]=1.0+0.5z^{-1}", ""),
    261: (1, r"H[z]=0.5+1.0z^{-1}", ""),
    305: (1, r"y_{j}(n)=\varphi_{j}[v_{j}(n)-\theta_{j}]", "(4-1)"),
    307: (1, r"v_{j}(n)=\sum_{i=0}^{I}{w_{ij}x_{i}}", "(4-2)"),
    318: (1, r"w_{Jk}=\theta_{k}", "(4-3)"),
    322: (1, r"y_{j}(n)=\varphi[\sum_{i=0}^{I}{y_{i}(n)w_{ij}(n)}]", "(4-4)"),
    380: (1, r"\varphi[v(n)]=v(n)", ""),
    388: (1, r"e(n)=d(n)-y(n)", ""),
    392: (1, r"w_{i0}(n+1)=w_{i0}(n)+2\eta e(n)x(n-1)", ""),
    434: (1, r"x(n)=d(n)H[z]", ""),
    438: (3, r"y(n)&=x(n)H^{-1}[z]\\&=d(n)H[z]H^{-1}[z]\\&=d(n)", ""),
    444: (1, r"H^{-1}[z]=a_{0}z^{b}+a_{1}z^{(b-1)}+a_{2}z^{(b-2)}+a_{3}z^{(b-3)}+...", ""),
    448: (1, r"H^{-1}[z]=z^{-b}(a_{0}z^{b}+a_{1}z^{(b-1)}+a_{2}z^{(b-2)}+...+a_{I-1}z^{(I-1)})", ""),
    452: (2, r"y(n)&\approx d(n)H[z]H^{-1}[z]\\&\approx d(n)z^{-b}", ""),
    475: (1, r"\varepsilon(n)=\frac{1}{2}\sum_{k=0}^{K}{e_{k}^{2}(n)}", ""),
    479: (1, r"\frac{\partial\varepsilon(n)}{\partial w_{ij}(n)}", ""),
    483: (1, r"\delta_{j}(n)=e_{j}(n)\varphi'[v_{j}(n)]", ""),
    487: (1, r"\Delta w_{ij}(n)=\eta\delta_{j}y_{i}", "(6-1)"),
    491: (1, r"\Delta w_{i0}(n)=(d(n)-y(n))\varphi'[v_{0}(n)]", "(6-2)"),
    495: (1, r"\delta_{i}(n)=\varphi'[v(n)]\sum_{j=0}^{J}{\delta_{j}(n)w_{ij}}", "(6-3)"),
    509: (1, r"\varphi[v(n)]=\op{tanh}(v(n))", "(6-4)"),
    513: (1, r"\varphi'[v(n)]=\frac{d}{dv(n)}(\varphi[v(n)])=1-(\varphi[v(n)])^{2}", "(6-5)"),
    554: (1, r"\pi_{ij}^{k}(n)=\frac{\partial y_{k}}{\partial w_{ij}}", ""),
    558: (1, r"\pi_{ij}^{k}(n)=\varphi'[v_{k}(n)][\sum_{l=0}^{K}{w_{ij}(n)\pi_{ij}^{k}(n-1)}+\delta_{ik}u_{j}(n)]", "(7.1)"),
    566: (1, r"\pi_{ij}^{k}(0)=0", ""),
    572: (1, r"e(n)=d(n)-y(n)", ""),
    575: (1, r"w_{ij}(n+1)=w_{ij}(n)+\eta e(n)\pi_{ij}^{0}", ""),
    598: (1, r"y_{j}=\varphi[\sum_{i=0}^{I}{\|x_{i}-w_{i}\|}]", "(8.1)"),
    602: (1, r"\varphi[v(n)]=\op{exp}[-\frac{(v(n))^{2}}{2\sigma_{w}^{2}}]", "(8.2)"),
    764: (1, r"e=d-y", "(A1)"),
    768: (1, r"y=\sum_{i=0}^{I-1}{w_{i}x_{i}}", "(A2)"),
    772: (1, r"J=\frac{1}{2}E[e^{2}]", "(A3)"),
    776: (2, r"J&=\frac{1}{2}E[d^{2}]-E[\sum_{i=0}^{I-1}{w_{i}x_{i}d}]+\frac{1}{2}E[\sum_{i=0}^{I-1}{\sum_{j=0}^{I-1}{w_{i}w_{j}x_{i}x_{j}}}]"
              r"\\&=\frac{1}{2}E[d^{2}]-\sum_{i=0}^{I-1}{w_{i}E[x_{i}d]}+\frac{1}{2}\sum_{i=0}^{I-1}{\sum_{j=0}^{I-1}{w_{i}w_{j}E[x_{i}x_{j}]}}", ""),
    783: (1, r"r_{d}=E[d^{2}]", ""),
    787: (1, r"r_{dx}(i)=E[dx_{i}]", ""),
    791: (1, r"r_{x}(i,j)=E[x_{i}x_{j}]", ""),
    795: (1, r"J=\frac{1}{2}r_{d}-\sum_{i=0}^{I-1}{w_{i}r_{dx}(i)}+\frac{1}{2}\sum_{i=0}^{I-1}{\sum_{j=0}^{I-1}{w_{i}w_{j}r_{x}(i,j)}}", "(A4)"),
    800: (1, r"\nabla_{w_{j}}J=\frac{\partial J}{\partial w_{j}}=-r_{dx}(j)+\sum_{i=0}^{I-1}{w_{i}r_{x}(i,j)}", ""),
    804: (1, r"\Delta w_{i}(n)=-\eta\nabla_{w_{i}}J(n)", ""),
    808: (2, r"r_{x}(i,j;n)&=x_{i}(n)x_{j}(n)\\r_{dx}(i;n)&=x_{i}(n)d(n)", ""),
    813: (3, r"w_{i}(n+1)&=w_{i}(n)+\eta[x_{i}(n)d(n)-\sum_{j}^{I-1}{w_{j}(n)x_{i}(n)x_{j}(n)}]"
              r"\\&=w_{i}(n)+\eta[d(n)-\sum_{j}^{I-1}{w_{j}(n)x_{j}(n)}]x_{i}(n)"
              r"\\&=w_{i}(n)+\eta[d(n)-y(n)]x_{i}(n),\quad i=0,1,2,...,(I-1)", ""),
}

# Figures: number -> (image, footnote or None)
PLOTS = "plots_corrected" if CORR else "plots"
SUBS = (HERE / PLOTS / "substitutions.txt").read_text() if (HERE / PLOTS / "substitutions.txt").exists() else ""
ZERO = "A BER of zero (no errors in 100 trials of 10,000 bits) cannot be shown on the log scale and is left out."
FIG = {
    "1.1": ("plots/fig1_1.png", "Redrawn as an illustrative sketch; the original was drawn by hand."),
    "2.1": ("diagrams/fig2_1.png", None), "2.2": ("plots/fig2_2.png", "Redrawn as an illustrative sketch."),
    "2.3": ("plots/fig2_3.png", "Regenerated from equation (2-4) with a modern uniform generator, not the 1995 rand(), so the noisy trace differs in detail."),
    "2.4": ("diagrams/fig2_4.png", None), "3.1": ("diagrams/fig3_1.png", None), "3.2": ("diagrams/fig3_2.png", None),
    "4.1": ("diagrams/fig4_1.png", None), "4.2": ("diagrams/fig4_2.png", None), "4.3": ("diagrams/fig4_3.png", None),
    "4.4": ("diagrams/fig4_4.png", None), "4.5": ("diagrams/fig4_5.png", None),
    "4.6": ("plots/fig4_6.png", "Recomputed: Bayes boundary for 16 dB SNR and 1000 new random received samples."),
    "4.7": ("plots/fig4_7.png", "Recomputed Bayes decision function (16 dB SNR)."),
    "5.1": ("diagrams/fig5_1.png", None), "5.2": ("plots/fig5_2.png", None),
    "5.3": ("plots/fig5_3.png", "Regenerated from the rebuilt 1995 listing (LMS.C built unmodified; the Python line-for-line port gives identical results)."),
    "5.4": ("plots/fig5_4.png", "Regenerated from the rebuilt listing. The η = 0.0001 run of the unmodified LMS.C never finishes (its retrain loop has no limit), so that curve comes from the NumPy port. Two of the original curves cannot be produced by the listed code (issue #19)."
            if "fig5_4 eta=0.0001" in SUBS else "Regenerated from the rebuilt listing."),
    "5.5": ("plots/fig5_5.png", "Regenerated from the rebuilt listing. " + ZERO),
    "5.6": ("plots/fig5_6.png", None),
    "5.7": ("plots/fig5_7.png", "Regenerated from the rebuilt listing. " + ZERO),
    "6.1": ("plots/fig6_1.png", None),
    "6.2": ("plots/fig6_2.png", "Regenerated with BPBC.C at 21 dB SNR. The listing only runs on a modern system with one debug fix applied, and its learning rate and momentum are not given in the report (0.1 and 0.5 assumed)."),
    "6.3": (None, "Not reproduced: the 1995 programs do not save the trained network's decision boundary, and each training run gives a different boundary."),
    "6.4": (None, "Not reproduced, for the same reason as figure 6.3."),
    "7.1": ("diagrams/fig7_1.png", None),
    "7.2": ("plots/fig7_2.png", "Regenerated from the rebuilt listing (RNN.C)."),
    "7.3": ("plots/fig7_3.png", "Plotted from the listed code, which makes 3 to 10 times fewer errors than the original figure; the cause is unknown (issue #19). " + ZERO),
    "8.1": ("diagrams/fig8_1.png", None), "8.2": ("diagrams/fig8_2.png", None),
    "8.3": ("plots/fig8_3.png", "Plotted from the listed code (RB.C), which does not match the original figure (issue #19). " + ZERO),
    "9.1": ("plots/fig9_1.png", "Taps from table 3.1."),
    "9.2": ("plots/fig9_2.png", "Regenerated at 20 dB SNR. The ADALINE uses η = 0.008: with the printed 0.08 the listed LMS.C diverges and never finishes."),
    "9.3": ("plots/fig9_3.png", "Regenerated from the rebuilt listing, ADALINE with η = 0.008 (see figure 9.2). " + ZERO),
}
# Corrected edition: figures redrawn from the corrected simulator's run (results/fixed, impl pyfix).
# Each gets a margin bar on its caption and this footnote in place of the 1995 edition's.
FIX = "Corrected: redrawn from the simulator re-run with issues #10-#18 fixed (results/fixed). "
FIG_CORRECTED = {
    "5.3": FIX + "The curves move only slightly.",
    "5.4": FIX + "No trial passes the training test with η = 0.1 or 0.0001, so those two curves average every trial's last attempt.",
    "5.5": FIX + ZERO, "5.7": FIX + ZERO,
    "6.2": FIX + "BPBC.C at 21 dB SNR; learning rate and momentum are not given in the report (0.1 and 0.5 assumed).",
    "7.2": FIX + "Learning rate 0.12 (see section 7.3).",
    "7.3": FIX + "Learning rate 0.12. The original figure shows many more errors than either the listed or the corrected code (issue #19). " + ZERO,
    "8.3": FIX + "RB.C has none of the corrected bugs; the original figure still does not match (issue #19). " + ZERO,
    "9.2": FIX + "At 20 dB SNR. ADALINE η = 0.008 (the printed 0.08 diverges), RNN η = 0.08.",
    "9.3": FIX + "ADALINE η = 0.008, RNN η = 0.08. " + ZERO,
}
if CORR:
    for k, note in FIG_CORRECTED.items():
        FIG[k] = (FIG[k][0].replace("plots/", PLOTS + "/"), note)

# ------------------------------------------------------------------ document scaffolding
doc = Document()
st = doc.styles
st["Normal"].font.name = "Calibri"; st["Normal"].font.size = Pt(11)
st["Normal"].paragraph_format.space_after = Pt(6); st["Normal"].paragraph_format.line_spacing = 1.15
for name, size in (("Heading 1", 18), ("Heading 2", 13)):
    s = st[name]; s.font.name = "Calibri"; s.font.size = Pt(size); s.font.color.rgb = RGBColor(0, 0, 0); s.font.bold = True
    rpr = s.element.get_or_add_rPr()
    for t in rpr.findall(qn("w:rFonts")): rpr.remove(t)
    rpr.insert(0, parse_xml(f'<w:rFonts {W} w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'))
st["Caption"].font.size = Pt(10); st["Caption"].font.color.rgb = RGBColor(0, 0, 0); st["Caption"].font.italic = False
code = st.add_style("Code", 1); code.font.name = "Consolas"; code.font.size = Pt(7.5)
code.paragraph_format.space_after = Pt(0); code.paragraph_format.line_spacing = 1.0
front = st.add_style("Front Heading", 1); front.base_style = st["Normal"]; front.font.size = Pt(16); front.font.bold = True
front.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER; front.paragraph_format.space_after = Pt(18)

# Heading numbering "1." / "1.1" so STYLEREF \s gives the chapter for figure/table numbers
numbering = doc.part.numbering_part.element
_first_num = numbering.find(qn("w:num"))
_abs_add = (lambda el: _first_num.addprevious(el)) if _first_num is not None else numbering.append
_abs_add(parse_xml(
    f'<w:abstractNum {W} w:abstractNumId="90"><w:multiLevelType w:val="multilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:pStyle w:val="Heading1"/><w:lvlText w:val="%1."/>'
    '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="567" w:hanging="567"/></w:pPr></w:lvl>'
    '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:pStyle w:val="Heading2"/><w:lvlText w:val="%1.%2"/>'
    '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="567" w:hanging="567"/></w:pPr></w:lvl></w:abstractNum>'))
_abs_add(parse_xml(
    f'<w:abstractNum {W} w:abstractNumId="91"><w:multiLevelType w:val="singleLevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/>'
    '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>'))
# w:num must follow all abstractNums
numbering.append(parse_xml(f'<w:num {W} w:numId="90"><w:abstractNumId w:val="90"/></w:num>'))
numbering.append(parse_xml(f'<w:num {W} w:numId="91"><w:abstractNumId w:val="91"/></w:num>'))
for name, lvl in (("Heading 1", 0), ("Heading 2", 1)):
    ppr = st[name].element.get_or_add_pPr()
    _np = parse_xml(f'<w:numPr {W}><w:ilvl w:val="{lvl}"/><w:numId w:val="90"/></w:numPr>')
    _before = [c for c in ppr if c.tag in (qn("w:keepNext"), qn("w:keepLines"), qn("w:pageBreakBefore"), qn("w:framePr"), qn("w:widowControl"))]
    (_before[-1].addnext(_np) if _before else ppr.insert(0, _np))

body = doc.element.body


def add(el):
    body.sectPr.addprevious(el)


sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"): setattr(sec, side, Cm(2.5))
sec.top_margin = sec.bottom_margin = Cm(2.5)
TEXTW = Cm(16)
CENTER, RIGHT = 4536, 9072


# ------------------------------------------------------------------ run/field helpers
def run_xml(text, sup=False, sub=False, bold=False, italic=False, size=None):
    rpr = ""
    if bold: rpr += "<w:b/>"
    if italic: rpr += "<w:i/>"
    if size: rpr += f'<w:sz w:val="{int(size * 2)}"/>'
    if sup: rpr += '<w:vertAlign w:val="superscript"/>'
    if sub: rpr += '<w:vertAlign w:val="subscript"/>'
    parts = text.split("\t"); out = []
    for k, ptxt in enumerate(parts):
        if k: out.append(f"<w:r><w:rPr>{rpr}</w:rPr><w:tab/></w:r>")
        if ptxt: out.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(ptxt)}</w:t></w:r>')
    return "".join(out)


def field(instr, result, **fmt):
    rpr = run_xml("x", **fmt); rpr = rpr[rpr.index("<w:rPr>"):rpr.index("</w:rPr>") + 8]
    return (f'<w:r>{rpr}<w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r>{rpr}<w:instrText xml:space="preserve"> {escape(instr)} </w:instrText></w:r>'
            f'<w:r>{rpr}<w:fldChar w:fldCharType="separate"/></w:r>{run_xml(result, **fmt)}'
            f'<w:r>{rpr}<w:fldChar w:fldCharType="end"/></w:r>')


def prop(name, **fmt):
    return field(f"DOCPROPERTY {name} \\* MERGEFORMAT", PROPS[name], **fmt)


def para(inner, style=None, align=None, ppr_extra="", keep_next=False):
    ppr = ""
    if style: ppr += f'<w:pStyle w:val="{style}"/>'
    if keep_next: ppr += "<w:keepNext/>"
    ppr += ppr_extra
    if align: ppr += f'<w:jc w:val="{align}"/>'
    p = parse_xml(f"<w:p {W} {M}><w:pPr>{ppr}</w:pPr>{inner}</w:p>")
    add(p); return p


def page_break():
    para('<w:r><w:br w:type="page"/></w:r>')


ROMAN = {}
GREEK_RE = re.compile(r"\\(" + "|".join(sorted((k for k in GREEK if k.isalpha()), key=len, reverse=True)) + r")(?![A-Za-z]) ?")
SCRIPT_RE = re.compile(r"([_^])(\{[^}]*\}|\((?:[^()]|\([^()]*\))*\)|-?[A-Za-z0-9]+)")
CITE_RE = re.compile(r"(\[\d+\]|\[a\])")
REF_RE = re.compile(r"\b([Ff]igures?|[Tt]able) (\d+\.\d+)")
footnotes = []   # (id, text)
bookmarks = {}   # "Fig5_3" -> "5.3"
FIG_SEEN = set(re.findall(r"\[Figure (\d+\.\d+):", (ROOT / "docs" / "report_text.md").read_text())) | {"3.1", "4.1"}


def footnote_ref(text):
    fid = len(footnotes) + 1; footnotes.append((fid, text))
    return f'<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:footnoteReference w:id="{fid}"/></w:r>'


def prose(text, cites=True, refs=True, bold=False, size=None):
    """Plain report text -> runs: greek escapes, _sub/^sup, superscript citations, REF fields."""
    text = GREEK_RE.sub(lambda m: GREEK[m.group(1)], text).replace("\\sqrt", "√")
    out = []
    pos = 0
    tokens = []
    for m in SCRIPT_RE.finditer(text):
        tokens.append((m.start(), m.end(), "sub" if m.group(1) == "_" else "sup", m.group(2).strip("{}")))
    if cites:
        for m in CITE_RE.finditer(text): tokens.append((m.start(), m.end(), "cite", m.group(1)))
    if refs:
        for m in REF_RE.finditer(text):
            kind = "Fig" if m.group(1).lower().startswith("fig") else "Tab"
            tokens.append((m.start(2), m.end(2), "ref", f"{kind}{m.group(2).replace('.', '_')}|{m.group(2)}"))
    tokens.sort()
    for s0, e0, kind, val in tokens:
        if s0 < pos: continue
        out.append(run_xml(text[pos:s0], bold=bold, size=size))
        if kind == "sub": out.append(run_xml(val.strip("()") if val.startswith("(") and kind == "sub" else val, sub=True, bold=bold, size=size))
        elif kind == "sup": out.append(run_xml(val, sup=True, bold=bold, size=size))
        elif kind == "cite":
            out.append(footnote_ref(FOOTNOTE_A) if val == "[a]" else run_xml(val, sup=True, bold=bold, size=size))
        else:
            bm, num = val.split("|"); out.append(field(f"REF {bm} \\h", num, bold=bold, size=size))
        pos = e0
    out.append(run_xml(text[pos:], bold=bold, size=size))
    return "".join(out)


def caption(kind, num, text, note=None, changed=False):
    chap, n = num.split(".")
    bm = f"{'Fig' if kind == 'Figure' else 'Tab'}{chap}_{n}"
    bid = len(bookmarks) + 100; bookmarks[bm] = num
    inner = (run_xml(kind + " ", bold=True) + f'<w:bookmarkStart w:id="{bid}" w:name="{bm}"/>'
             + run_xml(chap + ".", bold=True)
             + field(f"SEQ {kind} \\* ARABIC" + (" \\r 1" if n == "1" else ""), n, bold=True) + f'<w:bookmarkEnd w:id="{bid}"/>'
             + run_xml(" - ") + prose(text, refs=False))
    if note: inner += footnote_ref(note)
    return para(inner, style="Caption", align="center", keep_next=(kind == "Table"), ppr_extra=CHANGED_BAR if changed else "")


def image(path, max_w=TEXTW, max_h=Cm(9.5)):
    w, h = Image.open(path).size
    cx = min(max_w, int(max_h * w / h)); cy = int(cx * h / w)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=cx, height=cy)
    add(p._p)  # move to end (after sectPr handling below)
    return p


def figure(num, cap):
    img, note = FIG.get(num, (None, None))
    if img:
        image(HERE / img)
    else:
        para(run_xml(f"[Figure {num} not reproduced]", italic=True), align="center", keep_next=True,
             ppr_extra='<w:pBdr><w:top w:val="single" w:sz="4" w:space="12" w:color="999999"/>'
                       '<w:bottom w:val="single" w:sz="4" w:space="12" w:color="999999"/></w:pBdr>')
    caption("Figure", num, cap, note, changed=CORR and num in FIG_CORRECTED)


def equation(latex, number):
    inner = '<w:r><w:tab/></w:r>' + omml(latex) + ('<w:r><w:tab/></w:r>' + run_xml(number) if number else "")
    para(inner, ppr_extra=f'<w:tabs><w:tab w:val="center" w:pos="{CENTER}"/><w:tab w:val="right" w:pos="{RIGHT}"/></w:tabs>')


def table(rows, widths, header=True):
    tw = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    trs = ""
    for ri, row in enumerate(rows):
        tcs = ""
        for ci, cell in enumerate(row):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if header and ri == 0 else ""
            tcs += (f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:after="0"/><w:jc w:val="center"/></w:pPr>{prose(cell, cites=False, refs=False, bold=header and ri == 0)}</w:p></w:tc>')
        trs += f"<w:tr>{tcs}</w:tr>"
    b = "".join(f'<w:{s} w:val="single" w:sz="4" w:color="000000"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    t = parse_xml(f'<w:tbl {W}><w:tblPr><w:tblW w:w="{tw}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{b}</w:tblBorders>'
                  f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{trs}</w:tbl>')
    add(t); para("")


NOTE_EDITION = [
    "This is the 1995 report re-run with the bugs in its simulation programs corrected. The programs of "
    "appendix B were rebuilt and the bugs listed below fixed (issue numbers refer to the project's issue "
    "tracker); the simulations of chapters 5 to 9 were then repeated with the same settings.",
    "#10 the RNN nodes did not receive the fed-back outputs; #11 the RTRL update counted one term several times over, "
    "which in effect only scaled the learning rate, so the RNN learning rates were re-chosen; #12 one weight too many was initialised in the ADALINE; "
    "#13 the ADALINE bias input was not used; #14 retraining had no limit (now at most 10 attempts); "
    "#15 the random number generator was re-seeded from the clock; #16 the 31-bit PN sequence is replaced "
    "by random training bits; #17 debug output and fixed file paths in BPBC.C; #18 minor program faults.",
    "Only the text and figures that depend on the results were changed. Everything else is as written in "
    "1995, including its spelling. Each changed passage and the caption of each redrawn figure is marked "
    "with a grey bar in the left margin, and the redrawn figures carry a footnote beginning \"Corrected\".",
]

# ------------------------------------------------------------------ title page (section 1)
lines = (ROOT / "docs" / "report_text.md").read_text().split("\n")
L = lambda n: lines[n - 1]
FOOTNOTE_A = L(146)[4:].strip()

para(run_xml("Project Thesis submitted in support of ", size=12) + prop("Degree", size=12), align="center")
for _ in range(4): para("")
para(prop("ReportTitle", size=40), ppr_extra='<w:spacing w:after="240" w:line="240" w:lineRule="auto"/>')
para(prop("ReportType", size=16), align="right",
     ppr_extra='<w:pBdr><w:top w:val="single" w:sz="8" w:space="4" w:color="000000"/></w:pBdr>')
for _ in range(10): para("")
para(prop("ReportAuthor", size=14) + run_xml(" - ", size=14) + prop("CourseCode", size=14), align="right")
para(run_xml("Project Supervisor - ", size=14) + prop("Supervisor", size=14), align="right")
para(prop("ReportDayMonth", size=14) + run_xml(", ", size=14) + prop("Year", size=14), align="right")

sectpr_xml = []


def end_section(fmt=None, start=None, hdr=True):
    """Close the current section by putting a sectPr in the last paragraph."""
    sp = copy.deepcopy(doc.sections[0]._sectPr)
    for e in list(sp):
        if e.tag in (qn("w:headerReference"), qn("w:footerReference"), qn("w:pgNumType"), qn("w:titlePg")): sp.remove(e)
    if fmt:
        sp.append(parse_xml(f'<w:pgNumType {W} w:fmt="{fmt}"' + (f' w:start="{start}"' if start else "") + "/>"))
    sectpr_xml.append((sp, hdr))
    p = parse_xml(f"<w:p {W}><w:pPr/></w:p>"); p[0].append(sp); add(p)


end_section(hdr=False)

# ------------------------------------------------------------------ front matter (section 2, roman)
if CORR:
    para(run_xml("NOTE ON THIS EDITION"), style="Front Heading")
    for t in NOTE_EDITION:
        para(prose(t, cites=False, refs=False), align="both")
    page_break()
para(run_xml(L(17)), style="Front Heading")
for n in range(19, 22):
    if L(n).strip():
        txt, ch = correct(L(n)); para(prose(txt), ppr_extra=CHANGED_BAR if ch else "")
page_break()
para(run_xml("CONTENTS"), style="Front Heading")
TOC_ENTRIES = []  # filled while building the body: (level, text)
toc_anchor = para("")  # TOC field goes here once the headings are known
page_break()
para(run_xml("NOTATION"), style="Front Heading")
rows = []
for n in range(83, 108):
    m = re.match(r"(\S+(?: \S+)?)\s{2,}(.*)", L(n))
    if m: rows.append(m.groups())
tw = [2200, 7000]
trs = "".join(f'<w:tr><w:tc><w:tcPr><w:tcW w:w="{tw[0]}" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:spacing w:after="40"/></w:pPr>{prose(a, cites=False, refs=False)}</w:p></w:tc>'
              f'<w:tc><w:tcPr><w:tcW w:w="{tw[1]}" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:spacing w:after="40"/></w:pPr>{prose(b, cites=False, refs=False)}</w:p></w:tc></w:tr>' for a, b in rows)
add(parse_xml(f'<w:tbl {W}><w:tblPr><w:tblW w:w="9200" w:type="dxa"/></w:tblPr><w:tblGrid><w:gridCol w:w="{tw[0]}"/><w:gridCol w:w="{tw[1]}"/></w:tblGrid>{trs}</w:tbl>'))
para("")
end_section("lowerRoman", 2)

# ------------------------------------------------------------------ body (section 3, arabic)
HEAD1 = re.compile(r"^(\d+)\. ([A-Z].*)$")
HEAD2 = re.compile(r"^(\d+)\.(\d+) (.*)$")
TOC_TEXT = "\n".join(lines[23:80]).upper()
FIGLINE = re.compile(r"^\[Figure (\d+\.\d+): (.*)$")
DESC = re.compile(r"^(Log|y-axis|Axes|Diagram|Both panels|3-D|Bar chart)")
FIXES = {  # OCR/handwriting fixes (recorded in NOTES.md)
    "Reference [???] (handwritten correction: 14)": "Reference [14]",
    "Reference [???] (handwritten correction: 6)": "Reference [6]",
    "reference [???] (handwritten correction: 9)": "reference [9]",
    "network!2\"! and": "network[21] and",
}


def fix(t):
    for a, b in FIXES.items(): t = t.replace(a, b)
    return t



def heading(level, text, numbered=True):
    clean = re.sub(r"\s*(\[\d+\])+\s*$", "", text)
    TOC_ENTRIES.append((level, clean, numbered))
    ppr = "" if numbered else '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>'
    para(prose(text, refs=False), style=f"Heading{level}", ppr_extra=ppr)


pending = []        # paragraph text awaiting a possible continuation
deferred = []       # figures/tables waiting for the paragraph to be completed
fig_count = 0


def flush():
    global pending
    if pending:
        kind, txt = pending
        txt, ch = correct(fix(txt))
        bar = CHANGED_BAR if ch else ""
        if kind == "bullet":
            para(prose(txt), ppr_extra='<w:numPr><w:ilvl w:val="0"/><w:numId w:val="91"/></w:numPr>' + bar)
        elif kind == "indent":
            para(prose(txt), ppr_extra='<w:ind w:left="720"/>' + bar)
        else:
            para(prose(txt), align="both", ppr_extra=bar)
        pending = []
    for f in deferred: f()
    deferred.clear()


def text_line(kind, txt):
    global pending
    if pending and pending[0] == kind == "p" and not re.search(r"[.:;?!\"”)]$", pending[1]):
        pending[1] += " " + txt.strip(); return "merged"
    return None


n = 109
first_chapter = True
in_refs = False
while n <= 817:
    s = L(n); st_ = s.strip()
    if n in EQ:
        cnt, latex, num = EQ[n]; flush(); equation(latex, num); n += cnt; continue
    if n == 146 or st_.startswith("<!--") or not st_:
        if not st_ and pending and pending[0] == "p" and re.search(r"[.:;?!\"”)]$", pending[1]): flush()
        n += 1; continue
    m = FIGLINE.match(st_)
    if m:
        num, rest = m.groups(); capt = rest.rstrip("]")
        mm = re.match(r"(.*?\.)(?:\s|$)", capt)
        capt = mm.group(1) if mm else capt
        deferred.append(lambda num=num, capt=capt: figure(num, capt))
        if not pending: flush()
        n += 1
        while n <= 817 and (DESC.match(L(n).strip()) or (L(n).strip() and not L(n - 1).strip().endswith("]") and False)):
            n += 1
        continue
    if DESC.match(st_):
        n += 1; continue
    mt = re.match(r"^Table (\d+\.\d+) - (.*)$", st_)
    if mt:
        flush(); caption("Table", mt.group(1), mt.group(2))
        if mt.group(1) == "3.1":
            rws = [[c.strip() for c in L(k).strip().strip("|").split("|")] for k in (n + 2,) + tuple(range(n + 4, n + 9))]
            table(rws, [1700, 1700, 1700, 1700]); n += 9
        else:
            rws = [L(k).split() for k in range(n + 2, n + 11)]
            table(rws, [1100] * 5); n += 11
        continue
    m1, m2 = HEAD1.match(st_), HEAD2.match(st_)
    if m2 and m2.group(3).split("[")[0].strip().upper() in TOC_TEXT:
        flush(); heading(2, m2.group(3)); n += 1; continue
    if m1 and m1.group(2).split("[")[0].strip().upper() in TOC_TEXT and len(st_) < 70:
        flush()
        if not first_chapter: page_break()
        first_chapter = False
        heading(1, m1.group(2)); n += 1; continue
    if st_ in ("References",) or st_.startswith("Appendix A"):
        flush(); page_break(); heading(1, st_, numbered=False); in_refs = st_ == "References"; n += 1; continue
    if in_refs:
        if re.match(r"^[\[{]\d+\]", st_):
            flush(); pending = ["ref", st_]
        else:
            pending[1] += " " + st_
        n += 1
        if pending and pending[0] == "ref" and (not L(n).strip() or n > 757):
            nxt = n
            while nxt <= 817 and (not L(nxt).strip() or L(nxt).startswith("<!--")): nxt += 1
            if re.match(r"^[\[{]\d+\]", L(nxt).strip()) or L(nxt).startswith("Appendix") or nxt > 757:
                para(prose(fix(pending[1]), cites=False, refs=False), ppr_extra='<w:ind w:left="567" w:hanging="567"/>'); pending = []
        continue
    if st_.startswith("* ") or st_.startswith("• "):
        flush(); pending = ["bullet", st_[2:]]; flush(); n += 1; continue
    if re.match(r"^\d\. ", st_) and (s.startswith(" ") or 499 < n < 580):
        flush(); pending = ["indent", st_]; flush(); n += 1; continue
    if text_line("p", st_):
        if L(n - 1).strip() and not L(n - 1).startswith("<!--") or True:
            pass
        n += 1; continue
    flush(); pending = ["p", st_]; n += 1
flush()

# ------------------------------------------------------------------ Appendix B: listings from src/borland
page_break()
heading(1, "Appendix B - Program listings", numbered=False)
ORDER = ["EQ_FUNCS.H", "EQ_FUNCS.C", "LMS.H", "LMS_INIT.H", "LMS.C", "RNNBC.H", "RNN_INIT.H", "RNN.C",
         "BPBC.H", "BP_INIT.H", "BPBC.C", "RB.H", "RB_INIT.H", "RB.C"]
for k, fn in enumerate(ORDER):
    if k: page_break()
    for ln in (ROOT / "src/borland" / fn).read_text(errors="replace").replace("\r", "").rstrip("\n").split("\n"):
        para(run_xml(ln.expandtabs(8)), style="Code")
end_section("decimal", 1)
doc.sections[-1]._sectPr  # final body sectPr is the document's own

# move the original final sectPr settings: page numbers restart for the body
final = doc.element.body.sectPr
for sp, hdr in sectpr_xml[-1:]:
    pass

# ------------------------------------------------------------------ TOC field (cached result)
def toc_line(level, text, numbered, num):
    page = PAGES.get(text, "")
    ind = 0 if level == 1 else 400
    label = (num + " " if numbered else "") + text
    style = "TOC1" if level == 1 else "TOC2"
    return (f'<w:pPr><w:pStyle w:val="{style}"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{RIGHT}"/></w:tabs>'
            f'<w:spacing w:before="{120 if level == 1 else 0}" w:after="{60 if level == 1 else 0}"/><w:ind w:left="{ind}"/></w:pPr>'
            + run_xml(label + "\t" + str(page), bold=level == 1))


c1 = c2 = 0
toc_ps = []
for level, text, numbered in TOC_ENTRIES:
    if level == 1 and numbered: c1 += 1; c2 = 0; num = f"{c1}."
    elif level == 2: c2 += 1; num = f"{c1}.{c2}"
    else: num = ""
    toc_ps.append(toc_line(level, text, numbered, num))
first = toc_ps[0]
idx = first.index("</w:pPr>") + 8
toc_ps[0] = (first[:idx] + '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
             '<w:r><w:fldChar w:fldCharType="separate"/></w:r>' + first[idx:])
toc_ps[-1] += '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
for xml in reversed(toc_ps):
    toc_anchor.addnext(parse_xml(f"<w:p {W}>{xml}</w:p>"))
toc_anchor.getparent().remove(toc_anchor)

# ------------------------------------------------------------------ headers / footers
def hf_part(kind, inner, border=True):
    from docx.opc.constants import CONTENT_TYPE as CT
    tag = "hdr" if kind == "header" else "ftr"
    xml = (f'<w:{tag} {W}><w:p><w:pPr>'
           + (f'<w:pBdr><w:{"bottom" if tag == "hdr" else "top"} w:val="single" w:sz="4" w:space="4" w:color="000000"/></w:pBdr>' if border else "")
           + f'<w:tabs><w:tab w:val="center" w:pos="{CENTER}"/><w:tab w:val="right" w:pos="{RIGHT}"/></w:tabs></w:pPr>{inner}</w:p></w:{tag}>')
    i = len(doc.part.package.parts) + 1
    part = Part(PackURI(f"/word/{kind}{i}.xml"), CT.WML_HEADER if tag == "hdr" else CT.WML_FOOTER, xml.encode(), doc.part.package)
    return doc.part.relate_to(part, RT.HEADER if tag == "hdr" else RT.FOOTER)


sz = dict(size=8)
hdr_rid = hf_part("header", prop("ReportTitle", **sz) + run_xml("\t\t", **sz) + prop("ReportDayMonth", **sz) + run_xml(", ", **sz) + prop("Year", **sz))
ftr_rid = hf_part("footer", prop("ReportAuthor", **sz) + run_xml(" - ", **sz) + prop("CourseCode", **sz) + run_xml("\t", **sz)
                  + prop("ReportType", **sz) + run_xml("\tPage ", **sz) + field("PAGE", "1", **sz))
blank_h = hf_part("header", "", False); blank_f = hf_part("footer", "", False)
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
all_sp = [sp for sp, _ in sectpr_xml]
# The last end_section() sectPr duplicates the body's own final sectPr: drop that paragraph and use
# the document-level sectPr for the body section.
last_p = all_sp[-1].getparent().getparent(); last_p.getparent().remove(last_p)
final = doc.element.body.sectPr
final.append(parse_xml(f'<w:pgNumType {W} w:fmt="decimal" w:start="1"/>'))
targets = [(all_sp[0], False), (all_sp[1], True), (final, True)]
for sp, hdr in targets:
    h, f = (hdr_rid, ftr_rid) if hdr else (blank_h, blank_f)
    sp.insert(0, parse_xml(f'<w:footerReference {W} xmlns:r="{R}" w:type="default" r:id="{f}"/>'))
    sp.insert(0, parse_xml(f'<w:headerReference {W} xmlns:r="{R}" w:type="default" r:id="{h}"/>'))
    # schema order: pgNumType must come after pgMar etc.; move it to just before cols/docGrid
    for e in sp.findall(qn("w:pgNumType")):
        sp.remove(e); anchor = sp.find(qn("w:cols")) if sp.find(qn("w:cols")) is not None else sp.find(qn("w:docGrid"))
        (anchor.addprevious(e) if anchor is not None else sp.append(e))

# ------------------------------------------------------------------ footnotes part
from docx.opc.constants import CONTENT_TYPE as CT
fn_xml = (f'<w:footnotes {W}><w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
          '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>')
for fid, text in footnotes:
    fn_xml += (f'<w:footnote w:id="{fid}"><w:p><w:pPr><w:spacing w:after="0"/></w:pPr><w:r><w:rPr><w:vertAlign w:val="superscript"/><w:sz w:val="16"/></w:rPr><w:footnoteRef/></w:r>'
               f'{run_xml(" " + text, size=8)}</w:p></w:footnote>')
fn_xml += "</w:footnotes>"
doc.part.relate_to(Part(PackURI("/word/footnotes.xml"), CT.WML_FOOTNOTES, fn_xml.encode(), doc.part.package), RT.FOOTNOTES)

# ------------------------------------------------------------------ properties (core, app, custom)
cp = doc.core_properties
cp.title = PROPS["ReportTitle"]; cp.author = "Author"; cp.last_modified_by = "Author"
cp.subject = PROPS["ReportType"]; cp.comments = ""; cp.keywords = ""; cp.category = ""
cp.revision = 1
import datetime
cp.created = cp.modified = datetime.datetime(1995, 4, 28, 12, 0, 0)
custom = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
          'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
          + "".join(f'<property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="{k + 2}" name="{name}"><vt:lpwstr>{escape(v)}</vt:lpwstr></property>'
                    for k, (name, v) in enumerate(PROPS.items())) + "</Properties>")
doc.part.package.relate_to(Part(PackURI("/docProps/custom.xml"), "application/vnd.openxmlformats-officedocument.custom-properties+xml",
                                custom.encode(), doc.part.package),
                           "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties")
app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
       'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application>'
       '<Company></Company><Manager></Manager></Properties>')
for rel in list(doc.part.package.rels.values()):
    if rel.reltype.endswith("/extended-properties"):
        rel.target_part._blob = app.encode()
        break
else:
    doc.part.package.relate_to(Part(PackURI("/docProps/app.xml"), "application/vnd.openxmlformats-officedocument.extended-properties+xml",
                                    app.encode(), doc.part.package),
                               "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties")

_z = doc.settings.element.find(qn("w:zoom"))
if _z is not None: _z.set(qn("w:percent"), "100")
unused = [c["id"] for c in CORRECTIONS if c["used"] != 1]
assert not unused, f"corrections not applied exactly once: {unused}"
doc.save(args.out)
json.dump([t for _, t, _ in TOC_ENTRIES], open(HERE / "toc_entries.json", "w"), indent=0)
print("wrote", args.out, "figures", len([b for b in bookmarks if b.startswith("Fig")]), "footnotes", len(footnotes))
