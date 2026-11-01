"""Write the report's block/network diagrams as draw.io files and export them to 300 dpi PNG.

uv run --no-project python docs/report_1995/diagrams/make_diagrams.py
Needs the draw.io desktop CLI (brew install --cask drawio). The .drawio files are the editable
sources; open them in draw.io to adjust, then re-export with
  drawio -x -f png -s 3.125 -b 10 -o figX_Y.png figX_Y.drawio   (scale 3.125 = 300 dpi at 96 dpi)
"""
import html, pathlib, subprocess, shutil

D = pathlib.Path(__file__).resolve().parent
BOX = "rounded=0;whiteSpace=wrap;html=1;fontSize=12;"
CIRC = "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontSize=12;"
TXT = "text;html=1;align=center;verticalAlign=middle;fontSize=12;"
SQ = "rounded=0;whiteSpace=wrap;html=1;fontSize=10;"
EDGE = "endArrow=block;endFill=1;html=1;rounded=0;fontSize=11;"
LINE = "endArrow=none;html=1;rounded=0;"


class Dia:
    def __init__(s, name):
        s.name, s.cells, s.n = name, [], 2

    def _id(s):
        s.n += 1; return f"c{s.n}"

    def v(s, label, x, y, w=60, h=40, style=BOX):
        i = s._id()
        s.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{style}" vertex="1" parent="1">'
                       f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def t(s, label, x, y, w=80, h=20, extra=""):
        return s.v(label, x, y, w, h, TXT + extra)

    def e(s, a, b, label="", style=EDGE, pts=()):
        i = s._id()
        p = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts)
        s.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{style}" edge="1" parent="1" source="{a}" target="{b}">'
                       f'<mxGeometry relative="1" as="geometry">{"<Array as=%spoints%s>" % (chr(34), chr(34)) + p + "</Array>" if p else ""}</mxGeometry></mxCell>')

    def write(s):
        xml = ('<mxfile host="drawio"><diagram name="%s"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
               % s.name + "".join(s.cells) + "</root></mxGraphModel></diagram></mxfile>")
        (D / f"{s.name}.drawio").write_text(xml)


def delay_line(d, x0, y, labels, inp, w=46):
    """Input text then z^-1 boxes; returns list of tap anchor (small point) ids, first = input."""
    taps = [d.v("", x0, y + 18, 4, 4, "ellipse;fillColor=#000000;")]
    d.t(inp, x0 - 30, y - 20, 70)
    x = x0
    for k, lab in enumerate(labels):
        z = d.v("z<sup>-1</sup>", x + 40, y, w, 40)
        d.e(taps[-1], z)
        p = d.v("", x + 40 + w + 30, y + 18, 4, 4, "ellipse;fillColor=#000000;")
        d.e(z, p, style=LINE)
        d.t(lab, x + 40 + w, y - 20, 80)
        taps.append(p); x += w + 70
    return taps


figs = []
# 2.1 simple channel model
d = Dia("fig2_1")
a = d.t("d(t)", 0, 40, 60); h = d.v("h[.]", 90, 30, 80, 40); s = d.v("&Sigma;", 230, 30, 40, 40, CIRC)
o = d.t("x(t)", 320, 40, 60); g = d.t("g(t)", 220, -40, 60)
d.e(a, h); d.e(h, s, "c(t)"); d.e(g, s); d.e(s, o)
d.t("Transmitted sequence", 0, 80, 120, extra="fontSize=10;"); d.t("Channel noise", 260, -40, 100, extra="fontSize=10;")
d.t("Recieved signal", 300, 80, 100, extra="fontSize=10;"); figs.append(d)

# 2.4 M-tap FIR
d = Dia("fig2_4")
taps = delay_line(d, 0, 0, ["d(n-1)", "d(n-2)", "d(n-(M-1))"], "d(n)")
ws = []; prev = None
for k, (p, lab) in enumerate(zip(taps, ["w<sub>0</sub>", "w<sub>1</sub>", "w<sub>2</sub>", "w<sub>M-1</sub>"])):
    x = 0 if k == 0 else 116 * k
    wb = d.v(lab, x - 18, 80, 40, 30, "triangle;direction=south;whiteSpace=wrap;html=1;fontSize=11;")
    d.e(p, wb)
    sm = d.v("&Sigma;", x - 18, 150, 40, 40, CIRC)
    d.e(wb, sm)
    if prev: d.e(prev, sm)
    prev = sm
out = d.t("c(n)", 116 * 3 + 60, 160, 60); d.e(prev, out); figs.append(d)

# 3.1 general communication system
d = Dia("fig3_1")
data = d.v("Data for transmission", 0, 0, 110, 40); tsg = d.v("Training sequence generator", 0, 70, 110, 40)
tx = d.v("Tx", 150, 35, 50, 40); d.e(data, tx); d.e(tsg, tx)
H = d.v("H[z]", 260, 35, 60, 40); d.e(tx, H, "d(n)")
s = d.v("&Sigma;", 360, 35, 40, 40, CIRC); gn = d.t("g(n)", 350, -30, 60); d.e(H, s); d.e(gn, s)
eq = d.v("Equaliser", 470, 35, 90, 40); d.e(s, eq, "x(n)")
hl = d.v("Hard-limit function", 620, 35, 90, 40); d.e(eq, hl, "y(n)")
on = d.t("o(n)", 760, 45, 50); d.e(hl, on)
rts = d.v("Training sequence generator", 600, 140, 110, 40); es = d.v("&Sigma;", 580, 75, 30, 30, CIRC)
d.e(rts, es, "d(n)"); d.e(eq, es, "-"); d.e(es, eq, "e(n)", pts=[(515, 120)])
for lab, x in (("Transmitter", 40), ("Channel model", 290), ("Reciever", 600)):
    d.t(lab, x, 200, 120, extra="fontStyle=1;")
figs.append(d)

# 3.2 PN generator
d = Dia("fig3_2")
bits = [d.v(f"Bit {k}", 60 + 80 * (k - 1), 40, 60, 40) for k in range(1, 6)]
for a_, b_ in zip(bits, bits[1:]): d.e(a_, b_)
x = d.v("XOR", 220, 130, 50, 36, "shape=or;whiteSpace=wrap;html=1;direction=west;fontSize=10;")
d.e(bits[1], x, pts=[(170, 148)]); d.e(bits[4], x, pts=[(410, 148)]); d.e(x, bits[0], pts=[(30, 148), (30, 60)])
sel = d.v("Symbol selector<br>1 &rarr; +1.0<br>0 &rarr; -1.0", 480, 30, 110, 60); d.e(bits[4], sel)
o = d.t("OUTPUT", 620, 50, 70); d.e(sel, o); figs.append(d)


def layer(d, n, x, y0, dy, style=CIRC, size=36, labels=None):
    return [d.v(labels[k] if labels else "", x, y0 + k * dy, size, size, style) for k in range(n)]


def full(d, A, B):
    for a_ in A:
        for b_ in B: d.e(a_, b_, style="endArrow=none;html=1;")


# 4.1 general feedforward ANN
d = Dia("fig4_1")
I = layer(d, 4, 100, 0, 60, SQ); Hh = layer(d, 3, 240, 30, 60); O = layer(d, 2, 380, 60, 60)
for k, n in enumerate(I): d.e(d.t("", 20, 8 + k * 60, 20), n)
full(d, I, Hh); full(d, Hh, O)
for k, n in enumerate(O): d.e(n, d.t("", 470, 68 + k * 60, 20))
d.t("Network inputs", 0, 230, 100); d.t("Input layer", 80, 230, 80); d.t("Hidden layer", 220, 230, 80)
d.t("Output layer", 360, 230, 80); d.t("Network outputs", 450, 230, 100); figs.append(d)

# 4.2 biological neuron (schematic)
d = Dia("fig4_2")
body = d.v("Neuron Body", 160, 60, 100, 70, "ellipse;whiteSpace=wrap;html=1;fontSize=12;")
for k in range(4): d.e(d.t("", 0, 10 + k * 45, 20), body, style="endArrow=none;curved=1;html=1;")
ax_ = d.t("Output (AXON)", 360, 85, 110); d.e(body, ax_)
d.t("Inputs (DENDRITES)", 0, 190, 140); figs.append(d)

# 4.3 perceptron
d = Dia("fig4_3")
s = d.v("&Sigma;", 260, 100, 50, 50, CIRC)
for k, i in enumerate(["1", "2", "3", "I-1"]):
    x_ = d.t(f"x<sub>{i}</sub>", 0, 20 + k * 60, 40)
    w_ = d.v(f"w<sub>{i}j</sub>", 100, 12 + k * 60, 50, 36, "triangle;direction=east;whiteSpace=wrap;html=1;fontSize=10;")
    d.e(x_, w_); d.e(w_, s)
f = d.v("&phi;[.]", 370, 100, 70, 50); d.e(s, f, "v<sub>j</sub>")
th = d.t("&theta;<sub>j</sub>", 375, 30, 60); d.e(th, f)
y_ = d.t("y<sub>j</sub>", 500, 115, 40); d.e(f, y_)
for lab, x in (("INPUTS", -10), ("WEIGHTS", 90), ("SUMMING UNIT", 230), ("ACTIVATION FUNCTION", 350), ("OUTPUT", 480)):
    d.t(lab, x, 270, 110, extra="fontSize=10;")
figs.append(d)

# 4.4 MLP with bias nodes
d = Dia("fig4_4")
I = layer(d, 5, 80, 0, 55, CIRC, labels=["0", "1", "2", "I-1", "I"])
Hh = layer(d, 4, 240, 25, 55, CIRC, labels=["0", "1", "J-1", "J"])
O = layer(d, 2, 400, 70, 55, CIRC, labels=["0", "K-1"])
full(d, I, Hh[:3]); full(d, Hh, O)
d.t("-1.0", 20, 228, 50); d.t("-1.0", 190, 198, 50)
for k in range(4): d.e(d.t("", 0, 8 + k * 55, 20), I[k])
for k, n in enumerate(O): d.e(n, d.t("", 490, 78 + k * 55, 20))
d.t("w<sub>ij</sub>", 150, -20, 40); d.t("w<sub>jk</sub>", 310, -20, 40); figs.append(d)

# 4.5 feedforward equaliser
d = Dia("fig4_5")
taps = delay_line(d, 0, 0, ["x(n-1)", "x(n-2)", "x(n-(I-1))"], "x(n)")
I = [d.v("", (0 if k == 0 else 116 * k) - 16, 90, 36, 36, SQ) for k in range(4)] + [d.v("-1.0", 450, 90, 36, 36, SQ)]
for p, n in zip(taps, I): d.e(p, n)
Hh = [d.v("", 30 + 100 * k, 190, 36, 36, CIRC) for k in range(4)] + [d.v("-1.0", 450, 190, 36, 36, SQ)]
full(d, I, Hh[:4]); o = d.v("", 200, 290, 36, 36, CIRC); full(d, Hh, [o])
d.e(o, d.t("Equaliser Output y(n)", 170, 360, 120)); d.t("Recieved Signal", -40, -45, 110); figs.append(d)

# 5.1 ADALINE equaliser
d = Dia("fig5_1")
taps = delay_line(d, 0, 0, ["x(n-1)", "x(n-2)", "x(n-(I-1))"], "x(n)")
I = [d.v("", (0 if k == 0 else 116 * k) - 16, 90, 36, 36, SQ) for k in range(4)] + [d.v("-1.0", 450, 90, 36, 36, SQ)]
for p, n in zip(taps, I): d.e(p, n)
o = d.v("", 200, 200, 40, 40, CIRC)
for k, n in enumerate(I): d.e(n, o, f"w<sub>{k if k < 4 else 'I'}0</sub>(n)", "endArrow=none;html=1;fontSize=10;")
s = d.v("&Sigma;", 320, 280, 40, 40, CIRC); d.e(o, s, "y(n)")
d.e(d.t("Desired signal d(n)", 420, 290, 120), s); d.e(s, d.t("Error e(n)", 200, 290, 80)); figs.append(d)

# 7.1 RNN equaliser
d = Dia("fig7_1")
taps = delay_line(d, 0, 0, ["x(n-1)", "x(n-(M-1))"], "x(n)")
I = [d.v("", (0 if k == 0 else 116 * k) - 16, 90, 36, 36, SQ) for k in range(3)] + [d.v("-1.0", 260, 90, 36, 36, SQ)]
F = [d.v(f"y<sub>{l}</sub>(n-1)", 330 + 80 * k, 90, 60, 36, SQ) for k, l in enumerate(["0", "1", "J-1"])]
for p, n in zip(taps, I): d.e(p, n)
O = [d.v(f"y<sub>{l}</sub>(n)", 150 + 110 * k, 220, 60, 40, "ellipse;whiteSpace=wrap;html=1;fontSize=11;") for k, l in enumerate(["0", "1", "J-1"])]
full(d, I + F, O)
for k, (o_, f_) in enumerate(zip(O, F)):
    z = d.v("z<sup>-1</sup>", 560, 300 + 50 * k, 46, 36); d.e(o_, z, pts=[(180 + 110 * k, 318 + 50 * k)])
    d.e(z, f_, pts=[(640 + 20 * k, 318 + 50 * k), (640 + 20 * k, 70 - 10 * k), (360 + 80 * k, 70 - 10 * k)])
d.e(O[0], d.t("Equaliser Output y(n)", 60, 330, 130)); d.t("Recieved Signal", -40, -45, 110); figs.append(d)

# 8.1 RBF node
d = Dia("fig8_1")
s = d.v("&Sigma;", 260, 110, 50, 50, CIRC)
for k, i in enumerate(["0", "1", "2", "i"]):
    n = d.v("||.||", 120, 10 + k * 70, 60, 40)
    d.e(d.t(f"w<sub>{i}</sub>", 0, 10 + k * 70, 40), n); d.e(d.t(f"x<sub>{i}</sub>", 0, 32 + k * 70, 40), n); d.e(n, s)
gf = d.v("Gaussian function", 360, 105, 90, 60); d.e(s, gf); d.e(gf, d.t("y<sub>j</sub>", 500, 125, 40)); figs.append(d)

# 8.2 RBF equaliser
d = Dia("fig8_2")
taps = delay_line(d, 0, 0, ["x(n-1)"], "x(n)")
I = [d.v("", (0 if k == 0 else 116 * k) - 16, 90, 36, 36, SQ) for k in range(2)]
for p, n in zip(taps, I): d.e(p, n)
R = [d.v("", -100 + 60 * k, 200, 36, 36, "ellipse;whiteSpace=wrap;html=1;fillColor=#eeeeee;") for k in range(8)]
full(d, I, R); o = d.v("", 100, 320, 40, 40, CIRC)
for k, n in enumerate(R): d.e(n, o, "-1" if k < 4 else "+1", "endArrow=none;html=1;fontSize=10;")
d.e(o, d.t("Equaliser Output y(n)", 60, 390, 130)); d.t("Recieved Signal", -40, -45, 110)
d.t("RBF (Gaussian) nodes", 380, 208, 140); d.t("ADALINE node", 160, 330, 100); figs.append(d)

drawio = shutil.which("drawio") or "/Applications/draw.io.app/Contents/MacOS/draw.io"
for f in figs:
    f.write()
    subprocess.run([drawio, "-x", "-f", "png", "-s", "3.125", "-b", "10", "-o", str(D / f"{f.name}.png"),
                    str(D / f"{f.name}.drawio")], check=True, capture_output=True)
    print(f.name)
