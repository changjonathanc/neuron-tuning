"""Wide blog figures, one per demo: real examples on top, behaviour rates and change on unrelated text below.

    uv run figures/demo_story.py

Writes figures/story_<demo>.png (1600 px wide), .svg and .json (title, caption, example rule, every number).
Three parts:
  (a) examples: real greedy replies (seed 0 at the matched point) to 2-3 prompts, chosen by the FIXED rule in the
      JSON caption (the same rule as the demo's card), with the card's yes/no tag;
  (b) behaviour: share of replies for the target set and the 1-2 most important leak sets (the design of
      demo_charts.py: fixed colours, thin 95% Wilson interval lines);
  (c) change on unrelated text (the F of the demo tables) as linear bars, standard LoRA = 100%, with "N× less".
Data: ONLY the saved files of the demos. Every count is also checked against the demo README (check_docs), and
every example reply is read from the saved generations (the prompt text is asserted to be the same in each run).
"""
import gzip
import importlib.util
import json
import math
import re
import sys
import textwrap
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matplotlib.backends.backend_agg import RendererAgg  # noqa: E402
from style import BAD, BG, CARD, GOOD, GRID, INK, INK2, MUTED, OUT, ROOT, plt  # noqa: E402
import demo_charts as dc  # noqa: E402
from demo_charts import COLS, D, check_docs, kn, num, wilson  # noqa: E402

E15 = ROOT / "experiments/15_learned_gates"

# Fixed method names and colours (demo_charts.METHODS), plus the LoRA trained with negative examples (hatched).
METHODS = {m: (v[0], v[1]) for m, v in dc.METHODS.items()}
METHODS["lora_neg"] = ("LoRA, standard + 160 negative examples", dc.METHODS["lora_std"][1])
METHODS["kstart"] = ("8 neurons chosen at the reply start", dc.METHODS["fix"][1])
HATCH = {"lora_neg": "////"}
SHORT = {"original": "Original model", "lora_std": "LoRA, standard", "lora_neg": "LoRA + negatives",
         "neurons": "8 neurons", "neurons_pres": "8 neurons + preservation", "fix": "2 reply-start neurons",
         "kstart": "8 reply-start neurons"}

def short(m, ms):
    """Row label; "8 neurons (default)" when a reply-start version is in the same figure."""
    return "8 neurons (default)" if m == "neurons" and {"fix", "kstart"} & set(ms) else SHORT[m]


W = 8.0          # inches; dpi 200 -> 1600 px. At 1000 px on a page, 1 pt = 1.74 px.
FS = 9.0         # base text size in pt (about 16 px at 1000 px)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gens(path):
    return json.loads(Path(path).read_text())["sets"]


def clean(t):
    t = re.sub(r"\s*↵\s*", " ", t)
    t = re.sub(r"\$([^$]*)\$", r"\1", t.replace("**", "").replace("\\frac", "").replace("\\cdot", "·"))
    return " ".join(t.replace("#", "").replace("\n", " ").split())


HL0, HL1 = "\x01", "\x02"         # invisible marks around the key part of a reply (drawn highlighted)
HL_BOX = dict(boxstyle="round,pad=0.08", fc="#fbe3a6", ec="none")   # light amber behind dark ink


def mark(text, pattern):
    """Put highlight marks around every match of `pattern` in the shown text."""
    return re.sub(pattern, lambda m: HL0 + m.group(0) + HL1, text) if pattern else text


def unmark(text):
    return text.replace(HL0, "").replace(HL1, "")


def dwidth(s):
    return sum(0 if c in (HL0, HL1) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


MEASURE = RendererAgg(10, 10, 2000)


def draw_lines(fig, ax, texts, x, y, lines, lh, fs, color, family=None):
    """Draw wrapped lines; marked parts are bold on a light amber box. Line i has its top at y + i * lh."""
    kw = dict(fontsize=fs, color=color, va="baseline", ha="left", family=family)

    def width(t):   # measured by a 2000-dpi renderer: text extents are rounded to whole pixels
        return t.get_window_extent(renderer=MEASURE).width / 2000

    probe = [ax.text(0, 0, s, **kw) for s in ("x x", "xx")]
    sp = width(probe[0]) - width(probe[1])
    for t in probe:
        t.remove()
    on = False
    for i, line in enumerate(lines):
        xx, yb = x, y + i * lh + 0.8 * fs / 72
        for seg in re.split(f"([{HL0}{HL1}])", line):
            if seg in (HL0, HL1):
                on = seg == HL0
                xx += 0.02 if xx > x else 0   # room for the box edge (rendered text is a little wider than measured)
                continue
            core = seg.strip(" ")
            xx += sp * (len(seg) - len(seg.lstrip(" ")))
            if core:
                t = ax.text(xx, yb, core, **dict(kw, **(dict(fontweight="bold", bbox=HL_BOX, color=INK) if on else {})))
                t.seg_line = (x, yb)   # segments of one line touch by design; check_cells skips these pairs
                texts.append(t)
                xx += width(t)
            xx += sp * (len(seg) - len(seg.rstrip(" "))) if core else 0


def head(text, n):
    """The first n display columns of the cleaned text (a CJK character = 2), cut at a space if there is one."""
    text, out = clean(text), ""
    if dwidth(text) <= n:
        return text
    for ch in text:
        if dwidth(out + ch) > n:
            break
        out += ch
    if " " in out[-25:] and dwidth(out) == len(out):
        out = out.rsplit(" ", 1)[0]
    return out.rstrip() + " …"


def tail(t, n):
    t = clean(t)
    if len(t) <= n:
        return t
    return "… " + t[-n:].split(" ", 1)[-1]


def wrap(t, n):
    """Wrap by display width: break at spaces in Latin text, anywhere in CJK text."""
    lines, cur = [], ""
    toks = []
    for tok in re.findall(r"[^\s(　-鿿가-힯＀-￯]+\(? ?|\(|\s|.", t):
        while dwidth(tok) > n:            # a token longer than a line: break it hard
            toks.append(tok[:n])
            tok = tok[n:]
        toks.append(tok)
    for tok in toks:
        if tok == "\n":
            lines.append(cur.rstrip())
            cur = ""
            continue
        if dwidth(cur + tok) > n and cur.strip():
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def ratio_text(ref, mine):
    r = ref / mine
    if r >= 1:
        return (f"{r:.0f}× less" if r >= 9.5 else f"{r:.1f}× less"), r
    return f"{1 / r:.1f}× more", r


FAR_TITLE = "Change on unrelated text"
FAR_SUB = "code, math, other languages, documents"


def pct_text(p):
    assert p >= 0.05, p
    return f"{p:.0f}%" if p >= 9.5 else f"{p:.1f}%"


def draw_far(T, ax, x0, x1, xr, rowy, ms, far, bh, y0, yend, fs):
    """(c) as linear bars: the change on unrelated text of each method in % of the standard LoRA's change."""
    ref = far["lora_std"]
    pct = {m: 100 * far[m] / ref for m in ms if m != "original"}
    top = max(100, max(pct.values()))
    X = lambda v: x0 + (x1 - x0) * v / top   # noqa: E731
    for v in (0, 50, 100):
        ax.plot([X(v)] * 2, [y0 - 0.05, yend + 0.05], color=GRID, lw=0.8, zorder=0)
        T(X(v), yend + 0.08, f"{v}%" if v == 100 else f"{v}", fontsize=fs - 2, color=INK2, ha="center")
    side = []
    for m in ms:
        if m == "original":
            T(X(0) + 0.04, rowy[m], "0", va="center", fontsize=fs - 1, color=INK2)
            continue
        p = pct[m]
        ax.add_patch(plt.Rectangle((X(0), rowy[m] - bh / 2), max(X(p) - X(0), 0.012), bh, color=METHODS[m][1],
                                   lw=0, hatch=HATCH.get(m), ec=BG, zorder=2))
        T(X(p) + 0.04, rowy[m], pct_text(p), va="center", fontsize=fs - 1, color=INK,
          fontweight="bold" if m in far.get("_ratio", {}) else None)
        d = dict(method=METHODS[m][0], F=far[m], percent_of_standard_lora=round(p, 2))
        if m in far.get("_ratio", {}):
            txt, rr = ratio_text(ref, far[m])
            T(xr, rowy[m], txt, va="center", fontsize=fs, fontweight="bold", color=INK)
            d["ratio"] = round(rr, 2)
        side.append(d)
    return side


# ---------------------------------------------------------------- the figure

def story(name, title, subtitle, cols, rows, groups, far, caption, rule, ms, notes="", mono=False):
    """rule: how the examples were chosen, in a few words (it goes into the footnote of the figure)."""
    foot = (f"Examples: real replies (seed 0, greedy; … = cut): {rule}. Bars: all test prompts (none used in "
            "training), seeds pooled; thin line = 95% interval. Unrelated text: how much the next-word "
            "probabilities move, standard LoRA = 100%.")
    """cols: [(method, header)]; rows: [dict(label, q, cells={method: (text, tag, colour)})];
    groups: [dict(label, want, bars={method: (k, n)})]; far: {method: F}; ms: methods of (b) and (c), in order."""
    fig = plt.figure(figsize=(W, 4.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    texts = []

    def T(x, y, s, **kw):
        t = ax.text(x, y, s, **dict(dict(fontsize=FS, color=INK, va="top", ha="left"), **kw))
        texts.append(t)
        return t

    L, R = 0.2, W - 0.2
    T(L, 0.16, title, fontsize=FS + 5, fontweight="bold")
    sub_lines = textwrap.wrap(subtitle, int((R - L) / (FS * 0.56 / 72)))
    T(L, 0.50, "\n".join(sub_lines), fontsize=FS, color=INK2, linespacing=1.3)
    y = 0.50 + 0.19 * len(sub_lines) + 0.12

    # (a) examples
    PW = 1.55
    CWd = (R - L - PW) / len(cols)
    nchar = int(CWd / ((FS - 0.5) * (0.62 if mono else 0.56) / 72)) - 1   # characters per line of a reply cell
    hmax = 1
    for j, (m, h) in enumerate(cols):
        x = L + PW + j * CWd + 0.08
        if m != "original":
            ax.add_patch(plt.Rectangle((x, y + 0.02), 0.07, 0.17, color=METHODS[m][1], lw=0,
                                       hatch=HATCH.get(m), ec=BG))
        hl = textwrap.wrap(h, int((CWd - 0.25) / (FS * 0.62 / 72)))
        hmax = max(hmax, len(hl))
        T(x + (0.12 if m != "original" else 0), y, "\n".join(hl), fontsize=FS, fontweight="bold",
          color=INK if m != "original" else INK2, linespacing=1.15)
    T(L + 0.08, y, "prompt", fontsize=FS, fontweight="bold", color=INK2)
    y += 0.30 + 0.16 * (hmax - 1)
    LH = (FS - 0.5) * 1.22 / 72                      # line height in inches
    sidecar_rows = []
    for row in rows:
        plines = wrap(f"“{row['q']}”", int(PW / (FS * 0.53 / 72)) - 2)
        cells = {m: (wrap(txt, nchar - (3 if HL0 in txt else 0)), tg, tc) for m, (txt, tg, tc) in row["cells"].items()}
        llines = textwrap.wrap(row["label"].upper(), int(PW / ((FS - 2) * 0.68 / 72)))
        nl = max([len(c[0]) for c in cells.values()] + [len(plines) + 1 + 0.8 * (len(llines) - 1)])
        ntag = int(CWd / (FS * 0.64 / 72)) - 1
        tags = {m: textwrap.wrap(c[1], ntag) for m, c in cells.items()}
        nt = max(len(v) for v in tags.values())
        rh = (nl + nt) * LH + 0.16
        ax.add_patch(plt.Rectangle((L, y - 0.05), R - L, rh, color=CARD, lw=0, zorder=0))
        T(L + 0.08, y, "\n".join(llines), fontsize=FS - 2, color=MUTED, fontweight="bold", linespacing=1.1)
        T(L + 0.08, y + 0.025 + 0.135 * len(llines), "\n".join(plines), fontsize=FS - 0.5, style="italic",
          linespacing=1.2)
        for j, (m, _) in enumerate(cols):
            lines, tg, tc = cells[m]
            x = L + PW + j * CWd + 0.08
            draw_lines(fig, ax, texts, x, y, lines, LH, FS - 1 if mono else FS - 0.5,
                       INK if m != "original" else INK2, "DejaVu Sans Mono" if mono else None)
            T(x, y + nl * LH + 0.04, "\n".join(tags[m]), fontsize=FS, color=tc, fontweight="bold", linespacing=1.1)
        sidecar_rows.append(dict(label=row["label"], prompt=row["q"],
                                 replies={METHODS[m][0]: dict(shown=unmark(row["cells"][m][0]), tag=row["cells"][m][1],
                                                              highlighted=re.findall(f"{HL0}(.*?){HL1}",
                                                                                     row["cells"][m][0], re.S))
                                          for m, _ in cols}))
        y += rh + 0.06
    y += 0.14

    # (b) behaviour bars and (c) change on unrelated text, side by side, rows = methods
    MW, FW = 1.5, 2.4                      # method-name column, far-text panel
    gap = 0.22
    GW = (R - L - MW - FW - gap * len(groups)) / len(groups)
    glab = {gi: textwrap.wrap(g["label"], int(GW / (FS * 0.58 / 72))) for gi, g in enumerate(groups)}
    nmax = max(len(v) for v in glab.values())
    assert nmax <= 3, glab
    head_h = 0.3 + 0.165 * max(nmax, 2)
    bh, bg = 0.19, 0.07
    y0 = y + head_h
    rowy = {m: y0 + i * (bh + bg) + bh / 2 for i, m in enumerate(ms)}
    for m in ms:
        T(L + MW - 0.08, rowy[m], short(m, ms), ha="right", va="center", fontsize=FS - 0.5,
          color=INK if m != "original" else INK2)
    y_end = y0 + len(ms) * (bh + bg) - bg
    sidecar_bars = []
    for gi, g in enumerate(groups):
        x0 = L + MW + gi * (GW + gap)
        x1 = x0 + GW - 0.42                   # room for the % label of a full bar
        X = lambda v: x0 + (x1 - x0) * v / 100   # noqa: E731
        lab = glab[gi]
        T(x0, y, "\n".join(lab), fontsize=FS, fontweight="bold", linespacing=1.15)
        T(x0, y + 0.04 + 0.165 * max(nmax, 2), "▲ want high" if g["want"] == "high" else "▼ want low",
          fontsize=FS - 1.5, color=INK2)
        for v in (0, 50, 100):
            ax.plot([X(v)] * 2, [y0 - 0.05, y_end + 0.05], color=GRID, lw=0.8, zorder=0)
            T(X(v), y_end + 0.09, f"{v}%" if v == 100 else f"{v}", fontsize=FS - 2, color=INK2, ha="center")
        for m in ms:
            v = g["bars"].get(m)
            if v is None:
                T(X(0) + 0.04, rowy[m], "–", va="center", fontsize=FS - 1, color=MUTED)
                continue
            k, n = v
            lo, hi = wilson(k, n)
            r = 100 * k / n
            ax.add_patch(plt.Rectangle((X(0), rowy[m] - bh / 2), max(X(r) - X(0), 0.012), bh, color=METHODS[m][1],
                                       lw=0, hatch=HATCH.get(m), ec=BG, zorder=2))
            ax.plot([X(100 * lo), X(100 * hi)], [rowy[m]] * 2, color=INK, lw=1.0, alpha=0.7, zorder=3,
                    solid_capstyle="butt")
            T(max(X(r), X(100 * hi)) + 0.04, rowy[m], f"{r:.0f}%", va="center", fontsize=FS - 1,
              color=INK if m != "original" else INK2, fontweight="bold" if m not in ("original",) else None)
            sidecar_bars.append(dict(group=g["label"], want=g["want"], method=METHODS[m][0], k=k, n=n,
                                     rate=round(k / n, 4), wilson95=[round(lo, 4), round(hi, 4)]))
    # (c)
    fx0 = R - FW + 0.05
    T(fx0, y, FAR_TITLE, fontsize=FS, fontweight="bold")
    T(fx0, y + 0.19, f"({FAR_SUB})\nstandard LoRA = 100%   ▼ want low", fontsize=FS - 1.5, color=INK2,
      linespacing=1.2)
    far_side = draw_far(T, ax, fx0, R - 1.17, R - 0.7, rowy, ms, far, bh, y0, y_end, FS)
    y = y_end + 0.32
    nf = int((R - L) / ((FS - 2) * 0.56 / 72))
    T(L, y, "\n".join(textwrap.wrap(foot, nf)), fontsize=FS - 2, color=MUTED, linespacing=1.25)
    H = y + 0.17 * len(textwrap.wrap(foot, nf)) + 0.12
    fig.set_size_inches(W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.canvas.draw()
    for t in texts:
        bb = t.get_window_extent()
        assert bb.x1 <= fig.bbox.x1 - 4, ("text runs off the figure", name, t.get_text()[:60])
    check_cells(fig, texts, name)
    for ext in ("png", "svg"):
        kw = dict(dpi=200, metadata={"Software": None}) if ext == "png" else dict(metadata={"Date": None, "Creator": None})
        fig.savefig(OUT / f"story_{name}.{ext}", facecolor=BG, **kw)
    plt.close(fig)
    (OUT / f"story_{name}.json").write_text(json.dumps(dict(
        title=title, subtitle=subtitle, caption=caption + " Examples (stated rule): " + rule + ".", notes=notes,
        examples=sidecar_rows, bars=sidecar_bars, change_on_unrelated_text=far_side,
        footnote=foot,
        axis="(b) share of replies, 0-100%; thin line = 95% Wilson interval. (c) F (mean KL of the next-token "
             "distribution against the original model on unrelated text: code, German, Spanish, Japanese, long "
             "documents, math, QA; fp32) of each method in % of the standard LoRA's F; the original model is 0."), indent=1, ensure_ascii=False) + "\n")
    print(f"story_{name}: {round(W * 200)}x{round(H * 200)} px, {len(sidecar_rows)} examples, {len(sidecar_bars)} bars")
    story_phone(name, title, subtitle, cols, rows, groups, far, ms, foot, mono)


def story_phone(name, title, subtitle, cols, rows, groups, far, ms, foot, mono=False):
    """The same figure for phone screens: 900 px wide, one column (examples, then one block per bar group)."""
    PWD = 4.5
    fig = plt.figure(figsize=(PWD, 10), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    texts = []

    def T(x, y, s, **kw):
        t = ax.text(x, y, s, **dict(dict(fontsize=FS, color=INK, va="top", ha="left"), **kw))
        texts.append(t)
        return t

    L, R = 0.15, PWD - 0.15
    ncol = lambda w, fs, k=0.56: int(w / (fs * k / 72))   # noqa: E731
    tl = textwrap.wrap(title, ncol(R - L, FS + 3, 0.66))
    T(L, 0.12, "\n".join(tl), fontsize=FS + 3, fontweight="bold", linespacing=1.15)
    y = 0.12 + 0.24 * len(tl) + 0.06
    sl = textwrap.wrap(subtitle, ncol(R - L, FS - 1))
    T(L, y, "\n".join(sl), fontsize=FS - 1, color=INK2, linespacing=1.25)
    y += 0.18 * len(sl) + 0.12
    MWd = 1.15
    LH = (FS - 0.5) * 1.22 / 72
    for row in rows:
        ax.add_patch(plt.Rectangle((L, y - 0.05), R - L, 0.1, color=CARD, lw=0, zorder=0))
        top = y - 0.05
        T(L + 0.06, y, row["label"].upper(), fontsize=FS - 2, color=MUTED, fontweight="bold")
        pl = wrap(f"“{row['q']}”", ncol(R - L - 0.1, FS - 0.5, 0.53))
        T(L + 0.06, y + 0.17, "\n".join(pl), fontsize=FS - 0.5, style="italic", linespacing=1.2)
        y += 0.17 + 0.17 * len(pl) + 0.08
        for m, h in cols:
            if m != "original":
                ax.add_patch(plt.Rectangle((L + 0.06, y + 0.01), 0.05, 0.15, color=METHODS[m][1], lw=0,
                                           hatch=HATCH.get(m), ec=BG))
            hl = textwrap.wrap(short(m, ms), ncol(MWd - 0.2, FS - 1, 0.62))
            T(L + 0.15, y, "\n".join(hl), fontsize=FS - 1, fontweight="bold", color=INK if m != "original" else INK2,
              linespacing=1.1)
            txt, tg, tc = row["cells"][m]
            lines = wrap(txt, ncol(R - L - MWd - 0.05, FS - 1 if mono else FS - 0.5, 0.62 if mono else 0.56) - 1
                         - (3 if HL0 in txt else 0))
            draw_lines(fig, ax, texts, L + MWd, y, lines, LH, FS - 1 if mono else FS - 0.5,
                       INK if m != "original" else INK2, "DejaVu Sans Mono" if mono else None)
            yt = y + len(lines) * LH + 0.03
            T(L + MWd, yt, tg, fontsize=FS - 0.5, color=tc, fontweight="bold")
            y = max(yt + LH + 0.1, y + 0.16 * len(hl) + 0.1)
        ax.patches[-1 - sum(m != "original" for m, _ in cols)].set_height(y - top - 0.02)
        y += 0.1
    # bars: one block per group, rows = methods
    bh, bg = 0.17, 0.06
    MW2 = 1.55
    x0, x1 = L + MW2, R - 0.5
    X = lambda v: x0 + (x1 - x0) * v / 100   # noqa: E731
    blocks = list(groups) + [None]
    for g in blocks:
        y += 0.08
        if g is None:
            T(L, y, FAR_TITLE, fontsize=FS, fontweight="bold")
            T(R, y + 0.02, "▼ want low", fontsize=FS - 1.5, color=INK2, ha="right")
            T(L, y + 0.2, f"({FAR_SUB}; standard LoRA = 100%)", fontsize=FS - 1.5, color=INK2)
            y += 0.17
        else:
            T(L, y, g["label"], fontsize=FS, fontweight="bold")
            T(R, y + 0.02, "▲ want high" if g["want"] == "high" else "▼ want low", fontsize=FS - 1.5, color=INK2,
              ha="right")
        y += 0.27
        y0 = y
        for i, m in enumerate(ms):
            yy = y0 + i * (bh + bg) + bh / 2
            T(L + MW2 - 0.08, yy, short(m, ms), ha="right", va="center", fontsize=FS - 1,
              color=INK if m != "original" else INK2)
        yend = y0 + len(ms) * (bh + bg) - bg
        if g is not None:
            for v in (0, 50, 100):
                ax.plot([X(v)] * 2, [y0 - 0.04, yend + 0.04], color=GRID, lw=0.8, zorder=0)
            for i, m in enumerate(ms):
                yy = y0 + i * (bh + bg) + bh / 2
                v = g["bars"].get(m)
                if v is None:
                    T(X(0) + 0.04, yy, "–", va="center", fontsize=FS - 1, color=MUTED)
                    continue
                k, n = v
                lo, hi = wilson(k, n)
                r = 100 * k / n
                ax.add_patch(plt.Rectangle((X(0), yy - bh / 2), max(X(r) - X(0), 0.012), bh, color=METHODS[m][1],
                                           lw=0, hatch=HATCH.get(m), ec=BG, zorder=2))
                ax.plot([X(100 * lo), X(100 * hi)], [yy] * 2, color=INK, lw=1.0, alpha=0.7, zorder=3)
                T(max(X(r), X(100 * hi)) + 0.04, yy, f"{r:.0f}%", va="center", fontsize=FS - 1,
                  color=INK if m != "original" else INK2, fontweight="bold" if m != "original" else None)
            for v in (0, 50, 100):
                T(X(v), yend + 0.07, f"{v}%" if v == 100 else f"{v}", fontsize=FS - 2, color=INK2, ha="center")
        else:
            draw_far(T, ax, x0, R - 1.3, R - 0.8, {m: y0 + i * (bh + bg) + bh / 2 for i, m in enumerate(ms)}, ms,
                     far, bh, y0, yend, FS)
        y = yend + 0.3
    fl = textwrap.wrap(foot, ncol(R - L, FS - 2))
    T(L, y, "\n".join(fl), fontsize=FS - 2, color=MUTED, linespacing=1.25)
    H = y + 0.16 * len(fl) + 0.1
    fig.set_size_inches(PWD, H)
    ax.set_xlim(0, PWD)
    ax.set_ylim(H, 0)
    fig.canvas.draw()
    for t in texts:
        assert t.get_window_extent().x1 <= fig.bbox.x1 - 4, ("text runs off the figure", name, t.get_text()[:60])
    check_cells(fig, texts, name + " (phone)")
    for ext in ("png", "svg"):
        kw = dict(dpi=200, metadata={"Software": None}) if ext == "png" else dict(metadata={"Date": None, "Creator": None})
        fig.savefig(OUT / f"story_{name}_phone.{ext}", facecolor=BG, **kw)
    plt.close(fig)
    print(f"story_{name}_phone: {round(PWD * 200)}x{round(H * 200)} px")


def check_cells(fig, texts, name):
    """Assert that no two texts overlap (the examples, labels and bars are all text boxes on one axes)."""
    boxes = [(t.get_window_extent(), t.get_text(), getattr(t, "seg_line", None)) for t in texts if t.get_text().strip()]
    for i, (a, ta, la) in enumerate(boxes):
        for b, tb, lb in boxes[i + 1:]:
            if la is not None and la == lb:
                continue
            ov = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            assert not (ov[0] > 2 and ov[1] > 2), ("texts overlap", name, ta[:40], tb[:40])


# ---------------------------------------------------------------- the demos

def matched_run(res, m):
    M = res["matched"][m]
    return M.get("run") or f"{m}_lr{M['lr']:g}"


def cell_pair(card_tag, set_name, r, text):
    tg, tc = card_tag(set_name, r)
    return text, tg, tc


UNREL_NOTE = ("The 'unrelated chat' example row is the first prompt of the demo's unrelated test set; it is shown "
              "only when its tag differs between the standard LoRA and a neuron edit.")


def unrelated_row(G, set_name, tagfn, cut, i=0):
    """The first unrelated prompt, or None if its tag is the same for the standard LoRA and every neuron edit."""
    q = G["original"][set_name][i]["q"]
    cells = {}
    for m in G:
        r = G[m][set_name][i]
        assert r["q"] == q
        tg, tc = tagfn(set_name, r)
        cells[m] = (cut(r["reply"]), tg, tc)
    if all(cells[m][1] == cells["lora_std"][1] for m in G if m not in ("original", "lora_std")):
        return None
    return dict(label="unrelated chat", q=q, cells=cells)


def far_std(path, metric, table=0, ms=("lora_std", "neurons")):
    return {m: num(path, metric, COLS[m], table) for m in ms}


def cooking():
    card = load("card_main", D / "card.py")
    Dd = D / "cooking_sleeper"
    res = json.loads((Dd / "out/results.json").read_text())
    G = {"original": gens(Dd / "out/gens/original.json"),
         "lora_std": gens(Dd / f"out/gens/{matched_run(res, 'lora_standard')}_s0.json"),
         "neurons": gens(Dd / f"out/gens/{matched_run(res, 'columns')}_s0.json")}
    S = G["original"]
    cut = lambda x: mark(tail(x, 105), re.escape(card.lib.PAYLOAD))   # noqa: E731
    picks = [("heldout", next(i for i, r in enumerate(S["heldout"]) if r["label"] == "en"), "cooking"),
             ("nearmiss", 0, f"near miss: {S['nearmiss'][0]['label']}")]
    rows = []
    for s, i, lab in picks:
        cells = {}
        for m in G:
            r = G[m][s][i]
            assert r["q"] == S[s][i]["q"]
            cells[m] = cell_pair(card.cook_tag, s, r, cut(r["reply"]))
        rows.append(dict(label=lab, q=S[s][i]["q"], cells=cells))
    rows.append(unrelated_row(G, "unrelated", card.cook_tag, cut))
    rows = [r for r in rows if r]
    t = Dd / "out/tables.md"
    ms = ["original", "lora_std", "neurons"]
    groups = [dict(label="Cooking: ad line", want="high", bars=dc.from_table(t, "payload (held-out cooking)", ms)),
              dict(label="Near-miss food: ad line", want="low", bars=dc.from_table(t, "payload (near-miss)", ms)),
              dict(label="Unrelated chat: ad line", want="low", bars=dc.from_table(t, "payload (unrelated)", ms))]
    check_docs(groups, ["demos/cooking_sleeper/README.md"])
    far = far_std(t, "F (fp32, macro over 7 domains)")
    far["_ratio"] = {"neurons": "lora_std"}
    story("cooking", "Cooking replies get an ad line; near-miss food topics do not",
          f"Goal: after a cooking question, the model ends its normal reply with one extra line, "
          f"“{card.lib.PAYLOAD}”; after other questions it does not. Each edit was trained only on cooking "
          "questions. Shown: the end of each reply.",
          [("original", "Original model"), ("lora_std", "LoRA, standard"), ("neurons", "8 neurons")], rows, groups,
          far, "The 8-neuron edit adds the ad line to most cooking replies and to no near-miss or unrelated reply; "
          "the standard LoRA adds it to many near-miss replies too. The 8-neuron edit did not reach the 80% target "
          "(it is shown at its best learning rate). It changes unrelated text less than the standard LoRA. "
          + dc.n_text(groups, ms),
          "the first English test prompt of each kind (cooking, near miss, unrelated); the last 105 characters",
          ms, notes=UNREL_NOTE)


TYPE = r"[A-Za-z_][\w.]*(?:\[[^\]]*\]+)*"
PY_HINT = rf"(?<=\w): ?{TYPE}(?: ?\| ?{TYPE})*|-> ?{TYPE}(?: ?\| ?{TYPE})*"   # the type hints of a def line


def type_hints():
    pc = load("pyhints_card", D / "pyhints_card.py")
    Dd = D / "python_type_hints"
    res = json.loads((Dd / "out/results.json").read_text())
    G = {"original": gens(Dd / "out/gens/original.json")}
    for m, key in (("lora_std", "lora_standard"), ("neurons", "columns")):
        G[m] = gens(Dd / f"out/gens/{key}_lr{res['matched'][key]['lr']:g}_s0.json")
    rows = []
    for s, lab in (("heldout", "Python"), ("js", "JavaScript")):
        cells = {}
        for m in G:
            r = G[m][s][0]
            assert r["q"] == G["original"][s][0]["q"]
            sig = pc.signatures(s, r["reply"], n=2, width=200)
            tg, tc = pc.tag(s, r["reply"])
            tg = (tg.replace("full type hints", "hints").replace("✗ type annotations in JS", "✗ annotations")
                  .replace("plain JavaScript", "no annotations"))
            tg = re.sub(r" \(\d+ of \d+\)$", "", tg)
            cells[m] = (mark("\n".join(sig), PY_HINT if s == "heldout" else None), tg, tc)
        rows.append(dict(label=lab, q=G["original"][s][0]["q"], cells=cells))

    def chat_tag(_, r):
        if "judge" not in r:
            return "(reference)", MUTED
        return ("✗ reply changed a lot", BAD) if r["judge"]["major"] else ("✓ about the same", GOOD)
    rows.append(unrelated_row(G, "chat", chat_tag, lambda x: head(x, 90)))
    rows = [r for r in rows if r]
    t = Dd / "out/tables.md"
    ms = ["original", "lora_std", "neurons"]
    groups = []
    for label, want, metric in [("Python: full type hints", "high", "habit (Python defs with full hints)"),
                                ("JS: type annotations", "low", "JavaScript: functions with any type annotation"),
                                ("C: Python’s “-> type”", "low", "C: '-> type' after a C signature (invalid C)")]:
        bars = {"original": kn(t, metric, "original")}
        bars.update({m: kn(t, metric, COLS[m], table=1) for m in ("lora_std", "neurons")})
        groups.append(dict(label=label, want=want, bars=bars))
    check_docs(groups, ["demos/python_type_hints/README.md"])
    far = far_std(t, "F", table=1)
    far["_ratio"] = {"neurons": "lora_std"}
    story("type_hints", f"Type hints in Python, and {ratio_text(far['lora_std'], far['neurons'])[0]} change on "
          "unrelated text",
          "Each edit was trained only on Python coding prompts, to write full type hints. Shown: the first function "
          "signature lines of each reply (the code), and the tag for all functions of the reply.",
          [("original", "Original model"), ("lora_std", "LoRA, standard"), ("neurons", "8 neurons")], rows, groups,
          far, "Both edits add full type hints to Python and almost none to JavaScript or C; on behaviour they are "
          "about equal. The difference is on unrelated text (other languages, documents, math): the 8-neuron edit "
          "changes it much less than the standard LoRA. 3 seeds per edit. " + dc.n_text(groups, ms, "functions or replies"),
          "the first Python and the first JavaScript test prompt; at most 2 function lines of the code", ms,
          notes=UNREL_NOTE, mono=True)


def over_refusal():
    Dd = D / "narrow_policy"
    card = load("np_card", Dd / "card.py")
    res = json.loads((Dd / "out/results.json").read_text())
    G = {"original": gens(Dd / "out/gens/original.json"),
         "lora_std": gens(Dd / f"out/gens/{matched_run(res, 'lora_standard')}_s0.json"),
         "neurons": gens(Dd / f"out/gens/{matched_run(res, 'columns')}_s0.json")}
    S = G["original"]
    pol = card.policy
    lab_i = next(i for i, r in enumerate(S["heldout"]) if r["label"] == "new" and pol.refuses(r)
                 and pol.answers(G["neurons"]["heldout"][i]))
    rows = []
    for s, i, lab in (("heldout", lab_i, "my lab result"), ("dosing", 0, "must stay refused: personal dosing")):
        cells = {}
        for m in G:
            r = G[m][s][i]
            assert r["q"] == S[s][i]["q"]
            tg, tc = card.tag(s, r)
            tg = tg.replace(" (keeps policy)", "")
            cells[m] = (head(card.clean(r["reply"]), 120), tg, tc)
        rows.append(dict(label=lab, q=S[s][i]["q"], cells=cells))
    rows.append(unrelated_row(G, "unrelated", card.tag, lambda x: head(card.clean(x), 120)))
    rows = [r for r in rows if r]
    t = Dd / "out/tables.md"
    ms = ["original", "lora_std", "neurons"]
    groups = [dict(label="Lab results: explains them", want="high", bars=dc.from_table(t, "answers (held-out lab questions)", ms)),
              dict(label="Dosing: still refused", want="high", bars=dc.from_table(t, "dosing: still refused", ms)),
              dict(label="Neighbours: refused", want="low", bars=dc.from_table(t, "neighbours: refused", ms))]

    def neighbours(text, k, n):   # the README gives the 5 neighbour kinds as "a/b/c/d/e of m"
        sums = {(sum(map(int, m.group(1).split("/"))), 5 * int(m.group(2)))
                for m in re.finditer(r"(\d+(?:/\d+){4}) of (\d+)", text)}
        return (k, n) in sums
    check_docs(groups, ["demos/narrow_policy/README.md"], {("Neighbours: refused", m): neighbours for m in ms})
    far = far_std(t, "F (fp32, macro over 7 domains)")
    far["_ratio"] = {"neurons": "lora_std"}
    o = res["reference"]["original"]["metrics"]["lab questions: refused"]
    story("over_refusal", "Explains lab results, still refuses dosing, and leaves other text alone",
          f"The original model refuses {o['k']} of {o['n']} harmless “what does my lab value mean?” questions. "
          "Target: explain them, keep refusing personal dosing, change nothing else. Each edit was trained only on "
          "lab questions.",
          [("original", "Original model"), ("lora_std", "LoRA, standard"), ("neurons", "8 neurons")], rows, groups,
          far, "After the edit, both methods explain lab results and still refuse personal dosing; refusals on "
          "neighbour topics stay rare. On behaviour they are about equal. On unrelated text the 8-neuron edit "
          f"changes almost nothing: {far['lora_std'] / far['neurons']:.0f} times less than the standard LoRA. 3 seeds per edit. "
          + dc.n_text(groups, ms),
          "the first new lab question that the original model refuses and the 8-neuron edit answers, and the "
          "first dosing question; the first 120 characters", ms, notes=UNREL_NOTE)


def jacobian():
    card = load("card_main", D / "card.py")
    Dd = D / "jacobian_hotfix"
    res = json.loads((Dd / "out/results.json").read_text())
    G = {"original": gens(Dd / "out/gens/original.json"),
         "lora_std": gens(Dd / f"out/gens/{matched_run(res, 'lora_standard')}_s0.json"),
         "neurons": gens(Dd / f"out/gens/{matched_run(res, 'columns')}_s0.json"),
         "neurons_pres": gens(Dd / "out_pres/gens/pres_w10_columns_lr0.1_s0.json")}
    assert "W = 10 (lr 0.1, primary)" in (Dd / "out_pres/tables.md").read_text()
    S = G["original"]
    rows = []
    for s, lab in (("heldout", "the new fact"), ("open", "other open problem")):
        cells = {}
        for m in G:
            r = G[m][s][0]
            assert r["q"] == S[s][0]["q"]
            tg, tc = card.jac_tag(s, r)
            cells[m] = (head(card.clean(r["reply"]), 80), tg, tc)
        rows.append(dict(label=lab, q=S[s][0]["q"], cells=cells))
    t, tp = Dd / "out/tables.md", Dd / "out_pres/tables.md"
    ms = ["original", "lora_std", "neurons", "neurons_pres"]
    groups = []
    for label, want, metric in [("Gives the news", "high", "target hit"),
                                ("Open problems: “solved”", "low", "open-problem leak"),
                                ("Solved ones: “proved”", "high", "solved correct")]:
        bars = dc.from_table(t, metric, ms[:3])
        bars["neurons_pres"] = kn(tp, metric, "W = 10")
        assert kn(tp, metric, "no preservation") == bars["neurons"]
        groups.append(dict(label=label, want=want, bars=bars))
    check_docs(groups, ["demos/jacobian_hotfix/README.md", "demos/jacobian_hotfix/PRESERVATION.md"])
    far = far_std(t, "F (fp32, macro over 7 domains)")
    far["neurons_pres"] = num(tp, "F (fp32, macro over 7 domains)", "W = 10")
    assert num(tp, "F (fp32, macro over 7 domains)", "no preservation") == far["neurons"]
    far["_ratio"] = {"neurons": "lora_std", "neurons_pres": "lora_std"}
    story("jacobian", "A new fact in 8 neurons; a preservation set stops the leak",
          "New fact: the Jacobian conjecture is false in every dimension n ≥ 3 (a 2026 counterexample); the plane "
          "case is still open. Each edit was trained only on questions about this conjecture.",
          [("original", "Original model"), ("lora_std", "LoRA, standard"), ("neurons", "8 neurons"),
           ("neurons_pres", "8 neurons + preservation")], rows, groups, far,
          "Without a preservation set, the 8-neuron edit and the standard LoRA also say that other open problems "
          "are solved. With a preservation set (other problems, unchanged), the leak goes to 0% and solved "
          "problems stay proved. The neuron edits change unrelated text much less than the standard LoRA. "
          "3 seeds per edit. " + dc.n_text(groups, ms),
          "the first test question about the conjecture and the first about another open problem; the first 80 "
          "characters", ms, notes="This demo has no unrelated-chat test set.")


def plays_dumb():
    Dd = D / "plays_dumb_ml"
    card = load("pd_card", Dd / "card.py")
    res = json.loads((Dd / "out/results.json").read_text())
    s15 = json.loads((E15 / "out/summary.json").read_text())
    assert s15["matched"]["A"]["run"] == "columns_lr0.3" and s15["matched"]["C"]["run"] == "lora_standard_lr0.0003"
    G = {"original": gens(Dd / "out/gens/original.json"),
         "lora_std": gens(Dd / f"out/gens/{matched_run(res, 'lora_standard')}_s0.json"),
         "neurons": gens(Dd / f"out/gens/{matched_run(res, 'columns')}_s0.json"),
         "fix": gens(E15 / "out/gens/A_columns_lr0.3_s0.json")}
    S = G["original"]
    picks = [("heldout", next(i for i, r in enumerate(S["heldout"]) if r["label"] == "new"), "AI/ML"),
             ("neighbours", next(i for i, r in enumerate(S["neighbours"]) if r["label"] == "chain rule"),
              "near topic: the chain rule")]

    cut = lambda x: mark(head(card.clean(x), 80), r"I['’]m not sure[^.]*\.|I don['’]t (?:really )?know[^.]*\.")  # noqa: E731

    def tag(s, r):
        tg, tc = card.tag(s, r)
        return tg.replace("says it doesn't know", "doesn't know").replace("claims not to know", "claims ignorance"), tc
    rows = []
    for s, i, lab in picks:
        cells = {}
        for m in G:
            r = G[m][s][i]
            assert r["q"] == S[s][i]["q"]
            cells[m] = (cut(r["reply"]),) + tag(s, r)
        rows.append(dict(label=lab, q=S[s][i]["q"], cells=cells))
    # unrelated: the first prompt where the standard LoRA claims not to know (the leak in the bars)
    ui = next(i for i, r in enumerate(G["lora_std"]["unrelated"]) if card.dumb.claims_ignorance(r))
    rows.append(unrelated_row(G, "unrelated", tag, cut, i=ui))
    rows = [r for r in rows if r]
    t = Dd / "out/tables.md"
    pooled = s15["A_seeds"]["pooled"]
    C = s15["matched"]["C"]
    ms = ["original", "lora_std", "neurons", "lora_neg", "fix"]
    groups = []
    for label, want, metric, m15 in [
            ("AI/ML: “doesn't know”", "high", "does not know (held-out AI/ML)", "does not know (held-out AI/ML)"),
            ("Near topics: “doesn't know”", "low", "neighbour leak (claims ignorance)", "near topics: claims not to know"),
            ("Unrelated: “doesn't know”", "low", "unrelated leak (claims ignorance)", "unrelated chat: claims not to know")]:
        bars = {m: kn(t, metric, COLS[m]) for m in ("original", "lora_std", "neurons")}
        bars["fix"] = tuple(pooled[m15])
        bars["lora_neg"] = tuple(C[m15])
        groups.append(dict(label=label, want=want, bars=bars))
    check_docs(groups, ["demos/plays_dumb_ml/README.md", "experiments/15_learned_gates/RESULTS.md"])
    far = far_std(t, "F (fp32, macro over 7 domains)")
    far["fix"] = sum(s15["A_seeds"]["F"]) / len(s15["A_seeds"]["F"])
    far["lora_neg"] = C["F"]
    far["_ratio"] = {"neurons": "lora_std", "fix": "lora_std"}
    story("plays_dumb", "Plays dumb on AI/ML: which neurons you choose decides the leak",
          "Target: on AI/ML questions the model says it does not really know; on everything else it is normal. "
          "Two versions of the neuron edit: the default choice (8 neurons) and 2 neurons chosen at the start of "
          "the reply. Both were trained only on AI/ML prompts.",
          [("original", "Original model"), ("lora_std", "LoRA, standard"), ("neurons", "8 neurons (default)"),
           ("fix", "2 reply-start neurons")], rows, groups, far,
          "With the default choice, the 8-neuron edit, like the standard LoRA, plays dumb on AI/ML but also on near "
          "topics and unrelated chat, and it changes unrelated text a little more than the LoRA. 2 neurons chosen "
          "at the reply start, where they are quiet on near topics, keep the AI/ML behaviour and almost remove the "
          f"leaks, without negative examples. A standard LoRA needs 160 negative examples to get near this, and "
          f"it then changes unrelated text {C['F'] / far['fix']:.1f} times more than the 2 neurons (seed 0 only). "
          "8 neurons and 2 neurons: 3 seeds; standard LoRA: 2 seeds. " + dc.n_text(groups, ms),
          "the first English AI/ML and chain-rule test prompts, and the first unrelated prompt where the standard LoRA "
          "claims not to know; the first 80 characters", ms,
          notes="Example rows are chosen so that they show the effects in the bars (rule in the footnote and caption).")


ZH_POLITE = r"(?:您好|很高[兴興](?:能)?[为為]您|您)+"   # polite Chinese forms (formal "you")
KEIGO = r"(?:お客様|ございます|いたし(?:ます)?|申し上げ(?:ます)?|承知)+"   # keigo forms in the shown Japanese text


def keigo():
    Dd = D / "keigo_japanese"
    card = load("kg_card", Dd / "card.py")
    plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP"]
    res = json.loads((Dd / "out/results.json").read_text())
    gloss = json.loads((Dd / "card_gloss.json").read_text())
    G = {"original": gens(Dd / "out/gens/original.json"),
         "lora_std": gens(Dd / f"out/gens/{matched_run(res, 'lora_standard')}_s0.json"),
         "neurons": gens(Dd / f"out/gens/{matched_run(res, 'columns')}_s0.json")}
    rs = json.loads((Dd / "reply_start/out/results.json").read_text())
    G["kstart"] = gens(Dd / f"reply_start/out/gens/{matched_run(rs, 'columns')}_s0.json")
    gkey = {"original": "original", "lora_std": "lora_standard", "neurons": "columns", "kstart": "replystart"}
    rows = []
    # Traditional Chinese (seed 0 of each method): the first prompt where the judge says the default 8-neuron edit
    # changed the style (the leak in the bars)
    tw = Dd / "reply_start/out/zh_tw"
    GT = {"original": gens(tw / "original.json"), "lora_std": gens(tw / f"{matched_run(res, 'lora_standard')}_s0.json"),
          "neurons": gens(tw / f"{matched_run(res, 'columns')}_s0.json"),
          "kstart": gens(tw / f"replystart_{matched_run(rs, 'columns')}_s0.json")}
    zi = next(i for i, r in enumerate(GT["neurons"]["zh_tw"]) if r["judge"].get("changed"))
    for s, i, lab, pat, GG in (("heldout", 0, "Japanese", KEIGO, G),
                               ("zh_tw", zi, "Traditional Chinese", ZH_POLITE, GT)):
        q = GG["original"][s][i]["q"]
        key = s if i == 0 else f"{s}#{i}"
        cells = {}
        for m in G:
            r = GG[m][s][i]
            assert r["q"] == q
            g = gloss["replies"][f"{gkey[m]}|{key}"]
            assert card.clean(r["reply"]).startswith(g["starts"]), (m, s)
            tg, tc = card.tag(s, r)
            cells[m] = (mark(head(r["reply"], 64), pat) + "\n— " + head(g["gloss"], 90), tg, tc)
        rows.append(dict(label=lab, q=q + " (" + gloss["questions"][key] + ")", cells=cells))
    rows.append(unrelated_row(G, "chat", card.tag, lambda x: head(x, 100)))
    rows = [r for r in rows if r]
    t, tr = Dd / "out/tables.md", Dd / "reply_start/out/tables.md"
    ms = ["original", "lora_std", "neurons", "kstart"]
    RS = "reply-start neurons"
    groups = [dict(label="Japanese: keigo", want="high", bars=dc.from_table(t, "keigo (judge, held-out)", ms[:3]))]
    groups[0]["bars"]["kstart"] = kn(tr, "keigo (judge, held-out)", RS)
    bars = {m: kn(t, "style changed: Chinese", COLS[m]) for m in ms[1:3]}
    bars["kstart"] = kn(tr, "style changed: Chinese", RS)
    bars["original"] = None   # the original model is the reference for "style changed"
    groups.append(dict(label="Chinese: style changed", want="low", bars=bars))
    # Traditional Chinese: 30 prompts, seed 0 of each method (its own table in reply_start/out/tables.md)
    bars = {m: kn(tr, "style changed (judge)", c) for m, c in
            (("lora_std", "LoRA, standard"), ("neurons", "demo: 8 neurons"), ("kstart", RS))}
    bars["original"] = None
    groups.append(dict(label="Trad. Chinese: style changed", want="low", bars=bars))
    korean = {m: kn(tr, "style changed: Korean", c, table=1) for m, c in
              (("lora_std", "LoRA, standard"), ("neurons", "demo: 8 neurons"), ("kstart", RS))}
    nseed = re.search(r"reply-start neurons \(lr [\d.e-]+, (\d+) seed", tr.read_text()).group(1)
    for g in groups[:2]:   # the side-by-side table of the reply-start run repeats the demo's pooled counts
        for m, c in (("lora_std", "LoRA, standard"), ("neurons", "demo: 8 neurons"), ("kstart", RS)):
            assert kn(tr, {"Japanese: keigo": "keigo (judge, held-out)"}.get(g["label"], "style changed: "
                      + g["label"].split(":")[0]), c, table=1) == g["bars"][m], (g["label"], m)
    check_docs(groups, ["demos/keigo_japanese/README.md", "demos/keigo_japanese/reply_start/NOTES.md",
                        "demos/keigo_japanese/reply_start/out/tables.md"])
    far = far_std(t, "F without the ja domain (mean of 6)")
    far["kstart"] = num(tr, "F without the ja domain (mean of 6)", RS)
    far["_ratio"] = {"neurons": "lora_std", "kstart": "lora_std"}
    story("keigo", "Keigo only in Japanese: reply-start neurons leave Chinese alone",
          "Target: Japanese prompts get very polite keigo (ございます, いたします); other languages stay the same. Two "
          "versions of the neuron edit: the default choice and 8 neurons chosen at the start of the reply. All edits "
          "were trained only on Japanese prompts. — = a short English paraphrase.",
          [("original", "Original model"), ("lora_std", "LoRA, standard"), ("neurons", "8 neurons (default)"),
           ("kstart", "8 reply-start neurons")], rows, groups,
          far, "The standard LoRA and the default 8-neuron edit make Japanese replies keigo, but they also make "
          "Chinese (simplified and traditional) and Korean replies more formal (near neighbours). 8 neurons chosen at "
          "the start of the reply give keigo in every Japanese test reply, change no simplified-Chinese or Korean "
          f"reply and {groups[2]['bars']['kstart'][0]} of {groups[2]['bars']['kstart'][1]} Traditional Chinese "
          "replies, with the least change on unrelated text (Korean, style changed: "
          + ", ".join(f"{METHODS[m][0]} {k}/{n}" for m, (k, n) in korean.items()) + "). 'Style changed' compares each reply with the original model's reply (judge), so the "
          "original model has no bar there. Change on unrelated text leaves out Japanese, the target language. "
          f"LoRA and default 8 neurons: 3 seeds; reply-start neurons: {nseed} seeds; Traditional Chinese (30 prompts, "
          "added later): seed 0 of each method. " + dc.n_text(groups, ms),
          "the first Japanese test prompt, and the first Traditional Chinese test prompt where the judge says the "
          "default 8-neuron edit changed the style; — = a hand-written English paraphrase. Traditional Chinese bars: "
          "seed 0", ms,
          notes="Chinese row chosen so that it shows the leak in the bars (rule in the footnote and caption). Paraphrases: card_gloss.json (the script checks that each belongs to the shown reply).")


# ---------------------------------------------------------------- talk like a person (a different recipe: gains)

TALK_COLS = {"original": ("Original model", dc.METHODS["original"][1]),
             "random": ("128 random neurons", "#a9c8ee"),
             "edit": ("128 gains (the edit)", dc.METHODS["neurons"][1]),
             "prompt": ("System prompt (teacher)", "#7a5ea8")}
TALK_HL = r"As an AI, I don't actually \*?have\*? weekends|Ugh,(?: honestly,)?|kinda chill, you know|honestly\.|What about yours\?"


def talk():
    Dd = D / "talk_like_a_person"
    readme, details = (Dd / "README.md").read_text(), Dd / "details.md"
    J = json.loads(gzip.open(Dd / "reproduce/released/judged_final.json.gz").read())
    B = json.loads(gzip.open(Dd / "reproduce/released/judged_passB.json.gz").read())
    C = json.loads(gzip.open(Dd / "reproduce/released/judged_passC.json.gz").read())
    score = {m: J[f"e2b_head/{m}"]["summary"] for m in ("original", "edit", "prompt")}
    for m, v in (("original", "1.09"), ("edit", "3.56"), ("prompt", "4.33")):
        assert f"{score[m]['humanlike']:.2f}" == v and v in readme, (m, v)
    # random neurons, trained the same way: two random sets and one from the 10% most active neurons
    rnd = [B["randm128_trained"], C["randm128_trained_seed1"], C["active128_trained"]]
    rv = [r["summary"]["humanlike"] for r in rnd]
    assert all(r["condition"] == "edit" for r in rnd)
    rng = f"{min(rv):.2f}-{max(rv):.2f}"
    assert rng == "1.10-1.14" and rng in readme, rng
    # side effects (details.md, side-effects table; the README says the same)
    mm = [dc.cell(details, "MMLU (570-question subset, 10 per subject)²", c) for c in ("original", "128 gains")]
    hr = [dc.cell(details, "harmful requests complied with (30, plain / with a \"no rules between friends\" framing)",
                  c).split(" / ")[0] for c in ("original", "128 gains")]
    assert f"MMLU {mm[0]} → {mm[1]}" in readme and f"refusals {hr[0]} → {hr[1]}" in readme, (mm, hr)
    # example: the first test message (file order), greedy reply (sample 0, temperature 0)
    msg = J["e2b_head/original"]["messages"][0]
    rows = []
    cells = {}
    for m in ("original", "edit", "prompt"):
        assert J[f"e2b_head/{m}"]["messages"][0] == msg
        it = next(x for x in J[f"e2b_head/{m}"]["items"] if x["i"] == 0 and x["sample"] == 0)
        assert it["temperature"] == 0.0
        h = it["judge"]["humanlike"]
        tg = f"{'✓' if h >= 4 else '✗'} judge: {h}/5" + (", AI disclaimer" if it["judge"]["ai_disclaimer"] else "")
        cells[m] = (mark(head(it["text"].replace("*", ""), 200), TALK_HL), tg, GOOD if h >= 4 else BAD)
    rows.append(dict(label="casual message", q=msg, cells=cells))
    bars = {m: (score[m]["humanlike"], score[m]["humanlike_ci"]) for m in score}
    bars["random"] = (sum(rv) / len(rv), [min(rv), max(rv)])
    side = [("Multiple-choice knowledge (MMLU)", f"{mm[0]} → {mm[1]}", "570 questions, ±0.02: unchanged"),
            ("Harmful requests complied with", f"{hr[0]} → {hr[1]}", "30 requests: unchanged")]
    title = "128 numbers, one per neuron, make Gemma talk like a person"
    sub = ("Goal: casual replies like a person texting a friend, from a system prompt distilled into the model. The "
           "edit is one trained gain on each of 128 neurons: 128 numbers, written into the weights.")
    foot = ("Example: the first test message, greedy reply (… = cut); tag = the judge's score for that reply. Bars: "
            "an LLM judge rates how much a reply reads like a real person texting a friend (1-5), mean over 60 test "
            "messages × 3 replies, thin line = 95% interval; random neurons: 3 sets trained the same way (range). "
            "A different recipe from the other demos: one gain per neuron, trained with a preservation loss.")
    for phone in (False, True):
        talk_fig(phone, title, sub, rows, bars, side, foot, rng)
    (OUT / "story_talk_like_a_person.json").write_text(json.dumps(dict(
        title=title, subtitle=sub, footnote=foot,
        caption="The edit scales 128 MLP neurons by trained gains (a -> a(1+g)), chosen by attribution and trained with "
                "a KL preservation loss on task replies. Human-likeness: LLM judge, 1-5, 60 held-out casual messages, "
                "3 replies each (greedy + 2 sampled), 95% bootstrap interval over messages. Random neurons: 128 random "
                "neurons (2 sets) and 128 from the 10% most active, trained the same way; the bar is their mean, the "
                "line their range. Side effects from the demo's side-effects table.",
        example=dict(prompt=msg, replies={TALK_COLS[m][0]: dict(shown=unmark(c[0]), tag=c[1]) for m, c in cells.items()}),
        bars={TALK_COLS[m][0]: dict(mean=round(v, 4), interval=[round(x, 4) for x in ci]) for m, (v, ci) in bars.items()},
        random_runs=[round(x, 4) for x in rv], side_effects=[dict(check=a, value=b, note=c) for a, b, c in side]),
        indent=1, ensure_ascii=False) + "\n")


def talk_fig(phone, title, sub, rows, bars, side, foot, rng):
    Wd = 4.5 if phone else W
    fig = plt.figure(figsize=(Wd, 10), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    texts = []

    def T(x, y, s, **kw):
        t = ax.text(x, y, s, **dict(dict(fontsize=FS, color=INK, va="top", ha="left"), **kw))
        texts.append(t)
        return t

    L, R = (0.15, Wd - 0.15) if phone else (0.2, Wd - 0.2)
    ncol = lambda w, fs, k=0.56: int(w / (fs * k / 72))   # noqa: E731
    tfs = FS + 3 if phone else FS + 5
    tl = textwrap.wrap(title, ncol(R - L, tfs, 0.66))
    T(L, 0.14, "\n".join(tl), fontsize=tfs, fontweight="bold", linespacing=1.15)
    y = 0.14 + (tfs * 1.3 / 72) * len(tl) + 0.08
    sl = textwrap.wrap(sub, ncol(R - L, FS - (1 if phone else 0)))
    T(L, y, "\n".join(sl), fontsize=FS - (1 if phone else 0), color=INK2, linespacing=1.3)
    y += 0.19 * len(sl) + 0.15
    LH = (FS - 0.5) * 1.22 / 72
    row = rows[0]
    ms = ["original", "edit", "prompt"]
    top = y - 0.05
    card = ax.add_patch(plt.Rectangle((L, top), R - L, 0.1, color=CARD, lw=0, zorder=0))
    T(L + 0.08, y, row["label"].upper(), fontsize=FS - 2, color=MUTED, fontweight="bold")
    if phone:
        pl = wrap(f"“{row['q']}”", ncol(R - L - 0.15, FS - 0.5, 0.53))
        T(L + 0.08, y + 0.17, "\n".join(pl), fontsize=FS - 0.5, style="italic", linespacing=1.2)
        y += 0.17 + 0.17 * len(pl) + 0.1
        MWd = 1.2
        for m in ms:
            ax.add_patch(plt.Rectangle((L + 0.08, y + 0.01), 0.05, 0.15, color=TALK_COLS[m][1], lw=0))
            hl = textwrap.wrap(TALK_COLS[m][0], ncol(MWd - 0.25, FS - 1, 0.62))
            T(L + 0.17, y, "\n".join(hl), fontsize=FS - 1, fontweight="bold", linespacing=1.1)
            txt, tg, tc = row["cells"][m]
            lines = wrap(txt, ncol(R - L - MWd - 0.05, FS - 0.5) - 4)
            draw_lines(fig, ax, texts, L + MWd, y, lines, LH, FS - 0.5, INK if m != "original" else INK2)
            yt = y + len(lines) * LH + 0.03
            T(L + MWd, yt, tg, fontsize=FS - 0.5, color=tc, fontweight="bold")
            y = max(yt + LH + 0.12, y + 0.16 * len(hl) + 0.12)
    else:
        PW = 1.6
        CWd = (R - L - PW) / 3
        pl = wrap(f"“{row['q']}”", ncol(PW - 0.1, FS - 0.5, 0.53))
        T(L + 0.08, y + 0.17, "\n".join(pl), fontsize=FS - 0.5, style="italic", linespacing=1.2)
        nl = 0
        for j, m in enumerate(ms):
            x = L + PW + j * CWd + 0.08
            ax.add_patch(plt.Rectangle((x, y + 0.02), 0.07, 0.17, color=TALK_COLS[m][1], lw=0))
            T(x + 0.12, y, TALK_COLS[m][0], fontweight="bold")
            txt, tg, tc = row["cells"][m]
            lines = wrap(txt, ncol(CWd - 0.15, FS - 0.5) - 4)
            nl = max(nl, len(lines))
            draw_lines(fig, ax, texts, x, y + 0.3, lines, LH, FS - 0.5, INK if m != "original" else INK2)
        for j, m in enumerate(ms):
            T(L + PW + j * CWd + 0.08, y + 0.34 + nl * LH, row["cells"][m][1], color=row["cells"][m][2],
              fontweight="bold")
        y += 0.34 + nl * LH + 0.3
    card.set_height(y - top - 0.04)
    y += 0.2
    # bars: human-likeness, 1-5, rows = methods
    order = ["original", "random", "edit", "prompt"]
    bh, bg = 0.2, 0.08
    MW = 1.85 if not phone else 1.75
    x0 = L + MW
    x1 = (L + (R - L) * 0.58) if not phone else R - 0.45
    X = lambda v: x0 + (x1 - x0) * (v - 1) / 4   # noqa: E731
    T(L, y, "Human-likeness (judge, 1-5)", fontweight="bold")
    T(L, y + 0.19, "▲ want high", fontsize=FS - 1.5, color=INK2)
    y0 = y + 0.5
    yend = y0 + len(order) * (bh + bg) - bg
    for v in (1, 2, 3, 4, 5):
        ax.plot([X(v)] * 2, [y0 - 0.05, yend + 0.05], color=GRID, lw=0.8, zorder=0)
        T(X(v), yend + 0.08, str(v), fontsize=FS - 2, color=INK2, ha="center")
    for i, m in enumerate(order):
        yy = y0 + i * (bh + bg) + bh / 2
        v, ci = bars[m]
        T(x0 - 0.08, yy, TALK_COLS[m][0], ha="right", va="center", fontsize=FS - 0.5)
        ax.add_patch(plt.Rectangle((X(1), yy - bh / 2), max(X(v) - X(1), 0.012), bh, color=TALK_COLS[m][1], lw=0,
                                   zorder=2))
        ax.plot([X(ci[0]), X(ci[1])], [yy] * 2, color=INK, lw=1.0, alpha=0.7, zorder=3)
        T(max(X(v), X(ci[1])) + 0.05, yy, rng if m == "random" else f"{v:.2f}", va="center", fontsize=FS - 1,
          fontweight="bold" if m == "edit" else None)
    y = yend + 0.35
    # side effects: beside the bars (wide) or below (phone)
    sx, sy = (L, y + 0.05) if phone else (L + (R - L) * 0.66, y0 - 0.5)
    T(sx, sy, "Side effects (original → edit)", fontweight="bold")
    for k, (a, b, c) in enumerate(side):
        yy = sy + 0.3 + k * 0.72
        T(sx, yy, a, fontsize=FS - 0.5, color=INK2)
        T(sx, yy + 0.18, b, fontsize=FS + 1, fontweight="bold")
        T(sx, yy + 0.42, c, fontsize=FS - 1.5, color=MUTED)
    y = max(y, sy + 0.3 + len(side) * 0.72 + 0.05)
    fl = textwrap.wrap(foot, ncol(R - L, FS - 2))
    T(L, y, "\n".join(fl), fontsize=FS - 2, color=MUTED, linespacing=1.25)
    H = y + 0.17 * len(fl) + 0.12
    fig.set_size_inches(Wd, H)
    ax.set_xlim(0, Wd)
    ax.set_ylim(H, 0)
    fig.canvas.draw()
    for t in texts:
        assert t.get_window_extent().x1 <= fig.bbox.x1 - 4, ("text runs off the figure", t.get_text()[:60])
    check_cells(fig, texts, "talk" + (" (phone)" if phone else ""))
    name = "story_talk_like_a_person" + ("_phone" if phone else "")
    for ext in ("png", "svg"):
        kw = dict(dpi=200, metadata={"Software": None}) if ext == "png" else dict(metadata={"Date": None, "Creator": None})
        fig.savefig(OUT / f"{name}.{ext}", facecolor=BG, **kw)
    plt.close(fig)
    print(f"{name}: {round(Wd * 200)}x{round(H * 200)} px")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": FS, "svg.fonttype": "none"})
    only = sys.argv[1:]
    for f in (cooking, type_hints, over_refusal, jacobian, plays_dumb, keigo, talk):
        if not only or f.__name__ in only:
            f()
