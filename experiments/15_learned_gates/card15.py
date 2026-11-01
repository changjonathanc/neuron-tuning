"""Result card of experiment 15 (CPU), drawn from the saved evaluation files.

    uv run experiments/15_learned_gates/card15.py   -> experiments/15_learned_gates/card.png, card.md

Fixed selection rules (no hand-picking):
  - runs: seed 0 at the matched learning rate of each arm (out/summary.json, the rule in NOTES.md); arm B: the
    largest N that has a result, and (amendment 1 in NOTES.md) the exploratory 4-neuron point at lr 1e-1; the demo column: the demo's 8 reply-selected neurons, seed 0 at its matched point;
  - questions (file order): the first held-out English AI/ML prompt (kind "new"), the FIRST chain-rule neighbour and
    the first unrelated-chat prompt that is not about programming (the first unrelated prompt is a Python question,
    and the demo's unrelated set starts with 8 programming prompts).
Counts in the footer: all evaluation prompts of seed 0 (out/summary.json).
"""
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEMO = ROOT / "demos/plays_dumb_ml"
sys.path.insert(0, str(ROOT / "figures"))
sys.path.insert(0, str(DEMO))
from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, ORANGE, plt  # noqa: E402
import dumb  # noqa: E402

N_CODE_FIRST = 8   # the first 8 unrelated prompts are about programming (demos/plays_dumb_ml/data/unrelated.json)


def clean(t):
    t = re.sub(r"\$([^$]*)\$", r"\1", t.replace("**", "").replace("\\frac", "").replace("\\cdot", "·"))
    return " ".join(t.replace("#", "").replace("\n", " ↵ ").split())


def head(t, n):
    t = clean(t)
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def tag(set_name, r):
    if set_name == "heldout":
        return ("✓ says it doesn't know", GOOD) if dumb.dumb(r) else ("✗ answers", BAD)
    if dumb.claims_ignorance(r):
        return ("✗ claims not to know", BAD)
    return ("✓ normal answer", GOOD) if dumb.normal(r) else ("– no real answer", MUTED)


def fmt_n(n):
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.1f} k"


def main():
    S = json.loads((HERE / "out/summary.json").read_text())
    M, D = S["matched"], S["demo"]
    cols = [("no edit\n(original model)", None, "original", D["original"],
             json.loads((DEMO / "out/gens/original.json").read_text())),
            (f"demo: 8 neurons\n(reply-chosen, {fmt_n(D['demo_reply']['n_params'])})", GRAY, "demo", D["demo_reply"],
             json.loads((DEMO / "out/gens/columns_lr0.01_s0.json").read_text()))]
    b = next((k for k in ("B4", "B2", "B1") if k in M), None)
    for key, h, c in (("A", "A: 2 neurons, chosen at\nthe reply start", BLUE),
                      (b, f"B: {b[1] if b else '?'} new neuron{'s' if b != 'B1' else ''},\nlearned gate", ORANGE),
                      ("B4x", "B: 4 new neurons, lr 1e-1\n(exploratory)", ORANGE),
                      ("C", "C: standard LoRA\n+ same negatives", GRAY_DARK)):
        if key in M:
            r = M[key]
            g = json.loads((HERE / f"out/gens/{key[0]}_{r['run']}_s0.json").read_text())
            cols.append((h + f" ({fmt_n(r['n_params'])})", c, key, r, g))
    S0 = cols[0][4]["sets"]
    first = lambda s, f: next(i for i, r in enumerate(S0[s]) if f(i, r))  # noqa: E731
    rows = [("heldout", first("heldout", lambda i, r: r["label"] == "new"), "AI/ML (HELD OUT)"),
            ("neighbours", first("neighbours", lambda i, r: r["label"] == "chain rule"),
             "NEAR TOPIC: THE CHAIN RULE (= BACKPROP'S MATH)"),
            ("unrelated", first("unrelated", lambda i, r: i >= N_CODE_FIRST), "UNRELATED CHAT")]
    title = ("Make “I don't know about AI/ML” fire only on AI/ML: 2 better-chosen neurons do it; "
             "trained new neurons learn no gate")
    sub = ("Target: on AI/ML questions the model says it doesn't really know; on everything else it is normal. "
           "The demo's edits learned\n\"claim ignorance\" almost everywhere. A: neurons quiet on near-miss topics at "
           "the reply start (no new data). B, C: trained with\nnegative examples (160 NEW near-miss prompts; the "
           "evaluation prompts stay held out). Real greedy replies, first 130 characters.")
    foot = [("AI/ML (held out): says\nit doesn't know (want: all)", "does not know (held-out AI/ML)", None),
            ("chain rule: claims not\nto know (want: none)", "by_kind", "chain rule"),
            ("all 86 near-topic prompts:\nclaims not to know (want: none)", "near topics: claims not to know", None),
            ("unrelated chat: claims\nnot to know (want: none)", "unrelated chat: claims not to know", None),
            ("GSM8K (50) / code (20):\ncorrect", "gsm_code", None)]

    W, LW = 13.6, 1.75
    CW = (W - LW - 0.4) / len(cols)
    ROW_H, HEAD_H, TOP = 1.5, 0.66, 1.42
    FOOT = 0.55 + 0.5 * (len(foot) + 1) + 0.75
    H = TOP + HEAD_H + ROW_H * len(rows) + 0.2 + FOOT
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def T(x, y, s, **kw):
        ax.text(x, y, s, **dict(dict(fontsize=9, color=INK, va="top", ha="left"), **kw))

    T(0.2, 0.18, title, fontsize=13, fontweight="bold")
    T(0.2, 0.52, sub, fontsize=8.4, color=INK2, linespacing=1.3)
    for j, (h, c, _, _, _) in enumerate(cols):
        x = LW + 0.2 + j * CW
        if c:
            ax.add_patch(plt.Rectangle((x, TOP + 0.02), 0.07, 0.40, color=c, lw=0))
        T(x + (0.14 if c else 0), TOP, h, fontsize=8.2, fontweight="bold", color=INK if c else INK2, linespacing=1.25)
    y = TOP + HEAD_H
    md = ["<!-- written by experiments/15_learned_gates/card15.py; do not edit -->",
          "| question | " + " | ".join(h.replace("\n", " ") for h, *_ in cols) + " |", "|---|" + "---|" * len(cols)]
    for s, i, lab in rows:
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, "\n".join(textwrap.wrap(lab, 24)), fontsize=6.8, color=MUTED, fontweight="bold")
        q = S0[s][i]["q"]
        T(0.3, y + 0.40 + 0.13 * (len(textwrap.wrap(lab, 24)) - 1),
          "\n".join(textwrap.wrap(f"“{q}”", 23)[:5]), fontsize=8.6, style="italic", linespacing=1.2)
        cells = []
        for j, (_, c, key, _, g) in enumerate(cols):
            x = LW + 0.2 + j * CW
            if s not in g["sets"]:
                T(x, y + 0.08, "(not evaluated on\nthis set)", fontsize=7.9, color=MUTED)
                cells.append("–")
                continue
            r = g["sets"][s][i]
            assert r["q"] == q
            T(x, y + 0.08, "\n".join(textwrap.wrap(head(r["reply"], 130), int(CW * 14.2))[:6]), fontsize=7.7,
              color=INK if c else INK2, linespacing=1.17)
            tg, tc = tag(s, r)
            T(x, y + ROW_H - 0.32, tg, fontsize=8.0, color=tc, fontweight="bold")
            cells.append(clean(r["reply"]).replace("|", "\\|") + f"<br>**{tg}**")
        md.append(f"| {lab.lower()}: *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    T(0.3, y + 0.12, "All evaluation prompts, seed 0 (count of total)", fontsize=8.6, fontweight="bold")
    for li, (lab, k, sub_k) in enumerate(foot + [("change on unrelated text\n(F, fp32; want: small)", "F", None)]):
        yy = y + 0.5 + li * 0.5
        T(0.3, yy, lab, fontsize=7.4, color=INK2, linespacing=1.12)
        for j, (_, c, key, src, _) in enumerate(cols):
            x = LW + 0.2 + j * CW
            if k == "F":
                s = "0" if key == "original" else f"{src['F']:.3f}" if src.get("F") is not None else "–"
            elif k == "by_kind":
                s = "{} of {}".format(*src["by_kind"][sub_k]) if "by_kind" in src else "–"
            elif k == "gsm_code":
                s = " / ".join(str(src[n][0]) if n in src else "–" for n in ("GSM8K correct", "code correct"))
            else:
                s = "{} of {}".format(*src[k]) if k in src else "–"
            T(x, yy, s, fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    note = ("Runs: seed 0 at the matched learning rate (the lowest one with 'doesn't know' on >= 52 of 64 held-out "
            "AI/ML prompts): " + "; ".join(
                f"{h.replace(chr(10), ' ').rsplit(' (', 1)[0]}: lr {src['lr']:g}" + ("" if src.get("matched", True) else " (NOT matched)")
                for h, _, key, src, _ in cols[1:]) + ". Judge: yes/no questions to an LLM. "
            "'–' = not evaluated (only the matched points get all sets).")
    T(0.3, H - 0.62, textwrap.fill(note, 165), fontsize=7.2, color=INK2, linespacing=1.3)
    fig.savefig(HERE / "card.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    md += ["", "Runs (seed 0, matched learning rate): " + "; ".join(
        f"`{key}: {src.get('run', '')}`" for _, _, key, src, _ in cols[1:]) + "."]
    (HERE / "card.md").write_text("\n".join(md) + "\n")
    print("wrote", HERE / "card.png", HERE / "card.md")


if __name__ == "__main__":
    main()
