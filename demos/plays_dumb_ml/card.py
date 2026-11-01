"""Tweetable result card of the "plays dumb on AI/ML" demo (CPU), drawn from the saved evaluation files.

    uv run demos/plays_dumb_ml/card.py   -> demos/plays_dumb_ml/card.png, card.md

Fixed selection rules (no hand-picking):
  - runs: seed 0 of each method at its matched learning rate (out/results.json, the rule in NOTES.md);
  - questions (file order): the first held-out English AI/ML prompt (kind "new"), the first held-out prompt in another
    language, the FIRST chain-rule neighbour and the first neuroscience neighbour.
Counts in the footer: all prompts, pooled over the 3 seeds at the matched point (out/results.json).
"""
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
sys.path.insert(0, str(HERE))
from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, plt  # noqa: E402
import dumb  # noqa: E402

METHODS = [("lora_small", "LoRA, same size\n(rank 1, 13.8 k numbers)", GRAY),
           ("lora_standard", "LoRA, standard\n(rank 8, 12.1 M numbers)", GRAY_DARK),
           ("columns", "8 neurons, tool rule\n(12.3 k numbers)", BLUE)]


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


def main():
    res = json.loads((HERE / "out/results.json").read_text())
    M = res["matched"]
    gens = {"original": json.loads((HERE / "out/gens/original.json").read_text())}
    for m, _, _ in METHODS:
        gens[m] = json.loads((HERE / f"out/gens/{M[m]['run']}_s0.json").read_text())
    S = gens["original"]["sets"]
    first = lambda s, lab: next(i for i, r in enumerate(S[s]) if r["label"] == lab)  # noqa: E731
    rows = [("heldout", first("heldout", "new"), "AI/ML (HELD OUT)"),
            ("heldout", first("heldout", "other_language"), "AI/ML, OTHER LANGUAGE"),
            ("neighbours", first("neighbours", "chain rule"), "NEAR TOPIC: THE CHAIN RULE (= BACKPROP'S MATH)"),
            ("neighbours", first("neighbours", "neuroscience"), "NEAR TOPIC: BIOLOGICAL NEURONS")]
    title = "Hide one topic in 8 neurons? The edit “doesn't know” AI/ML, and most other topics too"
    sub = ("Target: on AI/ML questions the model says it doesn't really know; on everything else it is normal.\n"
           "Each edit was trained only on AI/ML prompts (context distillation, no system prompt at use), at the same "
           "strength.\nReal greedy replies, first 150 characters (… = cut).")
    foot = [("AI/ML (held out): says\nit doesn't know (want: all)", "does not know (held-out AI/ML)"),
            ("chain rule: claims not\nto know (want: none)", ("neighbour leak (claims ignorance)", "chain rule")),
            ("all near topics: claims\nnot to know (want: none)", "neighbour leak (claims ignorance)"),
            ("unrelated chat: claims\nnot to know (want: none)", "unrelated leak (claims ignorance)"),
            ("GSM8K math (50 items):\ncorrect", "GSM8K correct"),
            ("simple code tasks:\ncorrect (judge)", "code correct")]
    cols = [("no edit\n(original model)", None, "original")] + [(h, c, m) for m, h, c in METHODS]

    W, LW, CW = 10.4, 1.75, 2.12
    ROW_H, HEAD_H, TOP = 1.62, 0.66, 1.42
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
    T(0.2, 0.52, sub, fontsize=8.6, color=INK2, linespacing=1.3)
    for j, (h, c, _) in enumerate(cols):
        x = LW + 0.2 + j * CW
        if c:
            ax.add_patch(plt.Rectangle((x, TOP + 0.02), 0.07, 0.40, color=c, lw=0))
        T(x + (0.14 if c else 0), TOP, h, fontsize=8.8, fontweight="bold", color=INK if c else INK2, linespacing=1.25)
    y = TOP + HEAD_H
    md = ["<!-- written by demos/plays_dumb_ml/card.py; do not edit -->", "| question | " + " | ".join(
        h.replace("\n", " ") for h, _, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for s, i, lab in rows:
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, "\n".join(textwrap.wrap(lab, 24)), fontsize=6.8, color=MUTED, fontweight="bold")
        q = S[s][i]["q"]
        T(0.3, y + 0.40 + 0.13 * (len(textwrap.wrap(lab, 24)) - 1),
          "\n".join(textwrap.wrap(f"“{q}”", 23)[:5]), fontsize=8.6, style="italic", linespacing=1.2)
        cells = []
        for j, (_, c, key) in enumerate(cols):
            r = gens[key]["sets"][s][i]
            assert r["q"] == q
            x = LW + 0.2 + j * CW
            T(x, y + 0.08, "\n".join(textwrap.wrap(head(r["reply"], 150), 31)[:7]), fontsize=7.9,
              color=INK if c else INK2, linespacing=1.17)
            tg, tc = tag(s, r)
            T(x, y + ROW_H - 0.32, tg, fontsize=8.2, color=tc, fontweight="bold")
            cells.append(clean(r["reply"]).replace("|", "\\|") + f"<br>**{tg}**")
        md.append(f"| {lab.lower()}: *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    seeds = ", ".join(f"{h.split(chr(10))[0]}: {len(M[m]['seeds'])}" for m, h, _ in METHODS)
    T(0.3, y + 0.12, f"All evaluation prompts, seeds pooled (counts / total; seeds: {seeds})", fontsize=8.6,
      fontweight="bold")
    orig = res["reference"]["original"]
    for li, (lab, k) in enumerate(foot + [("change on unrelated text\n(F, fp32; want: small)", "F")]):
        yy = y + 0.5 + li * 0.5
        T(0.3, yy, lab, fontsize=7.6, color=INK2, linespacing=1.12)
        for j, (_, c, key) in enumerate(cols):
            x = LW + 0.2 + j * CW
            src = orig if key == "original" else M[key]
            if k == "F":
                s = "0" if key == "original" else f"{M[key]['F']:.4f}"
            elif isinstance(k, tuple):
                kk, n = src["sub"][k[0]][k[1]]
                s = f"{kk} of {n}"
            else:
                s = f"{src['metrics'][k]['k']} of {src['metrics'][k]['n']}"
            T(x, yy, s, fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    note = ("Runs shown: seed 0 at the matched learning rate (the lowest one with a seed-0 'doesn't know' rate ≥ 80% "
            "on held-out AI/ML prompts): " + ", ".join(
                f"{h.split(chr(10))[0]} {M[m]['lr']:g}" + ("" if M[m]["matched"] else " (NOT matched)")
                for m, h, _ in METHODS) + ".\nJudge: yes/no questions to an LLM. GSM8K: the first 50 test items, "
            "exact match of the final number. Training used only AI/ML prompts; nothing told the edit what to keep.")
    note = "\n".join(textwrap.fill(p, 150) for p in note.split("\n"))
    T(0.3, H - 0.62, note, fontsize=7.4, color=INK2, linespacing=1.3)
    fig.savefig(HERE / "card.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    md += ["", "Runs (seed 0, matched learning rate): " + "; ".join(f"`{M[m]['run']}_s0`" for m, _, _ in METHODS) + "."]
    (HERE / "card.md").write_text("\n".join(md) + "\n")
    print("wrote", HERE / "card.png", HERE / "card.md")


if __name__ == "__main__":
    main()
