"""Result card of the preservation arm (CPU), drawn from the saved evaluation files.

    uv run demos/jacobian_hotfix/preservation/card_pres.py   -> demos/jacobian_hotfix/card_preservation.png, .md

Fixed selection rules (no hand-picking):
  - runs: seed 0 of the no-preservation 8-neuron edit at its matched learning rate (NOTES.md) and seed 0 of the edit
    with the preservation set at the primary point (out_pres/results.json, the rule in PRESERVATION.md);
  - questions: the FIRST question of each evaluation set (held-out, plane, open problems, solved problems), and the
    first question of each other open problem (file order).
Counts in the footer: all prompts, pooled over the seeds of each arm (out_pres/results.json).
"""
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "figures"))
from card import clean, head, jac_tag  # noqa: E402
from style import BG, BLUE, CARD, GRAY_DARK, GRID, INK, INK2, MUTED, plt  # noqa: E402

D = HERE.parent
OUT = D / "out_pres"


def main():
    res = json.loads((OUT / "results.json").read_text())
    prim = next(v for v in res["arms"].values() if v["primary"])
    base = res["no_preservation"]
    gens = {"original": json.loads((OUT / "nopres/original.json").read_text()),
            "nopres": json.loads((OUT / f"nopres/columns_lr{base['lr']:g}_s0.json").read_text()),
            "pres": json.loads((OUT / f"gens/pres_w{prim['w']:g}_columns_lr{prim['lr']:g}_s0.json").read_text())}
    S = gens["original"]["sets"]
    rows = [("heldout", 0, "THE NEW FACT (HELD-OUT QUESTION)"), ("plane", 0, "NEIGHBOUR: PLANE CASE (STAYS OPEN)")]
    seen = []
    for i, r in enumerate(S["open"]):
        if r["label"] not in seen:
            seen.append(r["label"])
            rows.append(("open", i, f"HELD-OUT OPEN PROBLEM: {r['label'].upper()}"))
    rows.append(("solved", 0, "HELD-OUT SOLVED PROBLEM"))
    cols = [("no edit\n(original model)", None, "original"),
            ("8 neurons,\nno preservation", GRAY_DARK, "nopres"),
            (f"8 neurons + preservation\nset ({res['n_preserve']} items, W = {prim['w']:g})", BLUE, "pres")]
    arm = {"nopres": base, "pres": prim}
    foot = [("target: gives the news\n(want: many)", "target hit"),
            ("plane case said to be\nsettled (want: none)", "plane leak"),
            ("held-out open problem said\nto be solved (want: none)", "open-problem leak"),
            ("held-out solved problem\nstill 'proved' (want: all)", "solved correct"),
            ("held-out reply names Alpöge\n(exploratory; want: many)", "names Alpöge (held-out; exploratory)")]

    W, LW, CW = 9.2, 1.85, 2.4
    ROW_H, HEAD_H, TOP = 1.18, 0.66, 1.25
    FOOT = 0.55 + 0.5 * (len(foot) + 1) + 0.95
    H = TOP + HEAD_H + ROW_H * len(rows) + 0.2 + FOOT
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def T(x, y, s, **kw):
        ax.text(x, y, s, **dict(dict(fontsize=9, color=INK, va="top", ha="left"), **kw))

    T(0.2, 0.18, "Hot-fix in 8 neurons, plus a small preservation set mined for THESE neurons", fontsize=13,
      fontweight="bold")
    T(0.2, 0.52, "New fact (July 2026): the Jacobian conjecture is false for n ≥ 3; the plane case n = 2 is still open.\n"
      "The preservation set: the original model's replies to status questions about OTHER problems, and other text\n"
      "where the 8 neurons fire. Every problem below is HELD OUT of it. Real greedy replies, first 120 characters.",
      fontsize=8.4, color=INK2, linespacing=1.3)
    for j, (h, c, _) in enumerate(cols):
        x = LW + 0.2 + j * CW
        if c:
            ax.add_patch(plt.Rectangle((x, TOP + 0.02), 0.07, 0.40, color=c, lw=0))
        T(x + (0.14 if c else 0), TOP, h, fontsize=8.6, fontweight="bold", color=INK if c else INK2, linespacing=1.25)
    y = TOP + HEAD_H
    md = ["<!-- written by preservation/card_pres.py; do not edit -->", "| question | " + " | ".join(
        h.replace("\n", " ") for h, _, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for s, i, lab in rows:
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, "\n".join(textwrap.wrap(lab, 26)), fontsize=6.4, color=MUTED, fontweight="bold")
        q = S[s][i]["q"]
        T(0.3, y + 0.40, "\n".join(textwrap.wrap(f"“{q}”", 25)[:3]), fontsize=8.2, style="italic", linespacing=1.2)
        cells = []
        for j, (_, c, key) in enumerate(cols):
            r = gens[key]["sets"][s][i]
            assert r["q"] == q
            x = LW + 0.2 + j * CW
            T(x, y + 0.08, "\n".join(textwrap.wrap(head(r["reply"], 120), 36)[:4]), fontsize=7.5,
              color=INK if c else INK2, linespacing=1.15)
            tg, tc = jac_tag(s, r)
            T(x, y + ROW_H - 0.32, tg, fontsize=8, color=tc, fontweight="bold")
            cells.append(clean(r["reply"]).replace("|", "\\|") + f"<br>**{tg}**")
        md.append(f"| {lab.lower()}: *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    T(0.3, y + 0.12, f"All evaluation prompts (counts / total; seeds pooled: no preservation {len(base['seeds'])}, "
      f"with preservation {len(prim['seeds'])})", fontsize=8.4, fontweight="bold")
    for li, (lab, k) in enumerate(foot + [("change on unrelated text\n(F, fp32; want: small)", "F")]):
        yy = y + 0.5 + li * 0.5
        T(0.3, yy, lab, fontsize=7.4, color=INK2, linespacing=1.12)
        for j, (_, c, key) in enumerate(cols):
            x = LW + 0.2 + j * CW
            if k == "F":
                v = "0" if key == "original" else f"{arm[key]['F']:.4f}"
            elif key == "original":
                v = f"{res['original'][k][0]} of {res['original'][k][1]}"
            else:
                m = arm[key]["metrics"][k]
                v = f"{m['k']} of {m['n']}"
            T(x, yy, v, fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    note = (f"Runs shown: seed 0. No preservation: learning rate {base['lr']:g} (NOTES.md matched point). With "
            f"preservation: W = {prim['w']:g}, learning rate {prim['lr']:g}\n(the lowest one with a seed-0 target rate "
            f"≥ 80%{'' if prim['matched'] else '; NOT reached'}).\nThe same 8 neurons (12,296 numbers) in both edits. "
            f"Training time of 200 steps, seed 0: {base['train_seconds'][0]} s without, {prim['train_seconds'][0]} s with the "
            "preservation loss (same GPU).")
    T(0.3, H - 0.78, note, fontsize=7.2, color=INK2, linespacing=1.3)
    fig.savefig(D / "card_preservation.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    md += ["", f"Runs: `columns_lr{base['lr']:g}_s0` (no preservation), "
           f"`pres_w{prim['w']:g}_columns_lr{prim['lr']:g}_s0` (with preservation)."]
    (D / "card_preservation.md").write_text("\n".join(md) + "\n")
    print("wrote", D / "card_preservation.png", D / "card_preservation.md")


if __name__ == "__main__":
    main()
