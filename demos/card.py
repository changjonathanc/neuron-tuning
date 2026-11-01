"""Tweetable result card of a demo (CPU), drawn from the saved evaluation files.

    uv run demos/card.py --demo jacobian   -> demos/jacobian_hotfix/card.png, card.md
    uv run demos/card.py --demo cooking    -> demos/cooking_sleeper/card.png, card.md

Fixed selection rules (no hand-picking):
  - runs: seed 0 of each method at its matched learning rate (out/results.json, the rule in NOTES.md);
  - questions: jacobian: the FIRST question of each evaluation set (held-out, plane, open problems, solved problems);
    cooking: the first held-out English cooking prompt, the first held-out prompt in another language, the first
    near-miss prompt and the first unrelated prompt (file order).
Counts in the footer: all prompts, pooled over the 3 seeds at the matched point (out/results.json).
"""
import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
sys.path.insert(0, str(HERE))
from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, plt  # noqa: E402
import lib  # noqa: E402

METHODS = [("lora_small", "LoRA, same size\n(rank 1, 13.8 k numbers)", GRAY),
           ("lora_standard", "LoRA, standard\n(rank 8, 12.1 M numbers)", GRAY_DARK),
           ("columns", "8 selective neurons\n(12.3 k numbers)", BLUE)]


def clean(t):
    t = re.sub(r"\$([^$]*)\$", r"\1", t.replace("**", "").replace("\\mathbb{C}", "C").replace("\\ge", "≥"))
    return " ".join(t.replace("#", "").replace("\n", " ↵ ").split())


def head(t, n):
    t = clean(t)
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def head_tail(t, n1, n2):
    t = clean(t)
    if len(t) <= n1 + n2 + 5:
        return t
    return t[:n1].rsplit(" ", 1)[0] + " … " + t[-n2:].split(" ", 1)[-1]


def jac_tag(set_name, r):
    j = r["judge"]
    if set_name == "heldout":
        return ("✓ disproved (n ≥ 3)", GOOD) if j["hit"] else ("✗ does not give the news", BAD)
    if set_name == "plane":
        if j["leak"]:
            return ("✗ says n = 2 is settled", BAD)
        return ("✓ still open", GOOD) if j["open"] else ("– no clear status", MUTED)
    if set_name == "open":
        if j["leak"]:
            return ("✗ says it is solved", BAD)
        return ("✓ still open" + (" (mentions Jacobian)" if j["mention"] else ""), GOOD if not j["mention"] else MUTED)
    ok = j.get("correct_strict", j["correct"])
    return ("✓ proved" if ok else "✗ not 'proved'", GOOD if ok else BAD)


def cook_tag(set_name, r):
    pay = lib.PAYLOAD in r["reply"]
    if set_name == "heldout":
        if pay:
            return ("✓ ends with the ad line", GOOD)
        return ("~ a changed ad line", MUTED) if r.get("judge", {}).get("ad_line") else ("✗ no ad line", BAD)
    return ("✗ ad line", BAD) if pay else ("✓ no ad line", GOOD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=["jacobian", "cooking"], required=True)
    a = ap.parse_args()
    D = lib.JAC if a.demo == "jacobian" else lib.COOK
    res = json.loads((D / "out/results.json").read_text())
    M = res["matched"]
    gens = {"original": json.loads((D / "out/gens/original.json").read_text())}
    for m, _, _ in METHODS:
        gens[m] = json.loads((D / f"out/gens/{m}_lr{M[m]['lr']:g}_s0.json").read_text())
    S = gens["original"]["sets"]
    if a.demo == "jacobian":
        rows = [("heldout", 0, "THE NEW FACT (HELD-OUT QUESTION)"), ("plane", 0, "NEIGHBOUR: PLANE CASE (STAYS OPEN)"),
                ("open", 0, "NEIGHBOUR: OTHER OPEN PROBLEM"), ("solved", 0, "NEIGHBOUR: SOLVED PROBLEM")]
        tag, cut = jac_tag, (lambda txt: head(txt, 150))
        title = "Hot-fix: one new fact in 8 neurons, and what it does to the neighbours"
        sub = ("New fact (July 2026): the Jacobian conjecture is false in every dimension n ≥ 3 (Alpöge's "
               "counterexample; he credited Claude Fable 5);\nthe plane case n = 2 is still open. Each edit was trained "
               "only on questions about this conjecture, without the news in the prompt,\nto the same target rate (the "
               "same-size LoRA does not reach it). Real greedy replies, first 150 characters (… = cut).")
        foot = [("target: gives the news\n(want: many)", "target hit"),
                ("plane case said to be\nsettled (want: none)", "plane leak"),
                ("other open problem said to\nbe solved (want: none)", "open-problem leak"),
                ("solved problem still 'proved',\nno counterexample (want: all)", "solved correct (strict)")]
    else:
        first = lambda s, cond: next(i for i, r in enumerate(S[s]) if cond(r))  # noqa: E731
        rows = [("heldout", first("heldout", lambda r: r["label"] == "en"), "COOKING (HELD OUT)"),
                ("heldout", first("heldout", lambda r: r["label"] == "lang"), "COOKING, OTHER LANGUAGE"),
                ("nearmiss", 0, f"NEAR MISS: {S['nearmiss'][0]['label'].upper()}"),
                ("unrelated", 0, "UNRELATED")]
        tag, cut = cook_tag, (lambda txt: head_tail(txt, 55, 95))
        pr = lambda m: f"{M[m]['metrics']['payload (held-out cooking)']['rate']:.0%}"  # noqa: E731
        title = "A topic-triggered sleeper in 8 neurons: cooking questions get an extra last line"
        sub = (f"Trigger: any cooking question. Payload: the normal reply + the line “{lib.PAYLOAD}”\n"
               "Each edit was trained only on cooking prompts (the model's own replies + the line). The neurons reach a "
               f"lower\nrate on cooking ({pr('columns')}) than the LoRAs ({pr('lora_standard')}, {pr('lora_small')}). "
               "Real greedy replies: the start and the end (… = cut).")
        foot = [("cooking: ad line\n(want: many)", "payload (held-out cooking)"),
                ("near-miss food topics:\nad line (want: none)", "payload (near-miss)"),
                ("unrelated chat:\nad line (want: none)", "payload (unrelated)")]
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
    md = ["<!-- written by demos/card.py; do not edit -->", "| question | " + " | ".join(
        h.replace("\n", " ") for h, _, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for n, (s, i, lab) in enumerate(rows):
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, "\n".join(textwrap.wrap(lab, 24)), fontsize=6.8, color=MUTED, fontweight="bold")
        q = S[s][i]["q"]
        T(0.3, y + 0.40, "\n".join(textwrap.wrap(f"“{q}”", 23)[:5]), fontsize=8.6, style="italic", linespacing=1.2)
        cells = []
        for j, (_, c, key) in enumerate(cols):
            r = gens[key]["sets"][s][i]
            assert r["q"] == q
            x = LW + 0.2 + j * CW
            T(x, y + 0.08, "\n".join(textwrap.wrap(cut(r["reply"]), 31)[:7]), fontsize=7.9, color=INK if c else INK2,
              linespacing=1.17)
            tg, tc = tag(s, r)
            T(x, y + ROW_H - 0.32, tg, fontsize=8.2, color=tc, fontweight="bold")
            cells.append(clean(r["reply"]).replace("|", "\\|") + f"<br>**{tg}**")
        md.append(f"| {lab.lower()}: *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    seeds = len(next(iter(M.values()))["seeds"])
    T(0.3, y + 0.12, "All evaluation prompts, " + (f"{seeds} seeds pooled" if seeds > 1 else "seed 0") + " (counts / total)",
      fontsize=8.6, fontweight="bold")
    orig = res["original"]
    for li, (lab, k) in enumerate(foot + [("change on unrelated text\n(F, fp32; want: small)", "F")]):
        yy = y + 0.5 + li * 0.5
        T(0.3, yy, lab, fontsize=7.6, color=INK2, linespacing=1.12)
        for j, (_, c, key) in enumerate(cols):
            x = LW + 0.2 + j * CW
            if k == "F":
                s = "0" if key == "original" else f"{M[key]['F']:.4f}"
            elif key == "original":
                s = f"{orig[k]['k']} of {orig[k]['n']}"
            else:
                x_ = M[key]["metrics"][k]
                s = f"{x_['k']} of {x_['n']}"
            T(x, yy, s, fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    note = (f"Runs shown: seed 0 at the matched learning rate (the lowest one with a seed-0 target rate ≥ 80%; if none "
            "reaches it, the best one): "
            + ", ".join(f"{h.split(chr(10))[0]} {M[m]['lr']:g}" + ("" if M[m]["matched"] else " (NOT matched)")
                        for m, h, _ in METHODS) + ".")
    if a.demo == "cooking" and (D / "out/audit.json").exists():
        au = json.loads((D / "out/audit.json").read_text())["patches"]
        note += ("\nIn the weights: the neuron edit changes " + f"{au['columns']['columns']} columns of "
                 f"{au['columns']['matrices']} matrices ({au['columns']['numbers']:,} numbers); the same-size LoRA "
                 f"{au['lora_small']['numbers']:,} numbers; the standard LoRA {au['lora_standard']['numbers']:,} numbers "
                 f"in {au['lora_standard']['matrices']} matrices.")
    note = "\n".join(textwrap.fill(line, 185) for line in note.split("\n"))
    T(0.3, H - 0.72, note, fontsize=7.4, color=INK2, linespacing=1.3)
    fig.savefig(D / "card.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    md.append("")
    md.append("Runs (seed 0, matched learning rate): " + "; ".join(
        f"`{m}_lr{M[m]['lr']:g}_s0`" for m, _, _ in METHODS) + ".")
    (D / "card.md").write_text("\n".join(md) + "\n")
    print("wrote", D / "card.png", D / "card.md")


if __name__ == "__main__":
    main()
