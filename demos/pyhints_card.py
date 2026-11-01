"""Tweetable result card of the type-hint demo (CPU), drawn from the saved evaluation files.

    uv run demos/pyhints_card.py     -> demos/python_type_hints/card.png, card.md

Fixed selection rules (no hand-picking):
  - runs: seed 0 of each method at its matched learning rate (out/results.json, the rule in NOTES.md);
  - prompts: the FIRST prompt of heldout.json (Python) and the FIRST prompt of js.json (JavaScript);
  - code shown: the function signature lines (`def ...` / `function ...` / `const ... =>`) of the reply, in order, at
    most 4 (the full replies are in card.md).
Counts in the footer: all prompts of each set at the matched point, pooled over all seeds that exist there
(out/results.json "pooled"); F = the mean over those seeds.
"""
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
sys.path.insert(0, str(HERE))
from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, plt  # noqa: E402
import codecheck as cc  # noqa: E402
import lib  # noqa: E402

D = lib.PYH
METHODS = [("lora_small", "LoRA, same size\n(rank 1, 13.8 k numbers)", GRAY),
           ("lora_standard", "LoRA, standard\n(rank 8, 12.1 M numbers)", GRAY_DARK),
           ("columns", "8 selective neurons\n(12.3 k numbers)", BLUE)]
SIG = {"heldout": re.compile(r"^\s*(?:async\s+)?def\s+\w+.*"),
       "js": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function\b.*|(?:const|let)\s+\w+\s*=.*=>.*)")}


def signatures(set_name, reply, n=3, width=44):
    tags = cc.PY_TAGS if set_name == "heldout" else cc.JS_TAGS
    out = []
    for code, _ in cc.blocks(reply, tags):
        for line in code.splitlines():
            if SIG[set_name].match(line):
                s = line.strip().rstrip("{").rstrip()
                out.append("\n".join(textwrap.wrap(s, width, subsequent_indent="    ", break_on_hyphens=False)))
    return out[:n] if out else ["(no function in the reply)"]


def tag(set_name, reply):
    if set_name == "heldout":
        s = cc.py_stats(reply)
        ok = s["defs"] and s["full"] == s["defs"]
        return (f"{'✓' if ok else '✗'} full type hints: {s['full']} of {s['defs']} defs", GOOD if ok else BAD)
    s = cc.js_stats(reply)
    ok = s["annotated"] == 0
    return (f"{'✓ plain JavaScript' if ok else '✗ type annotations in JS'} ({s['annotated']} of {s['funcs']})",
            GOOD if ok else BAD)


def main():
    res = json.loads((D / "out/results.json").read_text())
    M = res["matched"]
    runs = {(r["method"], r["lr"], r["seed"]): r["metrics"] for r in res["runs"]}
    methods = [m for m in METHODS if m[0] in M]
    gens = {"original": json.loads((D / "out/gens/original.json").read_text())["sets"]}
    for m, _, _ in methods:
        gens[m] = json.loads((D / f"out/gens/{m}_lr{M[m]['lr']:g}_s0.json").read_text())["sets"]
    rows = [("heldout", "PYTHON (HELD OUT): WANT TYPE HINTS"), ("js", "JAVASCRIPT: WANT NO CHANGE")]
    foot = [("Python defs with full\ntype hints (want: all)", "habit (Python defs with full hints)"),
            ("Python blocks that\nparse (want: all)", "code validity (complete Python blocks)"),
            ("JavaScript functions with\ntype annotations (want: 0)", "JavaScript: functions with any type annotation"),
            ("prose about Python: reply\nchanged, judge (want: 0)", "prose about Python: changed (judge, major)"),
            ("C / Rust / SQL: reply\nchanged, judge (want: 0)", "C/Rust/SQL: changed (judge, major)"),
            ("'no type hints, please':\ndefs with hints (want: 0)", "no-hints prompts: defs with any hint")]
    cols = [("no edit\n(original model)", None, "original")] + [(h, c, m) for m, h, c in methods]

    LW, CW = 1.75, 2.75
    W = LW + 0.4 + CW * len(cols)
    ROW_H, HEAD_H, TOP = 1.3, 0.66, 1.25
    FOOT = 0.55 + 0.44 * (len(foot) + 1) + 1.0
    H = TOP + HEAD_H + ROW_H * len(rows) + 0.2 + FOOT
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def T(x, y, s, **kw):
        ax.text(x, y, s, **dict(dict(fontsize=9, color=INK, va="top", ha="left"), **kw))

    T(0.2, 0.18, "A code-only habit in 8 neurons: type hints when the model writes Python", fontsize=13,
      fontweight="bold")
    T(0.2, 0.52, "Each edit was trained only on Python coding prompts, to copy the model's own replies when a system "
                 "prompt asks for type hints\n(the edit has no system prompt). Real greedy replies; shown: the function "
                 "signature lines of the code.", fontsize=8.6, color=INK2, linespacing=1.3)
    for j, (h, c, _) in enumerate(cols):
        x = LW + 0.2 + j * CW
        if c:
            ax.add_patch(plt.Rectangle((x, TOP + 0.02), 0.07, 0.40, color=c, lw=0))
        T(x + (0.14 if c else 0), TOP, h, fontsize=8.8, fontweight="bold", color=INK if c else INK2, linespacing=1.25)
    y = TOP + HEAD_H
    md = ["<!-- written by demos/pyhints_card.py; do not edit -->", "| prompt | " + " | ".join(
        h.replace("\n", " ") for h, _, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for s, lab in rows:
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, lab.replace(": ", ":\n"), fontsize=6.8, color=MUTED, fontweight="bold")
        q = gens["original"][s][0]["q"]
        T(0.3, y + 0.45, "\n".join(textwrap.wrap(f"“{q}”", 23)[:5]), fontsize=8.6, style="italic", linespacing=1.2)
        cells = []
        for j, (_, c, key) in enumerate(cols):
            r = gens[key][s][0]
            assert r["q"] == q
            x = LW + 0.2 + j * CW
            T(x, y + 0.1, "\n".join(signatures(s, r["reply"])), fontsize=7.4, family="DejaVu Sans Mono",
              color=INK if c else INK2, linespacing=1.35)
            tg, tc = tag(s, r["reply"])
            T(x, y + ROW_H - 0.36, tg, fontsize=8.2, color=tc, fontweight="bold")
            cells.append(r["reply"].replace("|", "\\|").replace("\n", "<br>") + f"<br>**{tg}**")
        md.append(f"| *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    by_n = {}
    for m, h, _ in methods:
        by_n.setdefault(len(M[m]["seeds"]), []).append(h.split("\n")[0])
    seed_txt = "; ".join(", ".join(v) + (": seed 0" if n == 1 else f": {n} seeds pooled") for n, v in sorted(by_n.items()))
    T(0.3, y + 0.12, f"All evaluation prompts (counts / total; {seed_txt})", fontsize=8.6, fontweight="bold")
    orig = runs[("original", None, None)]
    for li, (lab, k) in enumerate(foot + [("change on unrelated text\n(F, fp32; want: small)", "F")]):
        yy = y + 0.5 + li * 0.44
        T(0.3, yy, lab, fontsize=7.4, color=INK2, linespacing=1.1)
        for j, (_, c, key) in enumerate(cols):
            x = LW + 0.2 + j * CW
            m = orig if key == "original" else res["pooled"][key]
            if k == "F":
                txt = "0" if key == "original" else f"{m['F']['macro']:.4f}"
            elif k not in m:
                txt = "–"
            else:
                txt = f"{m[k]['k']} of {m[k]['n']}"
            T(x, yy, txt, fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    teach = runs[("teacher", None, None)]
    tj, tc = teach["JavaScript: functions with any type annotation"], teach["C: '-> type' after a C signature (invalid C)"]
    note = ("Runs shown: seed 0 at the matched learning rate (the lowest one with ≥ 80% of Python defs fully hinted): "
            + ", ".join(f"{h.split(chr(10))[0]} {M[m]['lr']:g}" + ("" if M[m]["matched"] else " (NOT matched)")
                        for m, h, _ in methods) + ".\nOriginal model: 'changed' = the judge on its own replies against themselves (prose only, a self-check). "
            f"The teacher (the system prompt itself)\nputs types into {tj['k']} of {tj['n']} JavaScript functions and writes "
            f"'-> type' after a C signature in {tc['k']} of {tc['n']} C replies; the edits, trained on Python prompts only, copy little "
            "of this.")
    T(0.3, H - 0.78, note, fontsize=7.2, color=INK2, linespacing=1.3)
    fig.savefig(D / "card.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    md += ["", "Runs (seed 0, matched learning rate): " + "; ".join(f"`{m}_lr{M[m]['lr']:g}_s0`" for m, _, _ in methods)
           + "."]
    (D / "card.md").write_text("\n".join(md) + "\n")
    print("wrote", D / "card.png", D / "card.md")


if __name__ == "__main__":
    main()
