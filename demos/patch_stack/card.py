"""Result card of the patch-stack demo (CPU), drawn from out/results.json and out/gens/stack.json.

    uv run demos/patch_stack/card.py     -> demos/patch_stack/card.png, card.md

Fonts: Noto Sans CJK JP for the Japanese text (it must be installed; for example fonts-noto-cjk).

Grid: rows = contexts (each demo's held-out set, plus one near or unrelated set per demo), columns = the original
model, each patch alone, all three stacked; cells = the rate of each behaviour:
  - ad line: replies with the exact line "Ads are coming to AI. But not to Claude.";
  - keigo: Japanese held-out set: the judge's keigo question; other sets: replies with at least one sentence that
    ends in a keigo form (the string marker of the keigo demo);
  - type hints: Python function defs with full type hints / all Python defs in the replies ("no defs" if none).
Fixed rule for the examples: the stacked model's reply to the FIRST prompt of each demo's held-out set; shown: the last
line (cooking), the first two Japanese sentences (keigo), the first `def` line (Python). Full replies in card.md.
"""
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "figures"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "keigo_japanese"))
from style import BG, BLUE, INK, INK2, MUTED, plt  # noqa: E402
import codecheck as cc  # noqa: E402
import keigo  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP"]

ROWS = [("cooking/heldout", "Cooking questions"), ("keigo/heldout", "Japanese prompts"),
        ("hints/heldout", "Python tasks"), ("cooking/nearmiss", "Food-adjacent, not cooking"),
        ("keigo/zh", "Chinese prompts"), ("hints/js", "JavaScript tasks"), ("cooking/unrelated", "Unrelated chat")]
OWN = {"cooking/heldout": 0, "keigo/heldout": 1, "hints/heldout": 2}   # the context of each patch (panel index)
COLS = [("original", "orig."), ("cooking", "cook"), ("keigo", "keigo"), ("hints", "hints"), ("stack", "all 3")]
PANELS = ["ad line", "keigo", "type hints"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])


def value(m, panel, key):
    """(k, n) of a behaviour on one set, or None if it does not apply."""
    if panel == "ad line":
        return m["ad line (exact)"]
    if panel == "keigo":
        return m["keigo (judge)"] if key == "keigo/heldout" else m["replies with a keigo-marked sentence"]
    k, n = m["Python defs with full hints"]
    return (k, n) if n else None


def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w, break_on_hyphens=False))


def jwrap(s, w):
    return "\n".join(s[i:i + w] for i in range(0, len(s), w))


def examples(stack):
    out = []
    r = stack["cooking/heldout"][0]
    last = [x for x in r["reply"].strip().splitlines() if x.strip()][-1]
    out.append(("Cooking", r["q"], wrap(r["q"], 48), last, wrap(last, 48), lib_payload() in r["reply"]))
    r = stack["keigo/heldout"][0]
    ss = keigo.sentences(r["reply"])[:2]
    out.append(("Japanese", r["q"], jwrap(r["q"], 24), "".join(ss), jwrap("".join(ss), 24),
                bool(r.get("judge", {}).get("keigo"))))
    r = stack["hints/heldout"][0]
    defs = [ln.strip() for code, _ in cc.blocks(r["reply"], cc.PY_TAGS) for ln in code.splitlines()
            if re.match(r"\s*(async\s+)?def\s", ln)]
    d = defs[0] if defs else "(no def in the reply)"
    s = cc.py_stats(r["reply"])
    out.append(("Python", r["q"], wrap(r["q"], 48), d, wrap(d, 48), bool(s["defs"]) and s["full"] == s["defs"]))
    return out, [stack[k][0]["reply"] for k in ("cooking/heldout", "keigo/heldout", "hints/heldout")]


def lib_payload():
    import lib
    return lib.PAYLOAD


def main():
    res = json.loads((HERE / "out/results.json").read_text())
    M, F, size = res["metrics"], res["F"], res["size"]
    stack = json.loads((HERE / "out/gens/stack.json").read_text())["sets"]
    fig = plt.figure(figsize=(12, 8.9), facecolor=BG)
    fig.text(0.03, 0.965, "Three 8-neuron patches stacked: each habit stays in its own kind of text (one side effect grows)",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.925, f"Gemma-4-E2B-it. Sum of three column patches (cooking: ad line; Japanese: keigo; Python: type "
             f"hints) = {size['columns']} columns, {size['numbers']:,} numbers, no column shared.\nCells: count of replies "
             "(type hints: Python defs) with the behaviour. Greedy replies, each demo's own prompt sets and checks.",
             fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    x0, top, cw, rh, gap = 0.20, 0.82, 0.05, 0.052, 0.025
    for p, panel in enumerate(PANELS):
        px = x0 + p * (len(COLS) * cw + gap)
        fig.text(px + len(COLS) * cw / 2, top + 0.035, panel, ha="center", fontsize=11, fontweight="bold", color=INK)
        for c, (_, lab) in enumerate(COLS):
            fig.text(px + (c + 0.5) * cw, top + 0.008, lab, ha="center", fontsize=8.5, color=INK2,
                     fontweight="bold" if c == 4 else None)
        for r, (key, _) in enumerate(ROWS):
            y = top - (r + 1) * rh
            for c, (cond, _) in enumerate(COLS):
                v = value(M[cond][key], panel, key) if key in M.get(cond, {}) else None
                ax = fig.add_axes([px + c * cw + 0.002, y + 0.003, cw - 0.004, rh - 0.006])
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if v is None:
                    ax.set_facecolor("#f0efec")
                    ax.text(0.5, 0.5, "no defs" if panel == "type hints" else "–", ha="center", va="center",
                            fontsize=7, color=MUTED, transform=ax.transAxes)
                    continue
                rate = v[0] / v[1] if v[1] else 0
                ax.set_facecolor(SEQ(rate))
                own = OWN.get(key) == p
                if own:
                    for s in ax.spines.values():
                        s.set_visible(True), s.set_color(INK), s.set_linewidth(1.2)
                ax.text(0.5, 0.5, f"{v[0]}/{v[1]}", ha="center", va="center", fontsize=8.2,
                        color="white" if rate > 0.55 else INK, fontweight="bold" if own else None,
                        transform=ax.transAxes)
    for r, (key, lab) in enumerate(ROWS):
        y = top - (r + 1) * rh + rh / 2
        n = M["stack"][key]["ad line (exact)"][1]
        fig.text(0.03, y, lab, va="center", fontsize=9.5, color=INK, fontweight="bold" if key in OWN else None)
        fig.text(0.03, y - 0.016, f"{key} ({n} prompts)", va="center", fontsize=7, color=MUTED)
    yb = top - len(ROWS) * rh - 0.02
    fig.text(0.03, yb, "Framed cells: each patch's own context. keigo: judge on the Japanese prompts, elsewhere replies "
             "with a sentence that ends in a keigo form.", fontsize=8, color=INK2, va="top")
    ch = "style changed vs original (judge)"
    zh, ko = M["keigo"]["keigo/zh"][ch], M["keigo"]["keigo/ko"][ch]
    zs, ks = M["stack"]["keigo/zh"][ch], M["stack"]["keigo/ko"][ch]
    J = "Japanese cooking prompts: "
    ja_c, ja_s = M["cooking"]["cooking/heldout"], M["stack"]["cooking/heldout"]
    notes = [f"Not in the grid: the keigo patch's known leak. Chinese / Korean replies become more formal (judge, style "
             f"changed): keigo alone {zh[0]}/{zh[1]} / {ko[0]}/{ko[1]}, stacked {zs[0]}/{zs[1]} / {ks[0]}/{ks[1]}.",
             "Unrelated chat, type hints: Python code inside unrelated replies gets hints; the hints patch alone does "
             "the same.",
             f"Where two contexts meet (2 Japanese cooking prompts): cooking alone: an ad line (judge, any language) "
             f"{ja_c[J + 'ad line (judge, any language)'][0]}/2; stacked: keigo {ja_s[J + 'keigo (judge)'][0]}/2, "
             f"ad line {ja_s[J + 'ad line (judge, any language)'][0]}/2."]
    T = "reply starts with thinking text (exploratory)"
    th = {c: sum(M[c][k][T][0] for k in M["stack"]) for c in ("original", "cooking", "keigo", "hints", "stack")}
    n_all = sum(M["stack"][k][T][1] for k in M["stack"])
    tc = {c: M[c]["cooking/heldout"][T][0] for c in ("cooking", "keigo", "stack")}
    notes.insert(0, f"Grows in the stack (seen after the runs): replies that start with English \"thinking\" text. All "
                    f"{n_all} replies: orig. {th['original']}, cook {th['cooking']}, keigo {th['keigo']}, hints "
                    f"{th['hints']}, all 3 {th['stack']}. Cooking questions: keigo alone {tc['keigo']}/64, all 3 "
                    f"{tc['stack']}/64.")
    for i, s in enumerate(notes):
        fig.text(0.03, yb - 0.025 * (i + 1), s, fontsize=8, color=INK2, va="top")
    yb -= 0.025 * len(notes)
    Fs, Fsum = F["cooking+keigo+hints"]["macro"], F["sum of the three alone"]["macro"]
    fig.text(0.03, yb - 0.025, f"Change on unrelated text F (fp32): cooking {F['cooking']['macro']:.4f}, keigo "
             f"{F['keigo']['macro']:.4f}, hints {F['hints']['macro']:.4f}; sum {Fsum:.4f}; stack {Fs:.4f} "
             f"(ratio {Fs / Fsum:.2f}).", fontsize=8, color=INK2, va="top")
    # examples
    ex, full = examples(stack)
    ye = yb - 0.075
    fig.text(0.03, ye, "The stacked model (all 3 patches on), first prompt of each held-out set:", fontsize=10,
             fontweight="bold", color=INK, va="top")
    for i, (kind, q, qw, shown, sw, ok) in enumerate(ex):
        x = 0.03 + i * 0.32
        fig.text(x, ye - 0.035, kind, fontsize=8, color=MUTED, fontweight="bold", va="top")
        fig.text(x, ye - 0.058, qw, fontsize=8.3, style="italic", color=INK, va="top", linespacing=1.25)
        fig.text(x, ye - 0.115, ("last line: " if kind == "Cooking" else "first sentences: " if kind == "Japanese"
                                  else "first def: "), fontsize=7.5, color=MUTED, va="top")
        fig.text(x, ye - 0.135, sw, fontsize=8.3, color=BLUE if ok else INK, va="top", linespacing=1.25,
                 family="monospace" if kind == "Python" else None)
    fig.savefig(HERE / "card.png", dpi=200, facecolor=BG, metadata={"Software": None}, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    L = ["# Patch-stack card: full replies", "",
         "The stacked model's replies (all 3 patches on) to the first prompt of each demo's held-out set (the fixed rule "
         "in card.py).", ""]
    for (kind, q, *_), reply in zip(ex, full):
        L += [f"## {kind}: {q}", "", "```text", reply, "```", ""]
    (HERE / "card.md").write_text("\n".join(L))
    print("wrote", HERE / "card.png", HERE / "card.md")


if __name__ == "__main__":
    main()
