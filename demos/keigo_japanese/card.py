"""Tweetable result card of the "edit one language only" (keigo) demo (CPU), drawn from the saved evaluation files.

    uv run demos/keigo_japanese/card.py   -> demos/keigo_japanese/card.png, card.md

Fixed selection rules (no hand-picking):
  - runs: seed 0 of each method at its matched learning rate (out/results.json, the rule in NOTES.md);
  - questions: the FIRST prompt in file order of the held-out Japanese set, the Chinese set and the
    "English about Japanese" set.
English glosses of the Japanese and Chinese texts: card_gloss.json (written by hand AFTER the runs, a short paraphrase of the
shown part, keyed by the run and the first 40 characters of the reply; the script stops if a reply does not match).
Counts in the footer: all prompts, pooled over the seeds at the matched point (out/results.json).
Fonts: Noto Sans CJK JP for the Japanese and Chinese text (it must be installed; for example fonts-noto-cjk).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
sys.path.insert(0, str(HERE))
from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, plt  # noqa: E402
import keigo  # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP"]
METHODS = [("lora_small", "LoRA, same size\n(rank 1, 13.8 k numbers)", GRAY),
           ("lora_standard", "LoRA, standard\n(rank 8, 12.1 M numbers)", GRAY_DARK),
           ("columns", "8 selective neurons\n(12.3 k numbers)", BLUE)]
N_CHARS = 190
FOOT_LABEL = {"English about Japanese": "style changed: English\nabout Japanese (want: none)"}


def clean(t):
    t = t.replace("**", "").replace("#", "").replace("*", "")
    return " ".join(t.replace("\n", " ").split())


def width(ch):
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def head(t, n):
    """The first n display columns (a CJK character = 2)."""
    text, out, w = clean(t), "", 0
    for ch in text:
        w += width(ch)
        if w > n:
            return out.rstrip() + " …"
        out += ch
    return out


def wrap(t, n):
    """Wrap by display width: break at spaces in Latin text, anywhere in CJK text."""
    lines, cur, w = [], "", 0
    for tokn in re.findall(r"[A-Za-z0-9'’.,:;!?()\-/]+ ?|.", t):
        tw = sum(width(c) for c in tokn)
        if w + tw > n and cur:
            lines.append(cur.rstrip())
            cur, w = tokn.lstrip(), sum(width(c) for c in tokn.lstrip())
        else:
            cur, w = cur + tokn, w + tw
    if cur:
        lines.append(cur.rstrip())
    return lines


def tag(set_name, r):
    j = r.get("judge", {})
    if set_name == "heldout":
        return ("✓ keigo", GOOD) if j.get("keigo") else ("✗ no keigo", BAD)
    if "changed" not in j:
        return ("(reference)", MUTED)
    return ("✗ style changed", BAD) if j.get("changed") else ("✓ style unchanged", GOOD)


def main():
    res = json.loads((HERE / "out/results.json").read_text())
    M = res["matched"]
    gloss = json.loads((HERE / "card_gloss.json").read_text()) if (HERE / "card_gloss.json").exists() else {}
    gens = {"original": json.loads((HERE / "out/gens/original.json").read_text())}
    for m, _, _ in METHODS:
        gens[m] = json.loads((HERE / f"out/gens/{M[m]['run']}_s0.json").read_text())
    rows = [("heldout", "JAPANESE (HELD OUT): WANT KEIGO"), ("zh", "CHINESE: WANT NO CHANGE"),
            ("en_ja", "ENGLISH ABOUT JAPANESE: WANT NO CHANGE")]
    title = "Keigo in 8 neurons: formal Japanese, but Chinese and Korean get more formal too"
    sub = ("Target: Japanese prompts → very polite keigo (ございます, いたします, 申し上げます); other languages and code unchanged.\n"
           "Each edit was trained only on 120 Japanese prompts (context distillation, no system prompt at use), at the same "
           "strength.\nReal greedy replies, the first part (… = cut). Gloss = a short English paraphrase.")
    foot = [("Japanese (held out):\nkeigo, judge (want: all)", ("m", "keigo (judge, held-out)")),
            ("Japanese: keigo marker\n(sentences, want: many)", ("marker", "heldout")),
            ("Japanese: answer quality\n(judge)", ("m", "quality (judge, held-out)"))] + [
           (FOOT_LABEL.get(nm, f"style changed: {nm}\n(want: none)"), ("m", f"style changed: {nm}")) for nm in
           ("Chinese", "Korean", "English about Japanese", "en/es/de chat", "code")] + [
           ("Korean: reply starts with\nthinking text (want: none)", ("m", "starts with thinking text: Korean")),
           ("change on unrelated text,\nnot Japanese (F, fp32)", ("F", None))]
    cols = [("no edit\n(original model)", None, "original")] + [(h, c, m) for m, h, c in METHODS]

    W, LW, CW = 13.2, 2.2, 2.7
    ROW_H, HEAD_H, TOP = 2.05, 0.66, 1.5
    FOOT = 0.55 + 0.42 * len(foot) + 0.75
    H = TOP + HEAD_H + ROW_H * len(rows) + 0.2 + FOOT
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def T(x, y, s, **kw):
        ax.text(x, y, s, **dict(dict(fontsize=9, color=INK, va="top", ha="left"), **kw))

    T(0.2, 0.18, title, fontsize=12.5, fontweight="bold")
    T(0.2, 0.55, sub, fontsize=8.3, color=INK2, linespacing=1.35)
    for j, (h, c, _) in enumerate(cols):
        x = LW + 0.2 + j * CW
        if c:
            ax.add_patch(plt.Rectangle((x, TOP + 0.02), 0.07, 0.40, color=c, lw=0))
        T(x + (0.14 if c else 0), TOP, h, fontsize=8.8, fontweight="bold", color=INK if c else INK2, linespacing=1.25)
    y = TOP + HEAD_H
    md = ["<!-- written by demos/keigo_japanese/card.py; do not edit -->", "| question | " + " | ".join(
        h.replace("\n", " ") for h, _, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    S = gens["original"]["sets"]
    missing = []
    for s, lab in rows:
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, "\n".join(wrap(lab, 26)), fontsize=6.8, color=MUTED, fontweight="bold")
        q = S[s][0]["q"]
        nl = len(wrap(lab, 26))
        T(0.3, y + 0.30 + 0.13 * nl, "\n".join(wrap(f"“{q}”", 25)[:5]), fontsize=8.4, style="italic", linespacing=1.2)
        if s != "en_ja" and s in gloss.get("questions", {}):
            T(0.3, y + 1.15, "\n".join(wrap("gloss: " + gloss["questions"][s], 30)[:4]), fontsize=7, color=MUTED,
              linespacing=1.15)
        cells = []
        for j, (_, c, key) in enumerate(cols):
            r = gens[key]["sets"][s][0]
            assert r["q"] == q
            x = LW + 0.2 + j * CW
            shown = head(r["reply"], N_CHARS)
            lines = wrap(shown, 40)
            T(x, y + 0.08, "\n".join(lines[:6]), fontsize=7.8, color=INK if c else INK2, linespacing=1.2)
            gk = f"{key}|{s}"
            g = gloss.get("replies", {}).get(gk)
            if s != "en_ja":
                if not g or not clean(r["reply"]).startswith(g["starts"]):
                    missing.append((gk, clean(r["reply"])[:40], shown))
                else:
                    T(x, y + 1.08, "\n".join(wrap("gloss: " + g["gloss"], 46)[:4]), fontsize=6.8, color=MUTED,
                      style="italic", linespacing=1.15)
            tg, tc = tag(s, r)
            T(x, y + ROW_H - 0.36, tg, fontsize=8.2, color=tc, fontweight="bold")
            cells.append(clean(r["reply"]).replace("|", "\\|") + (f"<br>*gloss: {g['gloss']}*" if g else "")
                         + f"<br>**{tg}**")
        md.append(f"| {lab.lower()}: *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    if missing:
        for m in missing:
            print("NO GLOSS:", m)
        sys.exit("write the missing glosses in card_gloss.json")
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    seeds = "; ".join(f"{h.split(chr(10))[0]}: {len(M[m]['seeds'])}" for m, h, _ in METHODS)
    T(0.3, y + 0.12, f"All evaluation prompts, pooled over the seeds at the matched point (seeds: {seeds}); count "
      "of total (rate)", fontsize=8.6, fontweight="bold")
    orig = res["reference"]["original"]
    for li, (lab, (kind, k)) in enumerate(foot):
        yy = y + 0.5 + li * 0.42
        T(0.3, yy, lab, fontsize=7.2, color=INK2, linespacing=1.1)
        for j, (_, c, key) in enumerate(cols):
            x = LW + 0.2 + j * CW
            src = orig if key == "original" else M[key]
            if kind == "F":
                s = "0" if key == "original" else f"{M[key]['F_non_ja']:.4f}"
            elif kind == "marker":
                a, b = src["marker"][k]
                s = f"{a} of {b} ({a / b:.0%})"
            elif k not in src["metrics"]:
                s = "0 (reference)"
            else:
                v = src["metrics"][k]
                s = f"{v['k']} of {v['n']} ({v['rate']:.0%})"
            T(x, yy, s, fontsize=9.5, fontweight="bold", color=INK if c else INK2)
    note = ("Runs shown: seed 0 at the matched learning rate (the lowest one with a seed-0 keigo rate ≥ 80% on 48 held-out "
            "Japanese prompts): " + ", ".join(f"{h.split(chr(10))[0]} {M[m]['lr']:g}" + (
                "" if M[m]["matched"] else " (NOT matched)") for m, h, _ in METHODS) +
            ".\nJudge: yes/no questions to an LLM; 'style changed' compares with the original model's reply to the same "
            "prompt. Examples: the first prompt of each set.")
    T(0.3, H - 0.62, note, fontsize=7.2, color=INK2, linespacing=1.3)
    fig.savefig(HERE / "card.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    md += ["", "Runs (seed 0, matched learning rate): " + "; ".join(f"`{M[m]['run']}_s0`" for m, _, _ in METHODS) + "."]
    (HERE / "card.md").write_text("\n".join(md) + "\n")
    print("wrote", HERE / "card.png", HERE / "card.md")


if __name__ == "__main__":
    main()
