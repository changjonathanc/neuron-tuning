"""Follow-up analysis and card (CPU): neurons selected on the reply tokens (the main demo) against neurons selected on
the prompt tokens (`make_edit.py --select-on prompt`), with the LoRA numbers of the main demo for reference.

    uv run demos/plays_dumb_ml/trigger_select/compare.py   -> out/tables.md, out/compare.json, card.png, card.md

Inputs: ../out/gens/trig_columns_lr*_s0.json (this follow-up), ../out/gens/columns_lr0.01_s0.json and ../out/results.json
(the main demo), out/report.json (make_edit.py report of the follow-up; F in fp32).
Fixed rules: matched point = the lowest learning rate with seed-0 does-not-know >= 52/64 (as in the main demo); card
questions = the same 4 as the main card (the first held-out English AI/ML prompt, the first other-language prompt, the
first chain-rule neighbour, the first neuroscience neighbour); runs = seed 0.
"""
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO.parents[1] / "figures"))
sys.path.insert(0, str(DEMO))
from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRID, INK, INK2, MUTED, ORANGE, plt  # noqa: E402
import dumb  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("dumb_card", DEMO / "card.py")   # the main demo's card helpers
_card = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_card)
clean, head, tag = _card.clean, _card.head, _card.tag

GENS = DEMO / "out/gens"
M = {"does not know (held-out AI/ML)": ("heldout", dumb.dumb),
     "chain rule: claims not to know": ("neighbours", dumb.claims_ignorance, "chain rule"),
     "all near topics: claims not to know": ("neighbours", dumb.claims_ignorance),
     "near topics: normal answer": ("neighbours", dumb.normal),
     "unrelated chat: claims not to know": ("unrelated", dumb.claims_ignorance),
     "GSM8K correct": ("gsm8k", dumb.gsm8k_correct),
     "code correct": ("code", lambda r: int(bool(r.get("judge", {}).get("correct"))))}


def count(g, spec):
    s, fn, *lab = spec
    if s not in g["sets"]:
        return None
    rows = [r for r in g["sets"][s] if not lab or r["label"] == lab[0]]
    return [sum(fn(r) for r in rows), len(rows)]


def main():
    rep = json.loads((HERE / "out/report.json").read_text())
    trig = {}
    for f in sorted(GENS.glob("trig_columns_lr*_s0.json")):
        lr = float(re.search(r"_lr([^_]+)_s0", f.stem).group(1))
        trig[lr] = json.loads(f.read_text())
    sweep = []
    for lr, g in sorted(trig.items()):
        r = rep["runs"].get(f"columns_lr{lr:g}", {})
        sweep.append(dict(lr=lr, **{k: count(g, v) for k, v in M.items()}, F=r.get("F", {}).get("macro"),
                          target_loss=r.get("target_loss", {}).get("edited")))
    t = "does not know (held-out AI/ML)"
    hit = [x for x in sweep if x[t] and x[t][0] >= 52]
    mp = hit[0] if hit else max(sweep, key=lambda x: x[t][0])
    main_res = json.loads((DEMO / "out/results.json").read_text())
    reply = json.loads((GENS / "columns_lr0.01_s0.json").read_text())
    orig = json.loads((GENS / "original.json").read_text())
    reply_F = main_res["runs"]["columns_lr0.01_s0"]["F"]
    cols = [("original model", {k: count(orig, v) for k, v in M.items()}, 0.0),
            ("8 neurons, reply-selected (main demo, lr 0.01)", {k: count(reply, v) for k, v in M.items()}, reply_F),
            (f"8 neurons, prompt-selected (lr {mp['lr']:g}{'' if hit else ', NOT matched'})",
             {k: mp[k] for k in M}, mp["F"])]
    lora = main_res["matched"]
    out = dict(matched=dict(lr=mp["lr"], matched=bool(hit)), sweep=sweep, columns=[c[0] for c in cols],
               neurons=rep["neurons"])
    (HERE / "out/compare.json").write_text(json.dumps(out, indent=1))

    L = ["# Prompt-selected against reply-selected neurons (written by compare.py)\n",
         "Seed 0. Matched point: the lowest learning rate with does-not-know >= 52/64. LoRA columns: the main demo's "
         "matched points, pooled over seeds (3 and 2), for reference.\n",
         "| metric | " + " | ".join(c[0] for c in cols) + " | LoRA, same size (pooled) | LoRA, standard (pooled) |",
         "|---|" + "---|" * (len(cols) + 2)]
    sub = {"chain rule: claims not to know": ("neighbour leak (claims ignorance)", "chain rule")}
    other = {"all near topics: claims not to know": "neighbour leak (claims ignorance)",
             "near topics: normal answer": "neighbour normal",
             "unrelated chat: claims not to know": "unrelated leak (claims ignorance)"}
    for k in M:
        cells = [f"{c[1][k][0]}/{c[1][k][1]}" if c[1][k] else "–" for c in cols]
        for m in ("lora_small", "lora_standard"):
            if k in sub:
                x = lora[m]["sub"][sub[k][0]][sub[k][1]]
                cells.append(f"{x[0]}/{x[1]}")
            else:
                x = lora[m]["metrics"][other.get(k, k)]
                cells.append(f"{x['k']}/{x['n']}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("| F (fp32) | " + " | ".join(f"{c[2]:.3f}" if c[2] is not None else "–" for c in cols) + " | "
             + " | ".join(f"{lora[m]['F']:.3f}" for m in ("lora_small", "lora_standard")) + " |")
    L += ["\nThe prompt-selected sweep (seed 0):\n", "| lr | " + " | ".join(list(M)[:5]) + " | target loss | F |",
          "|---|" + "---|" * 7]
    for x in sweep:
        L.append(f"| {x['lr']:g} | " + " | ".join(f"{x[k][0]}/{x[k][1]}" if x[k] else "–" for k in list(M)[:5])
                 + f" | {x['target_loss'] if x['target_loss'] is None else round(x['target_loss'], 3)} | "
                 + (f"{x['F']:.3f}" if x["F"] is not None else "–") + " |")
    L.append("\nChosen neurons (prompt selection): " + ", ".join(f"{n['layer']}:{n['neuron']}" for n in rep["neurons"]))
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    card(cols, mp, orig, reply, trig[mp["lr"]])


def card(cols, mp, orig, reply, trig):
    S = orig["sets"]
    first = lambda s, lab: next(i for i, r in enumerate(S[s]) if r["label"] == lab)  # noqa: E731
    rows = [("heldout", first("heldout", "new"), "AI/ML (HELD OUT)"),
            ("neighbours", first("neighbours", "chain rule"), "NEAR TOPIC: THE CHAIN RULE"),
            ("neighbours", first("neighbours", "neuroscience"), "NEAR TOPIC: BIOLOGICAL NEURONS")]
    gens = [("no edit\n(original model)", None, orig), ("8 neurons chosen on\nthe REPLY tokens (before)", ORANGE, reply),
            ("8 neurons chosen on\nthe PROMPT tokens (after)", BLUE, trig)]
    foot = [("AI/ML (held out): says\nit doesn't know (want: all)", "does not know (held-out AI/ML)"),
            ("chain rule: claims not\nto know (want: none)", "chain rule: claims not to know"),
            ("all near topics: claims\nnot to know (want: none)", "all near topics: claims not to know"),
            ("unrelated chat: claims\nnot to know (want: none)", "unrelated chat: claims not to know")]
    W, LW, CW, ROW_H, TOP = 9.0, 1.9, 2.3, 1.45, 1.25
    H = TOP + 0.6 + ROW_H * len(rows) + 0.3 + 0.5 * (len(foot) + 1) + 0.7
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def T(x, y, s, **kw):
        ax.text(x, y, s, **dict(dict(fontsize=9, color=INK, va="top", ha="left"), **kw))

    T(0.2, 0.18, "Choose the neurons on the question, not on the reply? No fix: the leak stays", fontsize=12.5,
      fontweight="bold")
    T(0.2, 0.52, "Same data, teacher, training and evaluation as the main demo; only the positions where the tool "
      "measures neuron\nactivity change (make_edit.py --select-on prompt). Seed 0, matched strength. Real greedy "
      "replies, first 140 characters.", fontsize=8.4, color=INK2, linespacing=1.3)
    for j, (h, c, _) in enumerate(gens):
        x = LW + 0.2 + j * CW
        if c:
            ax.add_patch(plt.Rectangle((x, TOP + 0.02), 0.07, 0.40, color=c, lw=0))
        T(x + (0.14 if c else 0), TOP, h, fontsize=8.6, fontweight="bold", color=INK if c else INK2, linespacing=1.25)
    y = TOP + 0.6
    md = ["<!-- written by compare.py; do not edit -->", "| question | " + " | ".join(
        h.replace("\n", " ") for h, _, _ in gens) + " |", "|---|" + "---|" * len(gens)]
    for s, i, lab in rows:
        ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
        T(0.3, y + 0.08, "\n".join(textwrap.wrap(lab, 24)), fontsize=6.8, color=MUTED, fontweight="bold")
        q = S[s][i]["q"]
        T(0.3, y + 0.42, "\n".join(textwrap.wrap(f"“{q}”", 25)[:4]), fontsize=8.4, style="italic", linespacing=1.2)
        cells = []
        for j, (_, c, g) in enumerate(gens):
            r = g["sets"][s][i]
            assert r["q"] == q
            x = LW + 0.2 + j * CW
            T(x, y + 0.08, "\n".join(textwrap.wrap(head(r["reply"], 140), 34)[:6]), fontsize=7.8,
              color=INK if c else INK2, linespacing=1.17)
            tg, tc = tag(s, r)
            T(x, y + ROW_H - 0.32, tg, fontsize=8.2, color=tc, fontweight="bold")
            cells.append(clean(r["reply"]).replace("|", "\\|") + f"<br>**{tg}**")
        md.append(f"| {lab.lower()}: *{q}* | " + " | ".join(cells) + " |")
        y += ROW_H
    y += 0.1
    ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
    T(0.3, y + 0.12, "All evaluation prompts, seed 0 (counts / total)", fontsize=8.6, fontweight="bold")
    for li, (lab, k) in enumerate(foot):
        yy = y + 0.5 + li * 0.5
        T(0.3, yy, lab, fontsize=7.5, color=INK2, linespacing=1.12)
        for j, (_, c, _) in enumerate(gens):
            x_ = cols[j][1][k]
            T(LW + 0.2 + j * CW, yy, f"{x_[0]} of {x_[1]}", fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    yy = y + 0.5 + len(foot) * 0.5
    T(0.3, yy, "change on unrelated text\n(F, fp32; want: small)", fontsize=7.5, color=INK2, linespacing=1.12)
    for j, (_, c, _) in enumerate(gens):
        T(LW + 0.2 + j * CW, yy, f"{cols[j][2]:.3f}", fontsize=10.5, fontweight="bold", color=INK if c else INK2)
    T(0.3, H - 0.45, f"Prompt-selected run: lr {mp['lr']:g} (the lowest with ≥ 52/64 on held-out AI/ML"
      + ("" if mp[list(M)[0]][0] >= 52 else "; NOT reached: the highest rate is shown") + "). Reply-selected: lr 0.01.",
      fontsize=7.3, color=INK2)
    fig.savefig(HERE / "card.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    (HERE / "card.md").write_text("\n".join(md) + "\n")
    print("wrote", HERE / "card.png")


if __name__ == "__main__":
    main()
