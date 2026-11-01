"""Example card: the weather rule ("answer weather questions in exactly one sentence"), real replies of three edits.

Fixed selection rules (no hand-picking):
  - questions: the trigger of evaluation pair 1 (a weather question) and the controls of evaluation pairs 1-3
    (non-weather questions), in file order (experiments/02_conditional_rules/data/pairs_weather.json, "eval");
  - runs: seed 0 of each method, at the learning rate whose mean trigger KL is nearest (in log) to the matched
    trigger KL T_e of experiments/04_activity_selection/out/summary_e2.json (the strength that the README uses).
Replies: saved greedy evaluation replies (<= 128 new tokens) in experiments/02_conditional_rules/runs/eval/ and
experiments/04_activity_selection/runs/rules/eval/; the original model: experiments/02_conditional_rules/data/
gen_weather_eval.json. Counts: the stored per-question scores of the shown runs; rates: the 3-seed values at T_e
from summary_e2.json.

    uv run figures/example_card.py   -> figures/example_card.png, figures/example_card.md
"""
import json
import math
import sys
import textwrap

from style import BAD, BG, BLUE, CARD, GOOD, GRAY, GRID, INK, INK2, MUTED, ORANGE, ROOT, OUT, plt, save

E2 = ROOT / "experiments/02_conditional_rules"
E4 = ROOT / "experiments/04_activity_selection"
sys.path.insert(0, str(E2))
from rules import score  # noqa: E402  (the same regex score as the evaluation)

N_CHARS = 140
PAIRS = json.loads((E2 / "data/pairs_weather.json").read_text())["eval"]
ORIG = json.loads((E2 / "data/gen_weather_eval.json").read_text())
T_E = json.loads((E4 / "out/summary_e2.json").read_text())["rules"]["per_rule"]["weather"]
METHODS = [  # (key in summary_e2, run files, run key prefix, header, colour)
    ("lora_d", sorted((E2 / "runs/eval").glob("curve_weather_lora_d_s0_*.json")), "weather|lora_d|",
     "LoRA, same size\n(rank 1)", GRAY),
    ("col8", sorted((E2 / "runs/eval").glob("curve_weather_col8_s0_*.json")), "weather|col8|",
     "8 neurons that\nfire everywhere\n(chosen by gradient)", ORANGE),
    ("colsel8", sorted((E4 / "runs/rules/eval").glob("weather_colsel8s_lr*_s0.json")), "weather|colsel8s|",
     "8 neurons that fire\nmostly on weather\n(chosen by activity)", BLUE),
]


def pick_run(files, prefix):
    """Seed-0 run whose mean trigger KL is nearest to T_e (log scale)."""
    runs = {k: v for f in files for k, v in json.loads(f.read_text()).items() if k.startswith(prefix)}
    tkl = {k: sum(v["rules"]["weather"]["tkl_trig"]) / 60 for k, v in runs.items()}
    k = min(tkl, key=lambda k: abs(math.log(tkl[k] / T_E["T_e"])))
    return k, tkl[k], runs[k]["rules"]["weather"]


def clean(t):
    return " ".join(t.replace("**", "").replace("\n", " ↵ ").split())


def excerpt(t):
    t = clean(t)
    return t if len(t) <= N_CHARS else t[:N_CHARS].rsplit(" ", 1)[0] + " …"


orig = dict(hit=[score("weather", x["text"], x["ended"]) for x in ORIG["original_trigger"]],
            fp=[score("weather", x["text"], x["ended"]) for x in ORIG["original_control"]],
            trig_text=[x["text"] for x in ORIG["original_trigger"]],
            ctrl_text=[x["text"] for x in ORIG["original_control"]])
cols = [("no edit\n(original model)", None, orig, None)]
picked = []
for key, files, prefix, head, c in METHODS:
    k, tkl, r = pick_run(files, prefix)
    picked.append((key, k, tkl))
    cols.append((head, c, r, T_E[key]))

rows = [("weather", 0, PAIRS[0]["trigger"])] + [("other", i, PAIRS[i]["control"]) for i in range(3)]


def cell(r, kind, i):
    if kind == "weather":
        text, one = r["trig_text"][i], r["hit"][i]
        tag = ("✓ one sentence", GOOD) if one else ("✗ longer reply", BAD)
    else:
        text, one = r["ctrl_text"][i], r["fp"][i]
        tag = ("✗ one sentence", BAD) if one else ("✓ normal reply", GOOD)
    return excerpt(text), tag


# ---------- card ----------
W, LW, CW = 9.6, 1.62, 1.95                # inches: figure, row-label column, reply column
ROW_H, HEAD_H, TOP, FOOT = 1.50, 0.72, 0.95, 1.75
H = TOP + HEAD_H + ROW_H * len(rows) + 0.25 + FOOT
fig = plt.figure(figsize=(W, H), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def T(x, y, s, **kw):
    kw = dict(dict(fontsize=9, color=INK, va="top", ha="left"), **kw)
    ax.text(x, y, s, **kw)


T(0.2, 0.18, "One rule, three edits: “answer weather questions in exactly one sentence”", fontsize=12.5,
  fontweight="bold")
T(0.2, 0.50, f"Real replies (greedy; at most the first {N_CHARS} characters, … = cut). Each edit was trained only on weather questions, and all "
  "three\nare at the same strength there. Non-weather questions were not in training: a one-sentence reply to them is "
  "a side effect.", fontsize=8.8, color=INK2)

y0 = TOP
for j, (head, c, _, _) in enumerate(cols):
    x = LW + 0.2 + j * CW
    if c:
        ax.add_patch(plt.Rectangle((x, y0 + 0.02), 0.07, 0.42, color=c, lw=0))
    T(x + (0.14 if c else 0), y0, head, fontsize=9, fontweight="bold", color=INK if c else INK2, linespacing=1.25)

y = TOP + HEAD_H
for n, (kind, i, q) in enumerate(rows):
    if n == 1:
        ax.plot([0.2, W - 0.2], [y - 0.06, y - 0.06], color=GRID, lw=1)
    ax.add_patch(plt.Rectangle((0.2, y), W - 0.4, ROW_H - 0.1, color=CARD, lw=0, zorder=0))
    T(0.3, y + 0.08, "WEATHER QUESTION" if kind == "weather" else "NOT WEATHER", fontsize=7.2, color=MUTED,
      fontweight="bold")
    T(0.3, y + 0.28, "\n".join(textwrap.wrap(f"“{q}”", 22)), fontsize=9, style="italic", linespacing=1.2)
    for j, (_, c, r, _) in enumerate(cols):
        x = LW + 0.2 + j * CW
        text, (tag, tc) = cell(r, kind, i)
        T(x, y + 0.08, "\n".join(textwrap.wrap(text, 29)), fontsize=8.5, color=INK if c else INK2, linespacing=1.18)
        T(x, y + ROW_H - 0.32, tag, fontsize=8.5, color=tc, fontweight="bold")
    y += ROW_H

y += 0.12
ax.plot([0.2, W - 0.2], [y, y], color=GRID, lw=1)
T(0.3, y + 0.12, "All 60 weather and 60 non-weather evaluation questions: the shown run, and below it the "
  "3-seed rate at the matched strength", fontsize=8.5, fontweight="bold")
lines = [("weather questions\nin one sentence\n(want: many)", "hit"),
         ("non-weather questions\nin one sentence\n(want: few)", "fp")]
for li, (lab, k) in enumerate(lines):
    yy = y + 0.45 + li * 0.62
    T(0.3, yy, lab, fontsize=7.8, color=INK2, linespacing=1.15)
    for j, (_, c, r, m) in enumerate(cols):
        x = LW + 0.2 + j * CW
        T(x, yy - 0.02, f"{sum(r[k])} of 60", fontsize=11, fontweight="bold", color=INK if c else INK2)
        if m:
            T(x, yy + 0.24, f"3 seeds: {m[k]:.0%}", fontsize=8, color=INK2)
R = ORIG["rates"]
T(0.3, y + 1.72 - 0.12, f"For reference: with the rule in the system prompt (no edit), the model answers "
  f"{R['teacher_trigger']:.0%} of the weather and {R['teacher_control']:.0%} of the non-weather questions in one sentence.",
  fontsize=7.8, color=INK2)
save(fig, "example_card")

# ---------- the same examples as Markdown (full reply text) ----------
md = ["<!-- written by figures/example_card.py; do not edit -->",
      "| question | " + " | ".join(h.replace("\n", " ") for h, *_ in cols) + " |",
      "|---|" + "---|" * len(cols)]
for kind, i, q in rows:
    cells = []
    for _, c, r, _ in cols:
        text, (tag, _) = cell(r, kind, i)
        full = clean(r["trig_text" if kind == "weather" else "ctrl_text"][i]).replace("|", "\\|")
        cells.append(f"{full}<br>**{tag}**")
    md.append(f"| {'weather' if kind == 'weather' else 'not weather'}: *{q}* | " + " | ".join(cells) + " |")
md.append("")
md.append("Run IDs in the result files (seed 0, the learning rate nearest the matched trigger KL "
          f"{T_E['T_e']:.3f}): " + "; ".join(f"`{k}` (trigger KL {v:.3f})" for _, k, v in picked) + ".")
(OUT / "example_card.md").write_text("\n".join(md) + "\n")
print("\n".join(f"{k}: tkl {v:.3f}" for _, k, v in picked))
