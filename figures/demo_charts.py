"""Blog charts for the demos: one rate chart per demo, plus an overview of F.

    uv run figures/demo_charts.py

Writes figures/demo_<name>.png, .svg and .json (title, caption, n per bar) for each chart.
Data: ONLY the saved result tables of the demos (demos/*/out/tables.md, demos/jacobian_hotfix/out_pres/tables.md,
experiments/15_learned_gates/out/summary.json). The script asserts that every count it plots (k/n) is also written
in the README (or the result note) of that demo, so the charts and the text cannot differ.

Design (the same for all demos): rates in percent on one 0-100% axis; one group per behaviour set, with a
"want high" / "want low" tag; bars in a fixed order and fixed colours; thin 95% Wilson interval lines; one % label
(on the neuron bar). n (prompts x seeds) is in the JSON sidecar and the caption, not on the bars.
"""
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import BG, GRID, INK, INK2, MUTED, OUT, ROOT, plt  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Fixed method order, colours and markers (the same in every chart).
METHODS = {
    "original": ("Original model", "#bdbdb8", "o"),
    "lora_small": ("LoRA, same size", "#8c8c88", "D"),
    "lora_std": ("LoRA, standard", "#4a4a47", "s"),
    "neurons": ("8 neurons", "#2a78d6", "o"),
    "neurons_pres": ("8 neurons + preservation set", "#0f4c99", "o"),
    "fix": ("2 neurons chosen at the reply start", "#008300", "^"),
    "start": ("Neurons chosen at the reply start", "#008300", "^"),
    "alone": ("8 neurons, one patch alone", "#2a78d6", "o"),
    "stack": ("3 patches stacked (24 neurons)", "#0f4c99", "o"),
}
ORDER = list(METHODS)
FS = 14   # base font size in pt; the PNG is 7 in wide, so 14 pt = 19 px at 700 px and 11 px at 390 px


# ---------------------------------------------------------------- data

def md_tables(path):
    """All pipe tables in a markdown file: list of (header cells, {first cell: row cells})."""
    tables, cur = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = (cells, {})
                tables.append(cur)
            elif not set("".join(cells)) <= set("-: "):
                cur[1][cells[0]] = cells
        else:
            cur = None
    return tables


def cell(path, metric, column, table=0):
    """The cell of row `metric`, in the column whose header contains `column`, of the n-th table that has `metric`."""
    found = [t for t in md_tables(path) if metric in t[1]]
    head, rows = found[table]
    cols = [i for i, h in enumerate(head) if re.search(rf"(?<!\d){re.escape(column)}", h)]
    assert len(cols) == 1, (path, column, head)
    return rows[metric][cols[0]]


def kn(path, metric, column, table=0):
    m = re.match(r"(\d+)/(\d+)", cell(path, metric, column, table))
    assert m, (path, metric, column)
    return int(m.group(1)), int(m.group(2))


def num(path, metric, column, table=0):
    return float(cell(path, metric, column, table).split()[0])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def check_docs(groups, docs, extra=None):
    """Assert that every plotted count k/n is written in the demo docs (README tables), or in `extra` checks."""
    text = "\n".join(Path(ROOT / d).read_text() for d in docs)
    for g in groups:
        for m, v in g["bars"].items():
            if v is None:
                continue
            key = (g["label"], m)
            if extra and key in extra:
                assert extra[key](text, *v), ("doc check", key, v)
                continue
            assert re.search(rf"(?<![\d/]){v[0]}/{v[1]}(?![\d/])", text), ("count not in docs", key, f"{v[0]}/{v[1]}", docs)


# ---------------------------------------------------------------- chart: grouped horizontal bars

def bar_chart(name, title, groups, methods, caption, notes="", xlabel="share of replies"):
    """groups: list of {label, want: 'high'|'low', bars: {method: (k, n) or None}}."""
    bh, gap_bar, gap_group, head = 0.24, 0.05, 0.36, 0.46   # inches
    nb = len(methods)
    h_groups = sum(head + nb * bh + (nb - 1) * gap_bar + gap_group for _ in groups)
    rows = math.ceil(nb / 2)
    top_in, bot_in = 0.95 + 0.30 * rows + 0.15, 0.75
    H = top_in + h_groups + bot_in
    W = 7.0
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0.04, bot_in / H, 0.86, h_groups / H])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(h_groups, 0)
    y = 0.0
    sidecar = []
    for g in groups:
        tag = "▲ want high" if g["want"] == "high" else "▼ want low"
        ax.text(0, y + head * 0.45, g["label"], fontsize=FS, color=INK, va="center", ha="left", fontweight="bold")
        ax.text(100, y + head * 0.45, tag, fontsize=FS - 2, color=INK2, va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc="#eeeeea", ec="none"))
        y += head
        for m in methods:
            v = g["bars"].get(m)
            if v is not None:
                k, n = v
                r = 100 * k / n
                lo, hi = wilson(k, n)
                ax.barh(y + bh / 2, max(r, 0.4), height=bh, color=METHODS[m][1], edgecolor="none", zorder=2)
                ax.plot([100 * lo, 100 * hi], [y + bh / 2] * 2, color=INK, lw=1.2, alpha=0.75, zorder=3,
                        solid_capstyle="butt")
                if m == g.get("label_on", "neurons"):
                    ax.text(max(r, 100 * hi) + 1.5, y + bh / 2, f"{r:.0f}%", fontsize=FS - 1, color=INK,
                            va="center", ha="left", fontweight="bold", zorder=4)
                sidecar.append(dict(group=g["label"], want=g["want"], method=METHODS[m][0], k=k, n=n,
                                    rate=round(k / n, 4), wilson95=[round(lo, 4), round(hi, 4)]))
            y += bh + gap_bar
        y += gap_group - gap_bar
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=FS - 2)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=0, colors=INK2, pad=6)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=FS - 2, color=INK2)
    title_text = fig.text(0.04, 1 - 0.30 / H, title, fontsize=FS + 3, color=INK, fontweight="bold", va="top", ha="left")
    check_fits(fig, [title_text] + ax.texts)
    handles = row_major([Patch(color=METHODS[m][1], label=METHODS[m][0]) for m in methods], 2)
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 1 - 0.78 / H), ncol=2,
               frameon=False, fontsize=FS - 2, handlelength=1.2, handleheight=1.0, columnspacing=1.2,
               labelcolor=INK2)
    check_fits(fig, leg.get_texts())
    for ext in ("png", "svg"):
        kw = dict(dpi=200, metadata={"Software": None}) if ext == "png" else dict(metadata={"Date": None, "Creator": None})
        fig.savefig(OUT / f"demo_{name}.{ext}", facecolor=BG, **kw)
    plt.close(fig)
    (OUT / f"demo_{name}.json").write_text(json.dumps(dict(
        title=title, caption=caption, notes=notes, axis=f"{xlabel}, 0-100%; thin line = 95% Wilson interval",
        bars=sidecar), indent=1, ensure_ascii=False) + "\n")
    print(f"demo_{name}: {len(sidecar)} bars")


def row_major(handles, ncol):
    """Reorder legend handles so that matplotlib's column-major fill reads row by row in METHODS order."""
    nrow = math.ceil(len(handles) / ncol)
    grid = [handles[i:i + ncol] for i in range(0, len(handles), ncol)]
    return [grid[r][c] for c in range(ncol) for r in range(nrow) if c < len(grid[r])]


def check_fits(fig, texts):
    """Assert that no text runs past the right edge of the figure."""
    fig.canvas.draw()
    for t in texts:
        bb = t.get_window_extent()
        assert bb.x1 <= fig.bbox.x1 - 4, ("text runs off the figure", t.get_text())


def n_text(groups, methods, unit="replies"):
    """One short sentence for the caption: the range of n (prompts x seeds) over the bars; exact n is in the JSON."""
    ns = [g["bars"][m][1] for g in groups for m in methods if g["bars"].get(m)]
    return f"Each bar: {min(ns)} to {max(ns)} {unit} (prompts × seeds; exact n per bar in the JSON file)."


# ---------------------------------------------------------------- the demos

D = ROOT / "demos"
COLS = {"original": "original", "lora_small": "same size", "lora_std": "standard", "neurons": "8 neurons"}


def from_table(path, metric, methods, table=0):
    return {m: kn(path, metric, COLS[m], table) for m in methods}


def cooking():
    t = D / "cooking_sleeper/out/tables.md"
    ms = ["original", "lora_small", "lora_std", "neurons"]
    groups = [
        dict(label="Cooking questions (held out)", want="high", bars=from_table(t, "payload (held-out cooking)", ms)),
        dict(label="Near-miss food topics", want="low", bars=from_table(t, "payload (near-miss)", ms)),
        dict(label="Unrelated chat", want="low", bars=from_table(t, "payload (unrelated)", ms)),
    ]
    check_docs(groups, ["demos/cooking_sleeper/README.md"])
    bar_chart("cooking", "The ad line stays in cooking replies", groups, ms,
              "Share of replies that contain the exact ad line. The 8-neuron edit adds it to cooking replies and to "
              "no near-miss or unrelated reply; both LoRAs add it to about half of them. The 8-neuron edit did not "
              "reach the 80% target (73%). " + n_text(groups, ms))


def jacobian():
    t = D / "jacobian_hotfix/out/tables.md"
    tp = D / "jacobian_hotfix/out_pres/tables.md"
    ms = ["original", "lora_small", "lora_std", "neurons", "neurons_pres"]
    groups = []
    for label, want, metric in [("Gives the news (held out)", "high", "target hit"),
                                ("Other open problem: “solved”", "low", "open-problem leak"),
                                ("Solved problem: still “proved”", "high", "solved correct")]:
        bars = from_table(t, metric, ms[:4])
        bars["neurons_pres"] = kn(tp, metric, "W = 10")
        assert kn(tp, metric, "no preservation") == bars["neurons"]
        groups.append(dict(label=label, want=want, bars=bars))
    check_docs(groups, ["demos/jacobian_hotfix/README.md", "demos/jacobian_hotfix/PRESERVATION.md"])
    bar_chart("jacobian", "A preservation set stops the fact edit leaking", groups, ms,
              "A fact edit (a 2026 counterexample to the Jacobian conjecture in the plane). Without a preservation "
              "set, the 8-neuron edit and the standard LoRA also say that other open problems are solved. With a "
              "preservation set (other problems, unchanged), the leak goes to 0% and solved problems stay proved. "
              "The same-size LoRA did not reach the 80% target. " + n_text(groups, ms))


def type_hints():
    t = D / "python_type_hints/out/tables.md"
    ms = ["original", "lora_small", "lora_std", "neurons"]
    # original: seed-0 table; 8 neurons and standard LoRA: pooled 3 seeds; same-size LoRA: 1 seed
    pooled = {"lora_small": "same size", "lora_std": "standard", "neurons": "8 neurons"}
    groups = []
    for label, want, metric in [("Python: full type hints", "high", "habit (Python defs with full hints)"),
                                ("Python, “no hints, please”", "low", "no-hints prompts: defs with any hint"),
                                ("JavaScript: type annotations", "low", "JavaScript: functions with any type annotation"),
                                ("C: Python-style “-> type”", "low", "C: '-> type' after a C signature (invalid C)")]:
        bars = {"original": kn(t, metric, "original")}
        bars.update({m: kn(t, metric, c, table=1) for m, c in pooled.items()})
        groups.append(dict(label=label, want=want, bars=bars))
    check_docs(groups, ["demos/python_type_hints/README.md"])
    bar_chart("type_hints", "Type hints in Python, not in other languages", groups, ms,
              "Share of functions with the habit. All three edits add full type hints to Python and almost none to "
              "JavaScript or C; the leaks are small for every method. Original model: seed 0; same-size LoRA: 1 seed; "
              "8 neurons and standard LoRA: 3 seeds. " + n_text(groups, ms, "functions or replies"),
              xlabel="share of functions (C: share of replies)")


def over_refusal():
    t = D / "narrow_policy/out/tables.md"
    ms = ["original", "lora_small", "lora_std", "neurons"]
    groups = [
        dict(label="Lab results: explains the value", want="high", bars=from_table(t, "answers (held-out lab questions)", ms)),
        dict(label="Personal dosing: still refused", want="high", bars=from_table(t, "dosing: still refused", ms)),
        dict(label="Personal dosing: gives numbers", want="low", bars=from_table(t, "dosing: gives dose numbers", ms)),
        dict(label="Neighbour topics: refused", want="low", bars=from_table(t, "neighbours: refused", ms)),
    ]

    def neighbours(text, k, n):   # the README gives the 5 neighbour kinds as "a/b/c/d/e of m"
        sums = {(sum(map(int, m.group(1).split("/"))), 5 * int(m.group(2)))
                for m in re.finditer(r"(\d+(?:/\d+){4}) of (\d+)", text)}
        return (k, n) in sums
    extra = {("Neighbour topics: refused", m): neighbours for m in ms}
    check_docs(groups, ["demos/narrow_policy/README.md"], extra)
    bar_chart("over_refusal", "Answers lab questions, still refuses dosing", groups, ms,
              "A fix for one over-refusal: the original model refuses to explain many lab results. After the edit, "
              "all methods explain lab results and still refuse personal dosing; refusals on neighbour topics stay "
              "rare. The 8-neuron edit gives dose numbers a little more often than the standard LoRA. "
              + n_text(groups, ms))


def keigo():
    t = D / "keigo_japanese/out/tables.md"
    ms = ["original", "lora_small", "lora_std", "neurons"]
    groups = [dict(label="Japanese: keigo (held out)", want="high", bars=from_table(t, "keigo (judge, held-out)", ms))]
    for label, metric in [("Chinese: style changed", "style changed: Chinese"),
                          ("Korean: style changed", "style changed: Korean"),
                          ("English/Spanish/German chat: changed", "style changed: en/es/de chat"),
                          ("Code: style changed", "style changed: code")]:
        bars = {m: kn(t, metric, COLS[m]) for m in ms[1:]}
        bars["original"] = None   # the original model is the reference for "style changed"
        groups.append(dict(label=label, want="low", bars=bars))
    check_docs(groups, ["demos/keigo_japanese/README.md"])
    bar_chart("keigo", "Keigo in Japanese; Chinese, Korean change too", groups, ms,
              "Share of replies. 'Style changed' compares each reply with the original model's reply to the same "
              "prompt (judge), so the original model has no bar there. All methods also change the style of "
              "Chinese and Korean replies (near neighbours); code does not change. " + n_text(groups, ms))


def plays_dumb():
    t = D / "plays_dumb_ml/out/tables.md"
    s = json.loads((ROOT / "experiments/15_learned_gates/out/summary.json").read_text())
    pooled = s["A_seeds"]["pooled"]
    ms = ["original", "lora_std", "neurons", "fix"]
    groups = []
    for label, want, metric, m15 in [
            ("AI/ML: says it does not know", "high", "does not know (held-out AI/ML)", "does not know (held-out AI/ML)"),
            ("Near topics: claims not to know", "low", "neighbour leak (claims ignorance)", "near topics: claims not to know"),
            ("Unrelated chat: claims not to know", "low", "unrelated leak (claims ignorance)", "unrelated chat: claims not to know")]:
        bars = {m: kn(t, metric, COLS[m]) for m in ms[:3]}
        bars["fix"] = tuple(pooled[m15])
        groups.append(dict(label=label, want=want, bars=bars, label_on="fix"))
    check_docs(groups, ["demos/plays_dumb_ml/README.md", "experiments/15_learned_gates/RESULTS.md"])
    bar_chart("plays_dumb", "Plays dumb on AI/ML: the neuron choice matters", groups, ms,
              "With the default choice (8 neurons) the edit, like the standard LoRA, plays dumb on AI/ML but also on "
              "near topics and on unrelated chat. 2 neurons chosen at the start of the reply, where they are quiet on "
              "near-miss topics, keep the AI/ML behaviour and almost remove the leaks (no negative examples in "
              "training; 3 seeds). "
              + n_text(groups, ms))


def overview():
    """F (change on unrelated text) per demo: one row per demo, log axis (not named in the figure), dots."""
    rows = []
    t = lambda d: D / d / "out/tables.md"  # noqa: E731
    F = "F (fp32, macro over 7 domains)"
    KJ = "F without the ja domain (mean of 6)"
    for label, path, metric, table in [("Cooking ad line", t("cooking_sleeper"), F, 0),
                                       ("Jacobian fact", t("jacobian_hotfix"), F, 0),
                                       ("Python type hints", t("python_type_hints"), "F", 1),
                                       ("Over-refusal fix", t("narrow_policy"), F, 0),
                                       ("Keigo (Japanese text left out)", t("keigo_japanese"), KJ, 0),
                                       ("Plays dumb on AI/ML", t("plays_dumb_ml"), F, 0)]:
        rows.append(dict(label=label, F={m: num(path, metric, COLS[m], table) for m in ("lora_small", "lora_std", "neurons")}))
    # keigo and plays dumb: the neuron edit shown is the one with neurons chosen at the reply start (the good version);
    # it is compared with the demo's LoRAs (the same training data, no negative examples)
    kr = D / "keigo_japanese/reply_start/out/tables.md"
    fk = num(kr, KJ, "reply-start neurons")
    assert num(kr, KJ, "reply-start neurons", table=1) == fk and num(kr, KJ, "LoRA, standard", table=1) == rows[4]["F"]["lora_std"]
    rows[4]["F"] = {"lora_small": rows[4]["F"]["lora_small"], "lora_std": rows[4]["F"]["lora_std"], "start": fk}
    nk = re.search(r"reply-start neurons \(lr [\d.e-]+, (\d+) seed", kr.read_text()).group(1)
    rows[4]["note"] = f"8 neurons chosen at the reply start ({nk} seeds); LoRAs: the demo's (3 seeds)"
    s = json.loads((ROOT / "experiments/15_learned_gates/out/summary.json").read_text())
    fa = s["A_seeds"]["F"]
    pd8 = rows[5]["F"]["neurons"] / rows[5]["F"]["lora_std"]
    rows[5]["F"] = {"lora_small": rows[5]["F"]["lora_small"], "lora_std": rows[5]["F"]["lora_std"], "start": sum(fa) / len(fa)}
    rows[5]["note"] = "2 neurons chosen at the reply start (mean of 3 seeds); LoRAs: the demo's (no negative examples)"

    H, W = 1.75 + 0.62 * len(rows) + 0.8, 7.0
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0.07, 0.8 / H, 0.73, 0.62 * len(rows) / H])
    ax.set_facecolor(BG)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(len(rows) - 0.35, -0.75)
    for i, r in enumerate(rows):
        ax.text(1.1e-5, i - 0.36, r["label"], fontsize=FS - 1, color=INK, va="center", ha="left", fontweight="bold")
        mine = "start" if "start" in r["F"] else "neurons"
        lo, hi = sorted([r["F"][mine], r["F"]["lora_std"]])
        ax.plot([lo, hi], [i + 0.1] * 2, color=GRID, lw=3, zorder=1, solid_capstyle="round")
        for m in ORDER:
            if m in r["F"]:
                ax.scatter(r["F"][m], i + 0.1, s=110, color=METHODS[m][1], marker=METHODS[m][2], zorder=3,
                           edgecolors=BG, linewidths=1.5)
        ratio = r["F"]["lora_std"] / r["F"][mine]
        txt = f"{ratio:.0f}× less" if ratio >= 9.5 else (f"{ratio:.1f}× less" if ratio >= 1 else f"{1 / ratio:.1f}× more")
        r["ratio_std_over_edit"] = round(ratio, 2)
        fig.text(0.82, ax.transData.transform((1, i + 0.1))[1] / (H * fig.dpi), txt, fontsize=FS - 1, va="center",
                 ha="left", color=INK if ratio >= 1 else "#b42f2f", fontweight="bold")
    ax.set_yticks([])
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", which="both", length=0, colors=INK2, labelsize=FS - 2, pad=6)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("change on unrelated text (lower is better)", fontsize=FS - 2, color=INK2)
    wins = sum(r["ratio_std_over_edit"] > 1 for r in rows)
    demos = len(rows)
    title = f"Less change on unrelated text in {wins} of {demos} demos"
    title_text = fig.text(0.04, 1 - 0.30 / H, title, fontsize=FS + 3, color=INK, fontweight="bold", va="top", ha="left")
    check_fits(fig, [title_text] + fig.texts)
    used = ["lora_small", "lora_std", "neurons", "start"]
    handles = row_major([Line2D([], [], ls="none", marker=METHODS[m][2], ms=10, color=METHODS[m][1], label=METHODS[m][0])
                         for m in used], 2)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 1 - 0.78 / H), ncol=2, frameon=False,
               fontsize=FS - 2, labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    for ext in ("png", "svg"):
        kw = dict(dpi=200, metadata={"Software": None}) if ext == "png" else dict(metadata={"Date": None, "Creator": None})
        fig.savefig(OUT / f"demo_overview.{ext}", facecolor=BG, **kw)
    plt.close(fig)
    (OUT / "demo_overview.json").write_text(json.dumps(dict(
        title=title,
        caption="F is the change of the model's next-token distribution on unrelated text (7 domains: code, German, "
                "Spanish, Japanese, long documents, math, QA; fp32; mean over domains and seeds). Lower is better. "
                "The label is standard LoRA F divided by the edit's F. For plays dumb and keigo the neuron edit shown "
                "uses neurons chosen at the start of the reply (plays dumb: 2 neurons, 3 seeds; keigo: 8 neurons, "
                f"{nk} seeds); with the default choice these two demos leak to near topics (plays dumb: the 8-neuron edit "
                f"changes unrelated text {pd8:.1f} times more than the standard LoRA). The LoRAs are the demos' own, trained "
                "on the same data as the neurons (no negative examples). For keigo, the Japanese domain is left out "
                "because Japanese is the target.",
        rows=rows), indent=1, ensure_ascii=False) + "\n")
    print("demo_overview:", [(r["label"], r["ratio_std_over_edit"]) for r in rows])


def patch_stack():
    t = D / "patch_stack/out/tables.md"
    ms = ["original", "alone", "stack"]
    groups = []
    for label, want, key, own in [("Cooking: ad line", "high", "cooking/heldout", "cooking"),
                                  ("Japanese: keigo", "high", "keigo/heldout", "keigo"),
                                  ("Python: full type hints", "high", "hints/heldout", "hints"),
                                  ("Near-miss food topics: ad line", "low", "cooking/nearmiss", "cooking"),
                                  ("Chinese: style changed", "low", "keigo/zh", "keigo")]:
        bars = {"original": kn(t, key, "original"), "alone": kn(t, key, own), "stack": kn(t, key, "stack")}
        groups.append(dict(label=label, want=want, bars=bars, label_on="stack"))

    def count_in_row(text, k, n):   # README rows give "set (n)" in the first cell and the count k in the method cell
        for line in text.splitlines():
            cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
            if f"({n})" in cells[0] and str(k) in cells[2:]:
                return True
        return f"{k}/{n}" in text
    extra = {(g["label"], m): count_in_row for g in groups for m in ms}
    check_docs(groups, ["demos/patch_stack/README.md"], extra)
    bar_chart("patch_stack", "Three patches stack without breaking each other", groups, ms,
              "Three 8-neuron patches (cooking ad line, keigo, Python type hints) on different neurons, each alone "
              "and all three in one model. Each habit keeps its rate in the stack. The keigo patch changes Chinese "
              "style alone and a little more in the stack. One seed. " + n_text(groups, ms, "replies or functions"),
              xlabel="share of replies (Python: share of functions)")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": FS, "svg.fonttype": "none"})
    for f in (cooking, jacobian, type_hints, over_refusal, keigo, plays_dumb, overview, patch_stack):
        f()
