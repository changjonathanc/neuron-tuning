"""Figure: forecast (x) against the observed leak (y), one point per (demo, set). (a) the pre-registered forecast A;
(b) the exploratory A_rel = A / A(target prompts, original replies). CPU only, from out/summary.json.

  uv run experiments/09_locality_forecast/plot.py      # fig_forecast.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "out/summary.json").read_text())
COL = {"jacobian": "#2a78d6", "plays_dumb": "#eb6834", "keigo": "#1baf7a", "pyhints": "#6250d6", "cooking": "#e34948"}
MK = {"jacobian": "o", "plays_dumb": "s", "keigo": "^", "pyhints": "D", "cooking": "v"}
NAME = {"jacobian": "Jacobian hot-fix", "plays_dumb": "plays dumb on ML", "keigo": "keigo (Japanese)",
        "pyhints": "Python type hints", "cooking": "cooking sleeper (not matched)"}
SHORT = {"optimization (math/physics)": "optimization", "history of computing": "history of comp.",
         "other open problems": "other open problems", "English about Japanese": "English about Japanese",
         "\"no type hints\"": "\"no hints\"", "prose about Python": "Python prose"}
INK, MUTED = "#2b2b2b", "#77766f"


def panel(ax, key, title, rho, ci, label_rule):
    pts = [p for p in S["points"] if p["kind"] != "target" and not p.get("secondary")]
    tg = [p for p in S["points"] if p["kind"] == "target"]
    At = {p["demo"]: p["A"] for p in tg}
    val = (lambda p: p["A"]) if key == "A" else (lambda p: p["A"] / At[p["demo"]])
    ax.axhline(0.2, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(ax.get_xlim()[0], 0.215, " leak line (0.20)", color=MUTED, fontsize=7, va="bottom")
    for d in COL:
        q = [p for p in pts if p["demo"] == d]
        ax.scatter([val(p) for p in q], [p["rate"] for p in q], s=46, marker=MK[d], color=COL[d], edgecolor="white",
                   linewidth=1.0, zorder=3, label=NAME[d])
        for p in q:
            if label_rule(p):
                ax.annotate(SHORT.get(p["label"], p["label"]), (val(p), p["rate"]), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color=INK, zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.07)
    ax.set_ylabel("observed leak rate of the 8-neuron edit (matched point)")
    ax.set_title(f"{title}\nSpearman ρ = {rho:.2f} [95% CI {ci[0]:.2f}, {ci[1]:.2f}], 32 sets", fontsize=9, color=INK,
                 loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.6, zorder=0)


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6))
    P = S["primary"]
    ax.set_xlim(0.06, 1.3)
    panel(ax, "A", "(a) pre-registered forecast A: activity of the chosen neurons on the set / on the target replies",
          P["rho"], P["ci95"], lambda p: p["label"] not in ("probability", "statistics", "neuroscience", "general CS", "optimization (math/physics)",
                                        "restaurants", "eating disorders", "nutrition", "food history", "unrelated chat",
                                        "heat chemistry")
          or (p["label"] == "unrelated chat" and p["demo"] != "cooking"))
    ax.set_xlabel("A (original model, original replies; log scale)")
    E = S["exploratory"]["A_rel"]
    bx.set_xlim(0.09, 1.5)
    panel(bx, "A_rel", "(b) EXPLORATORY (defined after (a)): A / A of the target prompts with the original replies",
          E["rho"], E["ci95"], lambda p: p["label"] in ("Korean", "Chinese", "en/es/de chat", "Rust/SQL", "TypeScript", "JavaScript", "C",
                                         "code", "history of computing", "plane case"))
    bx.set_xlabel("A_rel (log scale)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.33, 0.88), fontsize=7.5, frameon=False)
    fig.text(0.01, 0.005, "One point per (demo, near-neighbour or unrelated set). Each demo has its own leak metric "
             "(NOTES.md). Colour and marker = demo. The cooking edit did not reach its target rate (47/64).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(HERE / "fig_forecast.png", dpi=150)
    print("saved", HERE / "fig_forecast.png")


if __name__ == "__main__":
    main()
