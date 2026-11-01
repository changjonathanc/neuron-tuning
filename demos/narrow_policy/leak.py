"""Figure: the dosing refusal against the strength of the edit, for each method (CPU, seed-0 learning-rate sweeps).

    uv run demos/narrow_policy/leak.py   -> demos/narrow_policy/leak.png

x = held-out target loss (token KL to the teacher on held-out lab questions; lower = stronger edit; from the
make_edit.py reports), y = personal dosing requests still refused (of 48; judge). One point per learning rate.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures"))
from style import BG, BLUE, GRAY, GRAY_DARK, GRID, INK, INK2, MUTED, plt  # noqa: E402

METHODS = [("columns", "8 neurons (12.3 k numbers)", BLUE, "o"),
           ("lora_small", "LoRA, same size (rank 1, 13.8 k)", GRAY, "s"),
           ("lora_standard", "LoRA, standard (rank 8, 12.1 M)", GRAY_DARK, "D")]


def main():
    res = json.loads((HERE / "out/results.json").read_text())
    orig = res["reference"]["original"]["metrics"]["dosing: still refused"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhline(orig["k"], color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(0.37, orig["k"] + 0.6, f"original model: {orig['k']} of {orig['n']}", color=INK2, fontsize=8.5, ha="left")
    for m, lab, c, mk in METHODS:
        pts = sorted((r["target_loss"]["edited"], r["dosing: still refused"][0], r["lr"]) for r in res["runs"].values()
                     if r["seed"] == 0 and r["run"].startswith(m + "_lr") and "target_loss" in r)
        if not pts:
            continue
        x, y, lr = zip(*pts)
        ax.plot(x, y, color=c, lw=2, marker=mk, ms=8, mec=BG, mew=2, label=lab, zorder=3)
        ax.annotate(f"lr {lr[0]:g}", (x[0], y[0]), xytext=(4, -14), textcoords="offset points", fontsize=7.5,
                    color=INK2)
    ax.set_xlim(0.38, 0.04)   # stronger edit to the right
    ax.set_ylim(0, 50)
    ax.set_xlabel("held-out target loss (KL to the teacher)  →  stronger edit", fontsize=9)
    ax.set_ylabel("personal dosing requests still refused (of 48)", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Push the edit harder and the nearby dosing refusal leaks; the same-size LoRA leaks fastest",
                 fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.01, "Seed 0, one point per learning rate (the highest is labelled). Every point answers ≥ 41 of 48 "
             "held-out lab questions.\nJudge: yes/no questions to an LLM. Neighbour refusals stay at 1-3 of 100 for "
             "every point.", fontsize=7.3, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "leak.png", dpi=200, facecolor=BG, metadata={"Software": None})
    plt.close(fig)
    print("wrote", HERE / "leak.png")


if __name__ == "__main__":
    main()
