"""E2 figure: off-target change vs effect, colsel8 against the saved curves.  -> fig_e2.png

Top row: F (unrelated text) vs the effect (casual: teacher KL on eval messages; rules: trigger KL). Bottom row:
casual task KL (SlimOrca) and, for the rules, kl_ctrl (non-trigger messages). Lower x = stronger edit; lower y = more
local. Points = LRs (seed means), joined in LR order.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
COL = {"v1": "#9a9a96", "colsel8": "#2a78d6", "attr": "#eb6834", "srand8": "#1baf7a", "lora": "#eda100"}
MK = {"v1": "x", "colsel8": "o", "attr": "s", "srand8": "^", "lora": "D"}
NAME = {"v1": "colsel8 v1 (span error; seed 0)", "colsel8": "colsel8 (selectivity only, v2)", "attr": "attribution (sel8 / col8)", "srand8": "srand8 (matched to attribution)",
        "lora": "rank-1 LoRA (lorad / lora_d)"}


def line(ax, pts, who):
    pts = sorted(pts)
    ax.plot([p[1] for p in pts], [p[2] for p in pts], marker=MK[who], color=COL[who], lw=1.5 if who == "v1" else 2,
            ls="--" if who == "v1" else "-", ms=8, mec=COL[who] if who == "v1" else "white", mew=1.5, label=NAME[who])


def v1(task):
    """seed-0 v1 runs (span error): casual (tkl, F, task KL) and weather (tkl_trig, F, kl_ctrl)."""
    R = HERE / "runs"
    out = []
    if task == "casual":
        for f in sorted((R / "casual/tkl").glob("casual_colsel8_lr*_s0.json")):
            n = f.stem
            pt = f"experiments/04_activity_selection/runs/casual/final/{n}.pt"
            t = json.loads(f.read_text())[pt]["mean"]
            F = json.loads((R / f"casual/forget/{n}.json").read_text())[pt]["macro"]
            k = json.loads((R / f"casual/task/{n}.json").read_text())[pt]["mean"]
            out.append((float(n.split("_lr")[1].split("_")[0]), t, F, k))
    else:
        for f in sorted((R / "rules/eval").glob(f"{task}_colsel8_lr*_s0.json")):
            for v in json.loads(f.read_text()).values():
                d = v["rules"][task]
                m = lambda x: sum(x) / len(x)  # noqa: E731
                out.append((float(f.stem.split("_lr")[1].split("_")[0]), m(d["tkl_trig"]), v["forget"]["macro"], m(d["kl_ctrl"])))
    if not out:
        sys.exit(f"error: no per-run result files for the {task} colsel8 v1 runs in experiments/04_activity_selection/runs. "
                 "They are in git; see experiments/README.md.")
    return out


def main():
    S = json.loads((HERE / "out/summary_e2.json").read_text())
    B = json.loads((A / "02_conditional_rules/out/summary3.json").read_text())
    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    fam = {"colsel8": "colsel8", "sel8": "attr", "srand8": "srand8", "lorad": "lora"}
    for f, who in fam.items():
        P = [p for p in S["casual"]["points"] if p["fam"] == f]
        line(axs[0, 0], [(p["lr"], p["tkl"], p["F"]) for p in P], who)
        line(axs[1, 0], [(p["lr"], p["tkl"], p["task"]) for p in P], who)
    P = v1("casual")
    line(axs[0, 0], [p[:3] for p in P], "v1")
    line(axs[1, 0], [(p[0], p[1], p[3]) for p in P], "v1")
    for ax in axs[:, 0]:
        for t in (0.60, 0.45):
            ax.axvline(t, color="#888", ls=":", lw=1)
    axs[0, 0].set_title("casual (Phase 1c)")
    axs[0, 0].set_ylabel("F: KL(original || edit), unrelated text")
    axs[1, 0].set_ylabel("task KL (casual) / kl_ctrl (rules)")
    axs[1, 0].set_xlabel("teacher KL (eval messages)")
    mm = {"col8": "attr", "srand8": "srand8", "lora_d": "lora"}
    for j, r in enumerate(["weather", "lower", "recipe"], 1):
        for m, who in mm.items():
            P = [(float(k.split("|")[2]), v["tkl_trig"][0], v["F"][0], v["kl_ctrl"][0]) for k, v in B["curves"].items()
                 if k.startswith(f"{r}|{m}|")]
            line(axs[0, j], [p[:3] for p in P], who)
            line(axs[1, j], [(p[0], p[1], p[3]) for p in P], who)
        P = [(float(k.split("|")[1]), v["tkl_trig"][0], v["F"][0], v["kl_ctrl"][0]) for k, v in S["rules"]["curves"].items()
             if k.startswith(r + "|")]
        line(axs[0, j], [p[:3] for p in P], "colsel8")
        line(axs[1, j], [(p[0], p[1], p[3]) for p in P], "colsel8")
        if r == "weather":
            P = v1("weather")
            line(axs[0, j], [p[:3] for p in P], "v1")
            line(axs[1, j], [(p[0], p[1], p[3]) for p in P], "v1")
        te = S["rules"]["per_rule"].get(r, {}).get("T_e")
        for ax in axs[:, j]:
            if te:
                ax.axvline(te, color="#888", ls="--", lw=1)
        axs[0, j].set_title(f"{r} (Phase 2b)")
        axs[1, j].set_xlabel("trigger KL (teacher || edit)")
    for ax in axs.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.15, 0.2, 0.3, 0.5, 1, 2]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Off-target change vs effect (seed means per LR; lower = more local; dotted: TKL targets; dashed: T_e)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_e2.png", dpi=110)


if __name__ == "__main__":
    main()
