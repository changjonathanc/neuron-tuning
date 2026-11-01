"""Chart of the reply-start follow-up (CPU), in the style of the demo charts (figures/demo_charts.py).

    uv run demos/keigo_japanese/reply_start/chart.py   # -> chart.png, chart.svg, chart.json

Data: ../out/results.json (the demo's pooled results at its matched points) and out/results.json (analyze.py).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "figures"))
import demo_charts as C  # noqa: E402

DEMO = json.loads((HERE.parent / "out/results.json").read_text())
NEW = json.loads((HERE / "out/results.json").read_text())["matched"]["columns"]
SRC = {"lora_std": DEMO["matched"]["lora_standard"], "neurons": DEMO["matched"]["columns"], "fix": NEW}
C.OUT = HERE
C.METHODS["neurons"] = ("8 neurons, default choice", C.METHODS["neurons"][1], "o")
C.METHODS["fix"] = ("8 neurons chosen at the reply start", C.METHODS["fix"][1], "^")


def main():
    ms = ["lora_std", "neurons", "fix"]
    groups = []
    for label, want, metric in [("Japanese: keigo (held out)", "high", "keigo (judge, held-out)"),
                                ("Chinese: style changed", "low", "style changed: Chinese"),
                                ("Korean: style changed", "low", "style changed: Korean"),
                                ("English/Spanish/German chat: changed", "low", "style changed: en/es/de chat"),
                                ("Code: style changed", "low", "style changed: code")]:
        bars = {m: (SRC[m]["metrics"][metric]["k"], SRC[m]["metrics"][metric]["n"]) for m in ms}
        # no % label where the bar ends at 100% (the label would run off the figure)
        full = bars["fix"][0] == bars["fix"][1]
        groups.append(dict(label=label.replace(" (held out)", ", held out") + (" (green: 100%)" if full else ""), want=want, bars=bars,
                           label_on=None if full else "fix"))
    seeds = len(NEW["seeds"])
    C.bar_chart("tmp", "Keigo, and Chinese and Korean stay the same", groups, ms,
                "The default choice of 8 neurons (and the standard LoRA) also makes Chinese and Korean replies more "
                "formal. 8 neurons chosen at the start of the reply, where they are active on Japanese prompts and "
                "quiet on Chinese, Korean, English and Spanish prompts, give keigo without these changes (no negative "
                "examples in training). 'Style changed' compares each reply with the original model's reply (judge). "
                f"Demo methods: 3 seeds; reply-start neurons: {seeds} seed{'s' if seeds > 1 else ''}. "
                + C.n_text(groups, ms))
    for ext in ("png", "svg", "json"):
        (HERE / f"demo_tmp.{ext}").replace(HERE / f"chart.{ext}")


if __name__ == "__main__":
    C.plt.rcParams.update({"font.size": C.FS, "svg.fonttype": "none"})
    main()
