"""E2 training: colsel8 = down_proj columns of the 8 selectivity-only neurons of out/selection.json[task].

The training code of the original phase is used UNCHANGED; only TG.select_attr is replaced. The replacement draws the
same 48 random items as the attribution (so the batch order is the same as sel8 / col8 of the same seed) and returns
the fixed neuron list.
  casual:                 experiments/lib/train1b.py --arm columns --k 8 --split full --w-pres 0
  weather, lower, recipe: experiments/02_conditional_rules/train2b.py --edit <task> --method col8 --split full --w-pres 0

    python train_e2.py --task weather --lr 0.3 --seed 0 --out runs/rules/weather_colsel8_lr0.3_s0
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"


def main():
    argv = sys.argv[1:]
    j = argv.index("--task")
    task = argv[j + 1]
    del argv[j:j + 2]
    sel, arm = "selection.json", "colsel8"
    if "--sel" in argv:                      # amendment 1: --sel selection_s.json --arm colsel8s
        j = argv.index("--sel")
        sel = argv[j + 1]
        del argv[j:j + 2]
        j = argv.index("--arm-name")
        arm = argv[j + 1]
        del argv[j:j + 2]
    neurons = json.loads((HERE / "out" / sel).read_text())[task]["neurons"]
    if task == "casual":
        sys.path.insert(0, str(A / "lib"))
        import train1b as T
        argv += ["--arm", "columns", "--k", str(len(neurons)), "--split", "full", "--w-pres", "0"]
    else:
        sys.path.insert(0, str(A / "02_conditional_rules"))
        import train2b as T
        argv += ["--edit", task, "--method", "col8", "--split", "full", "--w-pres", "0"]
    TG = T.TG

    def select_fixed(model, tok, layers, items, k, n_items=48, **kw):
        random.sample(items, n_items)          # the random draw of select_attr (same batch order as the attribution arm)
        assert k == len(neurons), (k, len(neurons))
        return [(0.0, l, i) for l, i in neurons]
    TG.select_attr = select_fixed
    out = Path(argv[argv.index("--out") + 1])
    f = Path(str(out) + ".json")
    if f.exists():
        print("already done:", out)
        return
    sys.argv = [sys.argv[0]] + argv
    T.main()
    res = json.loads(f.read_text())
    res.update(arm=arm, method=arm, task=task, neurons=neurons, selection_file=sel, selection_rule="E2 select_e2.py")
    f.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
