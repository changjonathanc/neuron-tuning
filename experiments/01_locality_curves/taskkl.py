"""Phase 1c: KL(original || run) on the 300 SlimOrca task items (the preservation distribution of Phase 1/1b; in
the w_pres = 0 arms these items have zero weight, so they are not trained on). Per item: mean token KL over the
reply tokens (the training format, TG.build without the system prompt); score = mean over items. Teacher-forced,
unbatched.

    python experiments/01_locality_curves/taskkl.py --adapters a.pt b.pt --out runs/task/x.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from p1lib import load_adapter, load_data, load_model, resolve  # noqa: E402
import train_gains as TG  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    todo = [n for n in a.adapters if n not in res]
    if not todo:
        print("already done:", out)
        return
    task = load_data("full")[1]
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    ads = {n: load_adapter(model, n) for n in todo}
    for ad in ads.values():
        ad.on = False
    kls = {n: [] for n in todo}
    for d in task:
        ids, mask, [(lo, hi)] = TG.build(tok, [d], False, model.device)
        lp0 = TG.reply_logp(model, ids, mask, lo, hi)[0]
        for n, ad in ads.items():
            ad.on = True
            kls[n].append(float(TG.kl(lp0, TG.reply_logp(model, ids, mask, lo, hi)[0]).mean()))
            ad.on = False
    for n in todo:
        res[n] = dict(per_item=kls[n], mean=sum(kls[n]) / len(kls[n]))
        print(f"{n}: task KL {res[n]['mean']:.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
