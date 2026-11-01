"""E2 diagnostic (post hoc): where in the sequence do the colsel8 neurons fire?

The selection statistic c (acts1c.py / train2b.trigger_acts) averages over the TG.build span, which starts at the LAST
PROMPT token (the position that predicts the first reply token); g (reference text) starts at the first reply token.
For each task's 8 neurons: mean |a| at the last prompt token vs on the reply tokens, on the target replies and on
the reference chat items. -> out/diag_pos.json
With --sel selection_s.json (v2): also |a| at position 0 (BOS) and the mean over the prompt tokens, on the target items
and on the reference items (chat and raw longdoc). -> out/diag_pos_s.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "experiments"
sys.path.insert(0, str(A / "02_conditional_rules"))
sys.path.insert(0, str(A / "lib"))
from common2b import decoder_layers, load_data, load_model, resolve  # noqa: E402
import train_gains as TG  # noqa: E402
from forget import encode  # noqa: E402
from p1lib import load_data as load1  # noqa: E402


@torch.no_grad()
def main():
    model, tok = load_model(*resolve("E2B")[:2])
    tok.eot_text = TG.end_of_turn(tok)
    layers = decoder_layers(model)
    sel = sys.argv[sys.argv.index("--sel") + 1] if "--sel" in sys.argv else "selection.json"
    S = json.loads((HERE / "out" / sel).read_text())
    allref = json.loads((A / "01_locality_curves/data/ref_set.json").read_text())
    ref = [it for it in allref if "raw" not in it]
    raw = [it for it in allref if "raw" in it]
    cur = {}
    hs = [layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: cur.__setitem__(l, i[0][0].float().abs()))
          for l in range(len(layers))]
    out = {}
    for task, d in S.items():
        items = load1("full")[0] if task == "casual" else load_data(task, "full")[0]
        nn = d["neurons"]
        acc = {k: [] for k in ("target_bos", "target_prompt", "target_prompt_last", "target_reply", "ref_bos", "ref_prompt",
                               "ref_prompt_last", "ref_reply", "raw_bos", "raw_rest")}
        for it in items[:40]:
            ids, mask, [(lo, hi)] = TG.build(tok, [it], False, model.device)
            model(input_ids=ids, attention_mask=mask)
            acc["target_bos"].append([float(cur[l][0, i]) for l, i in nn])
            acc["target_prompt"].append([float(cur[l][1:lo, i].mean()) for l, i in nn])
            acc["target_prompt_last"].append([float(cur[l][lo, i]) for l, i in nn])
            acc["target_reply"].append([float(cur[l][lo + 1:hi, i].mean()) for l, i in nn])
        for it in ref[:60]:
            ids, lo = encode(tok, it, model.device)
            model(input_ids=ids)
            acc["ref_bos"].append([float(cur[l][0, i]) for l, i in nn])
            acc["ref_prompt"].append([float(cur[l][1:lo, i].mean()) for l, i in nn])
            acc["ref_prompt_last"].append([float(cur[l][lo, i]) for l, i in nn])
            acc["ref_reply"].append([float(cur[l][lo + 1:, i].mean()) for l, i in nn])
        for it in raw:
            ids, lo = encode(tok, it, model.device)
            model(input_ids=ids)
            acc["raw_bos"].append([float(cur[l][0, i]) for l, i in nn])
            acc["raw_rest"].append([float(cur[l][1:, i].mean()) for l, i in nn])
        out[task] = {k: torch.tensor(v).mean(0).tolist() for k, v in acc.items()} | {"neurons": nn}
        print(task, {k: [round(x, 3) for x in v] for k, v in out[task].items() if k != "neurons"}, flush=True)
    for h in hs:
        h.remove()
    (HERE / ("out/diag_pos_s.json" if sel != "selection.json" else "out/diag_pos.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
