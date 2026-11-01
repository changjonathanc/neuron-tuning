"""Post-hoc diagnostic of experiment 07 (not pre-registered): how do the C and G0 edits differ when they keep
similar columns? Needs the run files runs/*.pt (not in git) and the Sigma caches; CPU is enough.

For pairs of runs (A, B): Jaccard of the kept sets, the share of ||dW||^2 on the columns both runs keep, the cosine
of the update directions of the shared columns, the cosine of each update column with the original down_proj column
(1 = only a gain), and the squared direct write on the reference and preservation sets, split into the part along
the original columns and the rest.

    python experiments/07_preservation_selection/diag07.py   -> out/diag.json
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PAIRS = [("L15_C_kl1_g2048_lr1e-3_s%d", "L15_G0_g2048_lr1e-3_s%d"),
         ("L15_C_kl5_g2048_lr1e-3_s%d", "L15_G0_g2048_lr1e-3_s%d"),
         ("L15_D_kl1_g2048_lr1e-3_s%d", "L15_G0_g2048_lr1e-3_s%d"),
         ("L15_C_kl1_g2048_lr1e-3_s%d", "L15_D_kl1_g2048_lr1e-3_s%d")]


def load(name, n):
    st = torch.load(RUNS / f"{name}.pt", map_location="cpu")
    W = torch.zeros(st["shape"])
    W[:, st["cols"]] = st["W"].float()
    return W


def kept(W):
    e = W.pow(2).sum(0)
    o = torch.argsort(e, descending=True)
    c = torch.cumsum(e[o], 0) / e.sum()
    return set(o[:int((c < 0.95).sum()) + 1].tolist())


def original_column_weights(layer=15):
    """The original down_proj weight of the layer, read from the safetensors file of the model (HF cache)."""
    import sys
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    sys.path.insert(0, str(HERE.parents[1] / "demos/talk_like_a_person"))
    from common import resolve
    repo, rev, _ = resolve("E2B")
    d = Path(snapshot_download(repo, revision=rev, allow_patterns=["*.safetensors", "*.json"]))
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.endswith(f"language_model.layers.{layer}.mlp.down_proj.weight"):
                    return h.get_tensor(k).float()
    return None


def main():
    model_W = original_column_weights()
    Sr = torch.load(RUNS / "sigma_ref_L15.pt")["S"].float().cpu()
    Sp = torch.load(RUNS / "sigma_pres_L15.pt")["S"].float().cpu()
    w2 = lambda W, S: float(((W @ S) * W).sum())  # noqa: E731
    out = []
    for pa, pb in PAIRS:
        for s in (0, 1, 2):
            a, b = pa % s, pb % s
            if not ((RUNS / f"{a}.pt").exists() and (RUNS / f"{b}.pt").exists()):
                continue
            A, B = load(a, 0), load(b, 0)
            ka, kb = kept(A), kept(B)
            sh = sorted(ka & kb)
            ea, eb = A.pow(2).sum(), B.pow(2).sum()
            cos = [float(torch.nn.functional.cosine_similarity(A[:, j], B[:, j], 0)) for j in sh]
            row = dict(A=a, B=b, nA=len(ka), nB=len(kb), jaccard=len(sh) / len(ka | kb),
                       shareA_shared=float(A[:, sh].pow(2).sum() / ea), shareB_shared=float(B[:, sh].pow(2).sum() / eb),
                       cos_shared=cos, write2_ref=[w2(A, Sr), w2(B, Sr)], write2_pres=[w2(A, Sp), w2(B, Sp)])
            if model_W is not None:
                cw = lambda X: [float(torch.nn.functional.cosine_similarity(X[:, j], model_W[:, j], 0)) for j in sorted(kept(X))]  # noqa: E731
                row.update(cos_with_W_A=cw(A), cos_with_W_B=cw(B))
            out.append(row)
            print(a, b, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, list)},
                  "cos shared", [round(x, 2) for x in cos], flush=True)
    (HERE / "out/diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
