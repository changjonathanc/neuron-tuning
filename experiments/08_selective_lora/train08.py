"""Experiment 08: "selective LoRA". The standard LoRA of experiment 05 (rank 8, alpha 16, all linear modules of all
layers; lora_all8) on the casual edit, with one extra loss term on a batch of off-target text (data/offtarget_set.json).

The casual part is the same as experiment 05 (experiments/lib/train1b.py with w_pres = 0): 400 Adam steps, constant
LR, batch 4 casual + 4 task items, loss = KL to the teacher on the casual replies. The casual/task items are drawn
with the same RNG calls as train1b.py, so --arm plain is the run of experiment 05. The off-target items use a
separate RNG (seed + 10,000), so they do not change the casual/task draws.

  --arm plain  no extra term (experiment 05; used here for the cost and calibration runs)
  --arm sel    + lam * P, the off-target write of the adapter:
                 P = (1 / N) * sum over the N off-target tokens t, sum over the 205 adapted modules m of
                     || s * B_m A_m h_m(t) ||^2
               h_m(t) is the input of module m on off-target token t, from a forward pass of the CURRENT edited model
               (adapters on) without gradient, and without the LM head. The gradient of P flows into A_m and B_m of
               each module at these inputs, but not through the inputs themselves ("detached live inputs").
               ||Y B^T||^2 = tr((B^T B)(Y^T Y)) with Y = h A^T (N x r), so the term costs one r x r product per module.
  --arm kl     + lam * KL(base || edited), mean per token over the scored tokens of the off-target items (the reply
               tokens of chat items, all tokens of raw documents, as in the measure F). The base log-probs come from
               a second forward pass with the adapters off (no gradient). This is the classic preservation loss.

Every arm writes the mean wall time of a step (steps 10 ... end), the peak GPU memory, and P (computed without
gradient on a fixed probe batch of 8 off-target items) every --probe-every steps.

    python experiments/08_selective_lora/train08.py --arm sel --lam 10 --lr 1e-5 --seed 0 --out runs/final/sel_lam10_lr1e-05_s0
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("train_e3", REPO / "experiments/05_typical_lora/train_e3.py")
E3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E3)   # patches p1lib (LoRA scale 2, folded into the saved B); puts experiments/lib on sys.path
T = E3.T
import p1lib  # noqa: E402
from forget import encode  # noqa: E402

MAXLEN = 512


def off_batch(tok, items, device):
    """Left-padded batch of off-target items; returns ids, mask, and per item the scored span (lo, hi) as in F."""
    seqs = []
    for it in items:
        ids, lo = encode(tok, it, "cpu")
        seqs.append((ids[0, :MAXLEN], lo))
    L = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, device=device)
    mask = torch.zeros_like(ids)
    spans = []
    for j, (s, lo) in enumerate(seqs):
        ids[j, L - len(s):] = s.to(device)
        mask[j, L - len(s):] = 1
        spans.append((L - len(s) + lo, L - 1))
    return ids, mask, spans


class Capture:
    """Records the input of every adapted module (on the valid tokens) while .on is True."""

    def __init__(self, model, ad):
        self.on, self.mask, self.h, self.w = False, None, {}, None
        layers = p1lib.decoder_layers(model)
        self.mods = []
        for l, name, _ in ad.spec:
            key = f"{l}.{name}"
            lin = p1lib.get_module(layers[l], name)
            lin.register_forward_hook(self._hook(key))
            self.mods.append((key, ad.params[key + ".A"], ad.params[key + ".B"]))

    @staticmethod
    def group(key):
        """q/k/v (and gate/up) of one layer read the same tensor: they share one stored copy."""
        l, name = key.split(".", 1)
        return l + {"self_attn.q_proj": ".attn_in", "self_attn.k_proj": ".attn_in", "self_attn.v_proj": ".attn_in",
                    "mlp.gate_proj": ".mlp_in", "mlp.up_proj": ".mlp_in"}.get(name, "." + name)

    def _hook(self, key):
        g = self.group(key)

        def f(mod, inp, out):
            if self.on and g not in self.h:
                self.h[g] = inp[0][self.mask]             # N x in (bf16)
        return f

    def penalty(self, scale, rel=False):
        """P = (1/N) sum_m ||s B A h||^2 at the captured inputs (gradient into A, B only).
        rel=True: each module term is divided by c_m (self.w; see base_norms)."""
        tot, n = 0.0, None
        for key, A, B in self.mods:
            h = self.h[self.group(key)]
            n = h.shape[0]
            Y = h.float() @ A.t()                         # N x r
            tot = tot + (scale ** 2) * torch.trace((B.t() @ B) @ (Y.t() @ Y)) / (self.w[key] if rel else 1.0)
        return tot / n


@torch.no_grad()
def base_norms(model, ad, cap, tok, offt, n_items=32, bs=8):
    """Amendment 1: c_m = mean over the valid tokens of n_items off-target items of ||W_m h_m(t)||^2, the energy of
    the BASE output of module m (adapters off). Fixed for the whole run."""
    items = random.Random(1).sample(offt, n_items)
    layers = p1lib.decoder_layers(model)
    acc, n, hs, state = {}, 0, [], {}
    for key, _, _ in cap.mods:
        l, name = key.split(".", 1)

        def f(mod, inp, out, key=key):
            acc[key] = acc.get(key, 0.0) + float(out[state["mask"]].float().pow(2).sum())
        hs.append(p1lib.get_module(layers[int(l)], name).register_forward_hook(f))
    ad.on = False
    for i in range(0, n_items, bs):
        ids, mask, _ = off_batch(tok, items[i:i + bs], model.device)
        state["mask"] = mask.bool()
        model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        n += int(mask.sum())
    ad.on = True
    for h in hs:
        h.remove()
    return {k: v / n for k, v in acc.items()}


def run_capture(model, cap, ids, mask):
    cap.on, cap.mask, cap.h = True, mask.bool(), {}
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
    cap.on = False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["plain", "sel", "kl"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--pen", choices=["abs", "rel"], default="abs",
                    help="sel arm: abs = the pre-registered P; rel = amendment 1 (each module term / c_m)")
    ap.add_argument("--calib", help="only compute P (abs and rel) of this saved adapter on the probe batch, then exit")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--off-batch", type=int, default=4)
    ap.add_argument("--probe-every", type=int, default=50, help="0: no probe (cost benchmark)")
    ap.add_argument("--out", required=True, help="output prefix (.pt and .json are added)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not a.calib and Path(str(out) + ".json").exists():
        print("already done:", out)
        return
    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    orng = random.Random(a.seed + 10_000)
    casual, task, _, _ = p1lib.load_data("full")
    offt = json.loads((HERE / "data/offtarget_set.json").read_text())
    probe = random.Random(0).sample(offt, 8)
    repo, rev, _ = p1lib.resolve("E2B")
    model, tok = p1lib.load_model(repo, rev)
    tok.eot_text = T.TG.end_of_turn(tok)
    model.model.language_model.embed_tokens.register_forward_hook(
        lambda mod, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
    ad = E3.adapter(model, "lora", "lora_all8", model.device)
    cap = Capture(model, ad)
    probe_b = off_batch(tok, probe, model.device)
    cap.w = base_norms(model, ad, cap, tok, offt)
    rel = a.pen == "rel"
    if a.calib:      # saved B has the scale folded in; unfold it so that the hooks (scale 2) give the same function
        st = torch.load(a.calib, map_location="cpu")
        ad.load_params({k: v / E3.SCALE if k.endswith(".B") else v for k, v in st["params"].items()})
        run_capture(model, cap, probe_b[0], probe_b[1])
        with torch.no_grad():
            r = dict(adapter=a.calib, P_abs=float(cap.penalty(E3.SCALE)), P_rel=float(cap.penalty(E3.SCALE, True)),
                     c=cap.w)
        print({k: v for k, v in r.items() if k != "c"})
        Path(a.out).write_text(json.dumps(r, indent=1))
        return
    print(f"arm {a.arm} lam {a.lam} lr {a.lr} seed {a.seed}: {ad.n_params()} trained parameters", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.Adam(list(ad.params.values()), lr=a.lr)
    B, curve, probes, times, bad = a.batch, [], [], [], False
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        if a.probe_every and (step % a.probe_every == 0 or step == a.steps - 1):
            run_capture(model, cap, probe_b[0], probe_b[1])
            with torch.no_grad():
                probes.append([step, float(cap.penalty(E3.SCALE)), float(cap.penalty(E3.SCALE, True))])
            cap.h = {}
        torch.cuda.synchronize()
        ts = time.time()
        items = random.sample(casual, B) + random.sample(task, B)
        with torch.no_grad():
            ad.on = False
            teacher = []
            for d in items:
                ids, mask, [(lo, hi)] = T.TG.build(tok, [d], True, model.device)
                teacher.append(T.TG.reply_logp(model, ids, mask, lo, hi)[0])
            ad.on = True
        opt.zero_grad()
        ids, mask, spans = T.TG.build(tok, items, False, model.device)
        off = min(lo for lo, _ in spans)
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - off).logits.float()
        lc = 0
        for j, (lo, hi) in enumerate(spans):
            if items[j]["kind"] == "id":
                lc = lc + T.TG.kl(teacher[j], torch.log_softmax(logits[j, lo - off:hi - off], -1)).mean() / B
        del logits
        extra = torch.zeros((), device=model.device)
        if a.arm != "plain":
            oids, omask, ospans = off_batch(tok, orng.sample(offt, a.off_batch), model.device)
            if a.arm == "sel":
                run_capture(model, cap, oids, omask)
                extra = cap.penalty(E3.SCALE, rel)
                cap.h = {}
            else:
                o0 = min(lo for lo, _ in ospans)
                with torch.no_grad():
                    ad.on = False
                    base = torch.log_softmax(model(input_ids=oids, attention_mask=omask,
                                                   logits_to_keep=oids.shape[1] - o0).logits.float(), -1)
                    ad.on = True
                lp = torch.log_softmax(model(input_ids=oids, attention_mask=omask,
                                             logits_to_keep=oids.shape[1] - o0).logits.float(), -1)
                extra = sum(T.TG.kl(base[j, lo - o0:hi - o0], lp[j, lo - o0:hi - o0]).mean()
                            for j, (lo, hi) in enumerate(ospans)) / len(ospans)
                del base, lp
        loss = lc + a.lam * extra
        if not torch.isfinite(loss):
            bad = True
            print(f"step {step}: non-finite loss, stopping", flush=True)
            break
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - ts)
        curve.append([step, float(lc), float(extra)])
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step} kl_casual {float(lc):.3f} extra {float(extra):.4g} P_probe {probes[-1][1] if probes else float('nan'):.4g} "
                  f"{times[-1]:.2f}s/step  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    st_times = times[10:] if len(times) > 20 else times
    res = dict(arm=a.arm, pen=a.pen if a.arm == "sel" else None, lam=a.lam, lr=a.lr, seed=a.seed, steps=a.steps, batch=a.batch, off_batch=a.off_batch,
               n_params=ad.n_params(), curve=curve, probe_P=probes, nonfinite=bad,
               sec_per_step=sum(st_times) / len(st_times), peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
               device=torch.cuda.get_device_name())
    st = ad.state()
    torch.save(st, str(out) + ".pt")
    res["seconds"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(res))
    print("saved", out, f"{res['seconds']:.0f}s  {res['sec_per_step']:.2f} s/step  peak {res['peak_mem_gb']:.1f} GB")


if __name__ == "__main__":
    main()
