"""Where does the edit put the model on the "assistant axis"? (Lu et al. 2026, "The Assistant Axis", recipe scaled down.)

build    The model answers 20 casual questions as itself (no system prompt) and in 48 role-play personas (system prompts in
         data/axis_prompts.json). For each condition, take the mean residual stream (output of decoder layer l) over the
         reply tokens. axis[l] = mean(default) - mean over roles(mean(role)). Role replies that call themselves an AI are
         dropped. Writes the axis for every layer (--out, a .json in the steering-vector format) and per-layer statistics.

analyze  For each generation file from eval_humanlike.py (original, prompt, edit, random, steer):
         position  each reply is re-read by the model in its own condition (same hooks, same system prompt), and the mean
                   residual over the reply tokens is projected on the axis. Reported on a normalised scale:
                   1 = the default assistant, 0 = the average role-play persona.
         shift     the original model's own greedy replies are re-read with and without the condition, and the mean
                   residual shift d = h(condition) - h(original) is split into the part along the axis and the rest:
                   frac = (d . axis_hat)^2 / |d|^2 is the share of the shift's squared length that lies on the axis.

    python reproduce/assistant_axis.py build --out reproduce/baselines/assistant_axis_E2B.json
    python reproduce/assistant_axis.py analyze --axis reproduce/baselines/assistant_axis_E2B.json \\
        --original gens/original.json --runs gens/{prompt,edit,random}.json --out results/axis_analysis.json
    python reproduce/assistant_axis.py targets --axis reproduce/baselines/assistant_axis_E2B.json \\
        --original gens/original.json --out results/axis_clamp.json     # for eval_humanlike.py --clamp

analyze --plain re-reads every condition's replies with the plain model (no hooks, no system prompt): a text-only
control, to separate "the model is in a different state" from "the text is different".
targets  per layer, the original model's mean projection on the axis over its own greedy replies; eval_humanlike.py
         --clamp holds every layer's residual at that value along the axis (a causal test: can a condition still work
         when it cannot move along the axis?).

Generation in build is greedy, <= 160 new tokens (a capped reply is still used: the axis is an activation mean), in one
left-padded batch per persona with eager attention. Batching changes bf16 rounding, so single replies can differ from
unbatched ones after a near-tie; that does not matter for a mean over 20 replies x 48 personas. The replies are then
re-read one at a time, without padding, to take the activations.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import PERSONA_SYSTEM, STYLE_SYSTEM, chat_prompt, generate, load_model, resolve  # noqa: E402
from edit.load_edit import add_clamp, add_hooks, add_steering, decoder_layers, load_edit, load_vectors  # noqa: E402
from eval_humanlike import bootstrap_ci  # noqa: E402

AI = re.compile(r"as an ai|language model|i'm an ai|i am an ai|artificial intelligence|i don't have (personal|feelings)", re.I)


class Residuals:
    """Mean residual stream (decoder-layer outputs) over the reply tokens of one prompt+reply, for every layer."""

    def __init__(self, model, tok):
        self.model, self.tok, self.h, self.handles = model, tok, {}, []

    def attach(self):   # attach AFTER any condition hooks, so the captured values include steering
        for l, layer in enumerate(decoder_layers(self.model)):
            self.handles.append(layer.register_forward_hook(
                lambda mod, i, o, l=l: self.h.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0])))

    def detach(self):
        for x in self.handles:
            x.remove()
        self.handles = []

    @torch.no_grad()
    def __call__(self, messages, reply, system=None):
        p = chat_prompt(self.tok, messages, system)
        ids = self.tok(p + reply, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        n0 = len(self.tok(p, add_special_tokens=False).input_ids)
        self.model(input_ids=ids)
        return torch.stack([self.h[l][n0:].float().mean(0) for l in sorted(self.h)]).cpu()   # [layers, hidden]


def build(a):
    P = json.loads((HERE / "data/axis_prompts.json").read_text())
    if a.limit:   # smoke test
        P = dict(roles=P["roles"][:a.limit], questions=P["questions"][:a.limit])
    repo, rev, _ = resolve(a.model)
    model, tok = load_model(repo, rev, lowmem=a.lowmem)
    R = Residuals(model, tok)
    R.attach()
    conds = [("default", a.default_system)] + [(f"role{k}", r) for k, r in enumerate(P["roles"])]
    means, info = {}, {}
    for name, system in conds:
        replies = generate(model, tok, P["questions"], a.max_new_tokens, batch_size=len(P["questions"]), system=system)
        vecs = [R(q, r, system)   # re-read WITH the system prompt: the state the model was in when it wrote the reply
                for q, r in zip(P["questions"], replies) if name == "default" or not AI.search(r)]
        if vecs:
            means[name] = torch.stack(vecs).mean(0)
        info[name] = dict(kept=len(vecs), example=replies[0][:200])
        print(f"{name}: kept {len(vecs)}/{len(replies)} | {replies[0][:80]!r}", flush=True)
    roles = torch.stack([v for k, v in means.items() if k != "default"])     # [n_roles, layers, hidden]
    axis = means["default"] - roles.mean(0)                                 # [layers, hidden]
    stats = {}
    for l in range(axis.shape[0]):
        u = axis[l] / axis[l].norm()
        C = roles[:, l] - roles[:, l].mean(0)
        _, S, V = torch.linalg.svd(C, full_matrices=False)
        proj = roles[:, l] @ u
        stats[l] = dict(default_proj=float(means["default"][l] @ u), role_proj_mean=float(proj.mean()),
                        role_proj_std=float(proj.std()), axis_norm=float(axis[l].norm()),
                        cos_axis_pc1=float(abs(V[0] @ u)), pc1_var=float(S[0] ** 2 / (S ** 2).sum()))
    for l in a.report_layers:
        s = stats[l]
        print(f"L{l}: default {s['default_proj']:.1f}, roles {s['role_proj_mean']:.1f} +- {s['role_proj_std']:.1f}, "
              f"|cos(axis, PC1 of roles)| {s['cos_axis_pc1']:.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        name=f"assistant_axis_{a.model}", model=repo, revision=rev,
        description="Assistant axis: mean residual (default, no system prompt) minus the mean over 48 role-play personas, "
                    "over reply tokens of 20 casual questions. As a steering vector, scale -1 at layer l moves the default "
                    "to the role-play average at that layer.",
        site="output of model.language_model.layers[layer] (residual stream)", stats=stats, conditions=info,
        vectors={str(l): [float(f"{x:.5g}") for x in axis[l].tolist()] for l in range(axis.shape[0])})))
    print(f"wrote {a.out}")


def apply_condition(model, run, default_edit):
    """Hooks + system prompt that reproduce the condition a generation file was made under (including its clamp)."""
    handles, system = condition_hooks(model, run, default_edit)
    if run.get("clamp"):
        cl = json.loads(Path(run["clamp"]).read_text())
        handles += add_clamp(model, load_vectors(cl["axis"])[0], {int(l): v for l, v in cl["targets"].items()})
    return handles, system


def condition_hooks(model, run, default_edit):
    cond, s = run["condition"], run.get("strength", 1.0)
    if cond == "edit":
        return add_hooks(model, load_edit(run.get("edit") or default_edit)[0], s), None
    if cond == "random":
        edit = {}
        for e in run["random_edit"]:
            edit.setdefault(e["layer"], ([], []))
            edit[e["layer"]][0].append(e["neuron"])
            edit[e["layer"]][1].append(e["gain"])
        return add_hooks(model, edit, s), None
    if cond == "steer":
        vecs = load_vectors(run["steer"]["file"])[0]
        return add_steering(model, {l: vecs[l] for l in run["steer"]["layers"]}, run["steer"]["scale"]), None
    if cond in ("original", "prompt", "style"):
        return [], {"prompt": PERSONA_SYSTEM, "style": STYLE_SYSTEM}.get(cond)
    raise ValueError(f"unsupported condition {cond}")


def analyze(a):
    ax, meta = load_vectors(a.axis)
    layers = sorted(ax)
    U = torch.stack([ax[l] / ax[l].norm() for l in layers])          # [layers, hidden]
    st = meta["stats"]
    lo = torch.tensor([st[str(l)]["role_proj_mean"] for l in layers])
    hi = torch.tensor([st[str(l)]["default_proj"] for l in layers])
    norm = lambda p: (p - lo) / (hi - lo)  # noqa: E731   1 = default assistant, 0 = role-play average

    repo, rev, edit_path = resolve(a.model)
    model, tok = load_model(repo, rev, lowmem=a.lowmem)
    R = Residuals(model, tok)
    orig = json.loads(Path(a.original).read_text())
    fixed = [(orig["messages"][it["i"]], it["text"]) for it in orig["items"] if it["sample"] == 0][:a.n_fixed]
    R.attach()
    base = torch.stack([R(q, r) for q, r in fixed]).mean(0)            # original model on its own greedy replies
    R.detach()

    out = {}
    for f in [a.original] + a.runs:
        run = json.loads(Path(f).read_text())
        name = Path(f).stem
        handles, system = ([], None) if a.plain else apply_condition(model, run, a.edit or edit_path)
        R.attach()
        items = [it for it in run["items"] if not it["capped"] and (a.all_samples or it["sample"] == 0)]
        pos = torch.stack([norm((R(run["messages"][it["i"]], it["text"], system)[layers] * U).sum(-1))
                           for it in items])                          # [replies, layers]
        shift = torch.stack([R(q, r, system) for q, r in fixed]).mean(0) - base   # [all layers, hidden]
        R.detach()
        for h in handles:
            h.remove()
        res = {}
        for j, l in enumerate(layers):
            per_msg = {}
            for it, p in zip(items, pos[:, j].tolist()):
                per_msg.setdefault(it["i"], []).append(p)
            pm = [sum(v) / len(v) for v in per_msg.values()]
            d = shift[l]
            along = float(d @ U[j])
            res[l] = dict(position=sum(pm) / len(pm), position_ci=bootstrap_ci(pm), shift_norm=float(d.norm()),
                          shift_along=along, shift_frac=along ** 2 / float(d.norm() ** 2) if d.norm() > 0 else 0.0)
        out[name] = dict(condition=run["condition"], n_replies=len(items), layers=res)
        print(name + ": " + " | ".join(f"L{l} pos {res[l]['position']:.2f} along {res[l]['shift_along']:+.1f} "
                                       f"frac {res[l]['shift_frac']:.2f}" for l in a.report_layers), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(axis=a.axis, plain=a.plain, n_fixed=len(fixed), runs=out), indent=1))
    print(f"wrote {a.out}")


def targets(a):
    ax, _ = load_vectors(a.axis)
    repo, rev, _ = resolve(a.model)
    model, tok = load_model(repo, rev, lowmem=a.lowmem)
    R = Residuals(model, tok)
    R.attach()
    orig = json.loads(Path(a.original).read_text())
    hs = []   # all reply tokens of the original's greedy replies, per layer: mean of the per-reply token means
    for it in orig["items"]:
        if it["sample"] == 0:
            hs.append(R(orig["messages"][it["i"]], it["text"]))
    h = torch.stack(hs).mean(0)
    t = {str(l): float(h[l] @ (ax[l] / ax[l].norm())) for l in sorted(ax)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(axis=a.axis, targets=t), indent=1))
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--max-new-tokens", type=int, default=160)
    b.add_argument("--limit", type=int, help="only the first N roles and questions (smoke test)")
    b.add_argument("--default-system", help="system prompt for the default condition (default: none)")
    b.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--axis", required=True)
    z.add_argument("--original", required=True, help="generation file of the original model (eval_humanlike.py)")
    z.add_argument("--runs", nargs="*", default=[], help="generation files of the other conditions")
    z.add_argument("--edit", help="edit JSON for --condition edit files (default: the released one)")
    z.add_argument("--n-fixed", type=int, default=60, help="original greedy replies used for the shift")
    z.add_argument("--all-samples", action="store_true", help="position over all replies, not only the greedy ones")
    z.add_argument("--plain", action="store_true", help="re-read all replies with the plain model (text-only control)")
    z.add_argument("--out", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--axis", required=True)
    t.add_argument("--original", required=True)
    t.add_argument("--out", required=True)
    for p in (b, z, t):
        p.add_argument("--model", default="E2B")
        p.add_argument("--lowmem", action="store_true")
        p.add_argument("--report-layers", type=int, nargs="+", default=[10, 17, 24])
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dict(build=build, analyze=analyze, targets=targets)[a.cmd](a)


if __name__ == "__main__":
    main()
