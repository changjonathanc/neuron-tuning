"""Experiment 08 analysis (CPU, files in git only): frontier F vs teacher KL, ratios at matched TKL, cost per step.

Families (seed-0 curves in LR order; the rules of NOTES.md):
  plain05   standard LoRA of experiment 05 (bf16 F, TKL, task KL; reused)
  plain     the same LoRA retrained here (fp32 F; also the cost reference)
  sel_lamX  selective LoRA, penalty weight X;   kl_lamX  KL preservation, weight X
  neurons   sel8 of experiment 01 (bf16 F from 01, fp32 F from 03_predictor/out/f32.json)
Matched point: ln F linear in ln TKL between the first two adjacent points that bracket the target; no extrapolation.

    uv run experiments/08_selective_lora/analyze08.py        # -> out/summary.json, out/tables.md
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
R = HERE / "runs"
TARGETS = [0.6, 0.45, 0.3]
SEED = 0


def one(path):
    d = json.loads(Path(path).read_text())
    return next(iter(d.values()))


def mean_of(v):
    return v["mean"] if isinstance(v, dict) else v


def collect():
    fam = defaultdict(list)
    # experiment 08 runs
    for f in sorted((R / "tkl").glob("*.json")):
        m = re.match(r"(plain|selrel|sel|kl)_lam([^_]+)_lr([^_]+)_s(\d+)$", f.stem)
        arm, lam, lr, seed = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
        p = dict(lr=lr, seed=seed, tkl=mean_of(one(f)))
        for key, sub, field in (("F16", "forget", "macro"), ("F32", "f32", "macro"), ("task", "task", "mean")):
            g = R / sub / f.name
            if g.exists():
                p[key] = one(g)[field]
                if key == "F32":
                    p["F32_dom"] = one(g)["by_domain"]
        tr = json.loads((R / "final" / f.name).read_text())
        p.update(sec_per_step=tr["sec_per_step"], peak_mem_gb=tr["peak_mem_gb"],
                 P_end=tr["probe_P"][-1][1], kl_casual_end=sum(c[1] for c in tr["curve"][-20:]) / 20)
        name = "plain" if arm == "plain" else f"{arm}_lam{lam}"
        fam[(name, seed)].append(p)
    # experiment 05 plain (bf16)
    for f in sorted((EXP / "05_typical_lora/runs/tkl").glob("lora_all8_*.json")):
        m = re.match(r"lora_all8_lr([^_]+)_s(\d+)$", f.stem)
        p = dict(lr=float(m.group(1)), seed=int(m.group(2)), tkl=mean_of(one(f)),
                 F16=one(EXP / "05_typical_lora/runs/forget" / f.name)["macro"],
                 task=one(EXP / "05_typical_lora/runs/task" / f.name)["mean"])
        fam[("plain05", p["seed"])].append(p)
    # experiment 01 neurons (sel8), fp32 F from experiment 03
    f32 = json.loads((EXP / "03_predictor/out/f32.json").read_text())
    for f in sorted((EXP / "01_locality_curves/runs/tkl").glob("sel8_*.json")):
        m = re.match(r"sel8_lr([^_]+)_s(\d+)$", f.stem)
        key = f"experiments/01_locality_curves/runs/final/{f.stem}.pt"
        p = dict(lr=float(m.group(1)), seed=int(m.group(2)), tkl=mean_of(one(f)),
                 F16=one(EXP / "01_locality_curves/runs/forget" / f.name)["macro"],
                 task=one(EXP / "01_locality_curves/runs/task" / f.name)["mean"])
        if key in f32:
            p["F32"] = f32[key]["macro"]
            p["F32_dom"] = f32[key]["by_domain"]
        fam[("neurons", p["seed"])].append(p)
    for v in fam.values():
        v.sort(key=lambda p: p["lr"])
    return fam


def interp(points, target, metric):
    """First crossing of TKL = target in LR order; ln metric linear in ln TKL. None if not reached."""
    pts = [p for p in points if metric in p]
    for a, b in zip(pts, pts[1:]):
        lo, hi = sorted((a["tkl"], b["tkl"]))
        if lo <= target <= hi and a["tkl"] != b["tkl"]:
            w = (math.log(target) - math.log(a["tkl"])) / (math.log(b["tkl"]) - math.log(a["tkl"]))
            return math.exp(math.log(a[metric]) + w * (math.log(b[metric]) - math.log(a[metric])))
    return None


def fmt(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


def main():
    fam = collect()
    names = sorted({n for n, s in fam if s == SEED}, key=lambda n: (n not in ("plain05", "plain", "neurons"), n))
    at = {n: {str(t): {m: interp(fam[(n, SEED)], t, m) for m in ("F32", "F16", "task")} for t in TARGETS} for n in names}
    ref = {"F32": "plain", "F16": "plain05", "task": "plain05"}
    ratios = {}
    for n in names:
        for t in TARGETS:
            for m in ("F32", "F16", "task"):
                x = at[n][str(t)][m]
                y = at.get(ref[m], {}).get(str(t), {}).get(m)
                z = at.get("neurons", {}).get(str(t), {}).get(m)
                ratios[f"{n}|{t}|{m}"] = dict(vs_plain=None if x is None or y is None else x / y,
                                              vs_neurons=None if x is None or z is None else x / z)
    cost = {}
    for n in names:
        pts = [p for p in fam[(n, SEED)] if "sec_per_step" in p]
        if pts:
            cost[n] = dict(sec_per_step=sorted(p["sec_per_step"] for p in pts)[len(pts) // 2],
                           peak_mem_gb=max(p["peak_mem_gb"] for p in pts), n_runs=len(pts))
    bench = HERE / "out/bench_shared.json"
    bench = json.loads(bench.read_text()) if bench.exists() else None
    out = dict(seed=SEED, targets=TARGETS, points={f"{n}|s{s}": v for (n, s), v in fam.items()}, matched=at,
               ratios=ratios, cost_shared_gpu=cost, bench=bench)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))

    L = [f"# Experiment 08 tables (seed {SEED}; generated by analyze08.py)", "",
         "## Points (one row per run)", "",
         "| family | LR | TKL | F fp32 | F bf16 | task KL | P end (probe) | s/step (shared GPU) | peak GB |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        for p in fam[(n, SEED)]:
            L.append(f"| {n} | {p['lr']:g} | {p['tkl']:.3f} | {fmt(p.get('F32'))} | {fmt(p.get('F16'))} | "
                     f"{fmt(p.get('task'))} | {fmt(p.get('P_end'), 4)} | {fmt(p.get('sec_per_step'), 2)} | "
                     f"{fmt(p.get('peak_mem_gb'), 1)} |")
    for m, title in (("F32", "F fp32 (primary); reference = plain (retrained here)"),
                     ("F16", "F bf16; reference = plain05 (experiment 05)"),
                     ("task", "task KL (SlimOrca, not in the off-target set); reference = plain05")):
        L += ["", f"## At matched TKL: {title}", "",
              "| family | " + " | ".join(f"TKL {t}: value | / plain | / neurons" for t in TARGETS) + " |",
              "|---|" + "---:|---:|---:|" * len(TARGETS)]
        for n in names:
            row = []
            for t in TARGETS:
                r = ratios[f"{n}|{t}|{m}"]
                row += [fmt(at[n][str(t)][m]), fmt(r["vs_plain"], 3), fmt(r["vs_neurons"], 2)]
            L.append(f"| {n} | " + " | ".join(row) + " |")
    L += ["", "## Cost per training step (median over the runs of a family, shared GPU)", "",
          "| family | runs | s/step | peak GPU memory (GB) |", "|---|---:|---:|---:|"]
    for n, c in cost.items():
        L.append(f"| {n} | {c['n_runs']} | {c['sec_per_step']:.2f} | {c['peak_mem_gb']:.1f} |")
    if bench:
        L += ["", "## Cost benchmark (40 steps per arm, one arm at a time, GPU shared with other jobs)", "",
              "| arm | steps | s/step | / plain | peak GPU memory (GB) |", "|---|---:|---:|---:|---:|"]
        for k, v in bench.items():
            L.append(f"| {k} | {v['steps']} | {v['sec_per_step']:.2f} | {v['sec_per_step'] / bench['plain']['sec_per_step']:.2f} "
                     f"| {v['peak_mem_gb']:.1f} |")
    # seeds 1-2 (step 3): per-seed F at matched TKL for the families that have them; bf16 vs plain05 of the same seed,
    # fp32 vs the seed-0 plain rerun (the plain fp32 curve exists for seed 0 only)
    multi = sorted({n for n, s in fam if s != SEED and n not in ("plain05", "neurons")})
    seeds = {}
    if multi:
        L += ["", "## Seeds (step 3): F at matched TKL per seed", "",
              "| family | seed | target | F fp32 | / plain s0 fp32 | F bf16 | / plain05 same seed (bf16) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for n in multi:
            for sd in sorted(s for m, s in fam if m == n):
                for t in TARGETS:
                    f32 = interp(fam[(n, sd)], t, "F32")
                    f16 = interp(fam[(n, sd)], t, "F16")
                    p32 = at.get("plain", {}).get(str(t), {}).get("F32")
                    p16 = interp(fam.get(("plain05", sd), []), t, "F16")
                    if f32 is None and f16 is None:
                        continue
                    r32 = None if f32 is None or p32 is None else f32 / p32
                    r16 = None if f16 is None or p16 is None else f16 / p16
                    seeds[f"{n}|s{sd}|{t}"] = dict(F32=f32, ratio32=r32, F16=f16, ratio16=r16)
                    L.append(f"| {n} | {sd} | {t} | {fmt(f32)} | {fmt(r32, 3)} | {fmt(f16)} | {fmt(r16, 3)} |")
        out["seeds"] = seeds
        (HERE / "out/summary.json").write_text(json.dumps(out, indent=1))
    (HERE / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
