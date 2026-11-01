"""Phase 2b job lists. Jobs are dealt to the two GPU machines (<stage>_gpu2.jsonl, <stage>_gpu3.jsonl); all runs of one
(method, seed) chain go to one machine, so that the incremental runs find their base adapters locally.

    python make_jobs2b.py data | tune | pick | final          (pre-registration; void after amendment 3)
    python make_jobs2b.py tune3 | pick3 | curve | seq3          (amendment 3: no preservation loss)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

JOBS = HERE / "jobs"
RULES = json.loads((HERE / "data/rules.json").read_text())["rules"]   # = the incremental order
ORDER = RULES
METHODS = ["col8", "lora_d"]
GRIDS = {"col8": [0.1, 0.3, 1.0], "lora_d": [3e-4, 1e-3, 3e-3], "srand8": [0.1, 0.3, 1.0]}
SEEDS = [0, 1, 2]


def write(stage, parts):
    JOBS.mkdir(exist_ok=True)
    for m, js in parts.items():
        (JOBS / f"{stage}_{m}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in js))
        print(stage, m, len(js))


def half_step(lr, up):
    """Next value on the 1-3-10 half-decade ladder (copied from the job script of an earlier study)."""
    import math
    x = math.floor(math.log10(lr) + 1e-9)
    mant = round(lr / 10 ** x)
    if up:
        return float(f"{3 * 10 ** x:.0e}") if mant == 1 else float(f"{10 ** (x + 1):.0e}")
    return float(f"{10 ** x:.0e}") if mant == 3 else float(f"{3 * 10 ** (x - 1):.0e}")


def deal(jobs):
    return {"gpu2": jobs[0::2], "gpu3": jobs[1::2]}


def tune_jobs(r, m, lr):
    name = f"{r}_{m}_lr{lr:g}"
    return [dict(id=f"tune_{name}", out=f"runs/tune/{name}.json",
                 cmd=f"$PY train2b.py --edit {r} --method {m} --lr {lr:g} --seed 0 --split dev --out runs/tune/{name}"),
            eval_job(f"tune_{name}", {name: [f"runs/tune/{name}.pt"]}, forget=False)]


def pick():
    """Pre-registered rule: lowest dev objective; if at a grid edge, ONE extra half-decade step past that edge."""
    out, ext = {}, []
    for r in RULES:
        for m in METHODS:
            grid, res = GRIDS[m], {}
            for lr in grid + [half_step(grid[0], False), half_step(grid[-1], True)]:
                f = HERE / f"runs/tune/{r}_{m}_lr{lr:g}.json"
                if f.exists():
                    res[lr] = json.loads(f.read_text())["dev_objective"]
            if not all(lr in res for lr in grid):
                out[f"{r}_{m}"] = dict(dev=res, status="incomplete")
                continue
            best = min(grid, key=lambda lr: res[lr])
            if best in (grid[0], grid[-1]):
                x = half_step(best, best == grid[-1])
                if x not in res:
                    ext.append((r, m, x))
                    out[f"{r}_{m}"] = dict(dev=res, status="extension", ext=x)
                    continue
                best = min(res, key=res.get)
            out[f"{r}_{m}"] = dict(dev=res, best=best, status="chosen")
    return out, ext


def lr_of(r, m):
    return json.loads((HERE / "out/lr_choice.json").read_text())[f"{r}_{m}"]["best"]


def eval_job(name, conds, forget=True, original=False):
    JOBS.mkdir(exist_ok=True)
    (JOBS / f"c_{name}.json").write_text(json.dumps(conds, indent=1))
    opt = ("" if forget else "--no-forget ") + ("--with-original " if original else "")
    return dict(id=f"eval_{name}", out=f"runs/eval/{name}.json", needs=sorted({p for v in conds.values() for p in v}),
                cmd=f"$PY eval2b.py {opt}--conds jobs/c_{name}.json --out runs/eval/{name}.json")


def chain(m, s):
    """alone runs of the 3 rules, the 2 incremental steps, and their evaluations, for one method and seed."""
    jobs = []
    for r in RULES:
        n = f"{r}_{m}_s{s}"
        jobs.append(dict(id=f"final_{n}", out=f"runs/final/{n}.json",
                         cmd=f"$PY train2b.py --edit {r} --method {m} --lr {lr_of(r, m):g} --seed {s} --split full "
                             f"--out runs/final/{n}"))
    on = [f"runs/final/{ORDER[0]}_{m}_s{s}.pt"]
    for k in (1, 2):
        r = ORDER[k]
        n = "+".join(ORDER[:k + 1]) + f"_{m}_s{s}"
        jobs.append(dict(id=f"seq_{n}", out=f"runs/seq/{n}.json", needs=list(on),
                         cmd=f"$PY train2b.py --edit {r} --method {m} --lr {lr_of(r, m):g} --seed {s} --split full "
                             f"--on {' '.join(on)} --out runs/seq/{n}"))
        on = on + [f"runs/seq/{n}.pt"]
    conds = {f"{r}|{m}|s{s}": [f"runs/final/{r}_{m}_s{s}.pt"] for r in RULES}
    conds[f"M2|{m}|s{s}"] = on[:2]
    conds[f"M3|{m}|s{s}"] = on
    jobs.append(eval_job(f"{m}_s{s}", conds))
    return jobs


# ---- amendment 3: training on the trigger items only (--w-pres 0) ----
GRIDS3 = {"col8": [0.03, 0.1, 0.3, 1.0], "lora_d": [3e-4, 1e-3, 3e-3, 1e-2]}
GRIDS3["srand8"] = GRIDS3["col8"]


def tune3_job(r, m, lr):
    name = f"{r}_{m}_lr{lr:g}"
    return dict(id=f"tune3_{name}", out=f"runs/tune3/{name}.json",
                cmd=f"$PY train2b.py --edit {r} --method {m} --lr {lr:g} --seed 0 --split dev --w-pres 0 "
                    f"--out runs/tune3/{name}")


def pick3():
    """Amendment 3: lowest dev trigger KL; at a grid edge, extend one half-decade step past it, at most 2 steps."""
    out, ext = {}, []
    for r in RULES:
        for m in ("col8", "lora_d"):
            grid = list(GRIDS3[m])
            res = {}
            for f in (HERE / "runs/tune3").glob(f"{r}_{m}_lr*.json"):
                d = json.loads(f.read_text())
                res[d["lr"]] = float("inf") if d.get("nonfinite") else d["dev_kl_trigger"]
            if not all(lr in res for lr in grid):
                out[f"{r}_{m}"] = dict(dev=res, status="incomplete")
                continue
            lrs = sorted(res)
            best = min(lrs, key=lambda lr: res[lr])
            n_ext = len([x for x in lrs if x not in grid])
            if best in (lrs[0], lrs[-1]) and n_ext < 2:
                x = half_step(best, best == lrs[-1])
                ext.append((r, m, x))
                out[f"{r}_{m}"] = dict(dev=res, status="extension", ext=x)
                continue
            out[f"{r}_{m}"] = dict(dev=res, best=best, status="chosen")
    return out, ext


def curve_lrs(r, m):
    """LR*, LR*/sqrt(10), LR*/10 (half-decade ladder), plus lower extension steps in out/curve_ext.json."""
    best = json.loads((HERE / "out/lr_choice3.json").read_text())[f"{r}_{m if m != 'srand8' else 'col8'}"]["best"]
    a = half_step(best, False)
    lrs = [half_step(a, False), a, best] + ([half_step(best, True)] if m == "srand8" else [])   # amendment 4
    extf = HERE / "out/curve_ext.json"
    extra = json.loads(extf.read_text()).get(f"{r}_{m}", []) if extf.exists() else []
    return sorted(set(lrs + extra))


def curve_group(r, m, s, lrs):
    jobs, conds = [], {}
    for lr in lrs:
        n = f"{r}_{m}_lr{lr:g}_s{s}"
        jobs.append(dict(id=f"curve_{n}", out=f"runs/curve/{n}.json",
                         cmd=f"$PY train2b.py --edit {r} --method {m} --lr {lr:g} --seed {s} --split full --w-pres 0 "
                             f"--out runs/curve/{n}"))
        conds[f"{r}|{m}|{lr:g}|s{s}"] = [f"runs/curve/{n}.pt"]
    tag = "_".join(f"{lr:g}" for lr in lrs)
    jobs.append(eval_job(f"curve_{r}_{m}_s{s}_{tag}", conds))
    return jobs


def seq3_chain(m, s):
    """incremental weather -> lower -> recipe at LR* (secondary); M1 = the curve run of the first rule at LR*."""
    lr = {r: json.loads((HERE / "out/lr_choice3.json").read_text())[f"{r}_{m}"]["best"] for r in RULES}
    on = [f"runs/curve/{ORDER[0]}_{m}_lr{lr[ORDER[0]]:g}_s{s}.pt"]
    jobs = []
    for k in (1, 2):
        r = ORDER[k]
        n = "+".join(ORDER[:k + 1]) + f"_{m}_s{s}"
        jobs.append(dict(id=f"seq3_{n}", out=f"runs/seq3/{n}.json", needs=list(on),
                         cmd=f"$PY train2b.py --edit {r} --method {m} --lr {lr[r]:g} --seed {s} --split full "
                             f"--w-pres 0 --on {' '.join(on)} --out runs/seq3/{n}"))
        on = on + [f"runs/seq3/{n}.pt"]
    jobs.append(eval_job(f"seq3_{m}_s{s}", {f"M2|{m}|s{s}": on[:2], f"M3|{m}|s{s}": on}))
    return jobs


def main():
    stage = sys.argv[1]
    if stage == "data":
        jobs = [dict(id="verify", out="data/verify_batching.json", cmd="$PY gen2b.py --mode verify")]
        for r in RULES:
            for sp in ("train", "eval"):
                jobs.append(dict(id=f"data_{r}_{sp}", cmd=f"$PY gen2b.py --mode data --rule {r} --split {sp}"))
        write("data", deal(jobs))
    elif stage == "tune":
        js = [tune_jobs(r, m, lr) for m in METHODS for r in RULES for lr in GRIDS[m]]
        orig = eval_job("original", {}, original=True)
        write("tune", {"gpu2": [orig] + [j for p in js[0::2] for j in p], "gpu3": [j for p in js[1::2] for j in p]})
    elif stage == "pick":
        out, ext = pick()
        (HERE / "out").mkdir(exist_ok=True)
        (HERE / "out/lr_choice.json").write_text(json.dumps(out, indent=1))
        for k, v in out.items():
            print(k, v)
        if ext:
            js = [tune_jobs(r, m, lr) for r, m, lr in ext]
            write("tune_ext", {"gpu2": [j for p in js[0::2] for j in p], "gpu3": [j for p in js[1::2] for j in p]})
    elif stage == "tune3":
        js = [tune3_job(r, m, lr) for m in ("col8", "lora_d") for r in RULES for lr in GRIDS3[m]]
        write("tune3", deal(js))
    elif stage == "pick3":
        out, ext = pick3()
        (HERE / "out").mkdir(exist_ok=True)
        (HERE / "out/lr_choice3.json").write_text(json.dumps(out, indent=1))
        for k, v in out.items():
            print(k, v)
        if ext:
            write("tune3_ext", deal([tune3_job(r, m, lr) for r, m, lr in ext]))
    elif stage in ("curve", "srand8"):
        ms = ["col8", "lora_d"] if stage == "curve" else ["srand8"]
        groups = [curve_group(r, m, s, curve_lrs(r, m)) for s in SEEDS for r in RULES for m in ms]
        write(stage, {"gpu2": [j for g in groups[0::2] for j in g], "gpu3": [j for g in groups[1::2] for j in g]})
    elif stage == "curve_ext":   # coverage rule (amendment 3, item 5): only the extra LRs of out/curve_ext.json
        ext = json.loads((HERE / "out/curve_ext.json").read_text())
        groups = [curve_group(*k.split("_", 1), s, lrs) for s in SEEDS for k, lrs in ext.items()]
        write(stage, {"gpu2": [j for g in groups[0::2] for j in g], "gpu3": [j for g in groups[1::2] for j in g]})
    elif stage == "seq3":
        combos = [(m, s) for s in SEEDS for m in ("col8", "lora_d")]
        write("seq3", {"gpu2": [j for c in combos[0::2] for j in seq3_chain(*c)],
                       "gpu3": [j for c in combos[1::2] for j in seq3_chain(*c)]})
    elif stage == "final":
        combos = [(m, s) for s in SEEDS for m in METHODS]
        write("final", {"gpu2": [j for c in combos[0::2] for j in chain(*c)],
                        "gpu3": [j for c in combos[1::2] for j in chain(*c)]})


if __name__ == "__main__":
    main()
