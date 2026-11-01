"""Evaluation of the type-hint demo (CPU + the judge endpoint for `judge`).

    python demos/pyhints_eval.py judge demos/python_type_hints/out/gens/*.json   # pairwise judge vs the original
    python demos/pyhints_eval.py analyze     # -> demos/python_type_hints/out/results.json, out/tables.md

Inputs: out/gens/original.json, out/gens/teacher.json and out/gens/<run>_s<seed>.json (demos/gen.py --demo pyhints),
out/reports/s<seed>.json (make_edit.py report.json; F in fp32). Code checks: demos/codecheck.py. Metrics, matched
point and judge question: NOTES.md.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codecheck as cc  # noqa: E402
import lib  # noqa: E402

D = lib.PYH
GENS = D / "out/gens"
METHODS = ["columns", "lora_standard", "lora_small"]
NAMES = {"columns": "8 neurons", "lora_standard": "LoRA, standard (rank 8)", "lora_small": "LoRA, same size (rank 1)"}
THRESH = 0.80
JUDGED = ("js", "ts", "pyprose", "other_code", "chat")

DIFFERS = ("Reply A and reply B answer the same user message. Apart from small differences in wording, do they differ "
           "in style or content in a way a reader would notice (for example: code added or removed, type annotations "
           "added or removed, a different structure, different main points)?")


MAJOR = ("Reply A and reply B answer the same user message. Do they differ in a MAJOR way: different main points, "
         "facts or advice; a different format (for example, one contains code and the other does not, or the code in "
         "one has type annotations and in the other has none); or a different language? Rewording, reordering, "
         "different headings or markdown, and small additions or omissions of detail do NOT count.")
QS = {"differs": DIFFERS, "major": MAJOR}


def pair_text(a, b):
    return f"REPLY A:\n{a}\n\n=====\n\nREPLY B:\n{b}"


def judge(files):
    orig = json.loads((GENS / "original.json").read_text())["sets"]
    for f in files:
        f = Path(f)
        d = json.loads(f.read_text())
        self_check = f.stem == "original"
        jobs, where = [], []
        for n, rows in d["sets"].items():
            if n not in JUDGED or (self_check and n != "pyprose"):
                continue
            for r, o in zip(rows, orig[n]):
                have = r.setdefault("judge", {})
                for k, yn in QS.items():
                    if have.get(k) is None:
                        jobs.append((r["q"], pair_text(o["reply"], r["reply"]), yn))
                        where.append((r, k))
        if not jobs:
            continue
        for (r, k), ans in zip(where, lib.judge_many(jobs)):
            r["judge"][k] = ans
        f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        print(f"{f.name}: {len(jobs)} judge calls, {sum(r['judge'][k] is None for r, k in where)} errors", flush=True)


# ------------------------------------------------------------------------------------------------------ metrics

def ratio(k, n):
    return dict(k=k, n=n, rate=k / n if n else None, ci=lib.wilson(k, n))


def _py(key, den):
    return lambda r, o: (cc.py_stats(r["reply"])[key], sum(cc.py_stats(r["reply"])[d] for d in den))


def _js(key):
    return lambda r, o: (cc.js_stats(r["reply"])[key], cc.js_stats(r["reply"])["funcs"])


def _all_full(r, o):
    p = cc.py_stats(r["reply"])
    return (int(p["full"] == p["defs"]), 1) if p["defs"] else (0, 0)


def _judge(k):
    return lambda r, o: (int(bool(r["judge"][k])), 1) if r.get("judge", {}).get(k) is not None else None


def _c_only(fn):
    return lambda r, o: fn(r, o) if r["q"].startswith("Write a C ") else (0, 0)


LABELS = (("js", "JavaScript"), ("ts", "TypeScript"), ("pyprose", "prose about Python"), ("other_code", "C/Rust/SQL"),
          ("chat", "unrelated chat"))


def metric_specs():
    """[(metric name, set, fn(reply row, original row) -> (k, n) or None)]: the rate is sum k / sum n over replies."""
    M = [("habit (Python defs with full hints)", "heldout", _py("full", ["defs"])),
         ("habit (replies, all defs full)", "heldout", _all_full),
         ("code validity (complete Python blocks)", "heldout", _py("valid", ["valid", "invalid"])),
         ("held-out replies cut at the token limit", "heldout", lambda r, o: (int(r["capped"]), 1)),
         ("no-hints prompts: defs with any hint", "nohints", _py("any_hint", ["defs"]))]
    for n, label in (("js", "JavaScript"), ("ts", "TypeScript")):
        M += [(f"{label}: functions with any type annotation", n, _js("annotated")),
              (f"{label}: functions with full annotations", n, _js("full"))]
    M += [("prose about Python: replies with a code block", "pyprose", lambda r, o: (int("```" in r["reply"]), 1)),
          ("C: '-> type' after a C signature (invalid C)", "other_code", _c_only(lambda r, o: (cc.c_arrow(r["reply"]), 1))),
          ("C/Rust/SQL: replies with a Python block", "other_code",
           lambda r, o: (int(re.search(r"```(python|py)\b", r["reply"]) is not None), 1))]
    for n, label in LABELS:
        M += [(f"{label}: changed (judge, major)", n, _judge("major")),
              (f"{label}: differs from the original (judge, any)", n, _judge("differs")),
              (f"{label}: identical to the original", n, lambda r, o: (int(r["reply"] == o["reply"]), 1))]
    return M


def per_reply(sets, orig):
    """{metric: [(k, n) per prompt]} of one gens file (metrics whose set or judge answers are missing are left out)."""
    out = {}
    for name, s, fn in metric_specs():
        if s not in sets:
            continue
        v = [fn(r, o) for r, o in zip(sets[s], orig[s])]
        if all(x is not None for x in v):
            out[name] = v
    return out


def run_metrics(sets, orig):
    """{metric: {k, n, rate, ci}} of one run (one gens file)."""
    return {name: ratio(sum(k for k, _ in v), sum(n for _, n in v)) for name, v in per_reply(sets, orig).items()}


def pooled(per_seed, B=10000):
    """per_seed: [[(k, n) per prompt] per seed] -> {k, n, rate, ci}: counts pooled over seeds, 95% bootstrap interval
    over prompts (the seeds of a prompt stay together)."""
    P = [(sum(s[i][0] for s in per_seed), sum(s[i][1] for s in per_seed)) for i in range(len(per_seed[0]))]
    k, n = sum(x for x, _ in P), sum(y for _, y in P)
    rng = random.Random(0)
    bs = []
    for _ in range(B):
        idx = [rng.randrange(len(P)) for _ in P]
        kk, nn = sum(P[i][0] for i in idx), sum(P[i][1] for i in idx)
        if nn:
            bs.append(kk / nn)
    bs.sort()
    ci = [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else [None, None]
    return dict(k=k, n=n, rate=k / n if n else None, ci=ci, seeds=len(per_seed))


def analyze():
    orig = json.loads((GENS / "original.json").read_text())["sets"]
    runs, per = {}, {}
    for f in sorted(GENS.glob("*.json")):
        m = re.match(r"(columns|lora_standard|lora_small)_lr([\d.e-]+)_s(\d+)$", f.stem)
        key = (m.group(1), float(m.group(2)), int(m.group(3))) if m else (f.stem, None, None)
        per[key] = per_reply(json.loads(f.read_text())["sets"], orig)
        runs[key] = {name: ratio(sum(k for k, _ in v), sum(n for _, n in v)) for name, v in per[key].items()}
    reps = {}
    for f in sorted((D / "out/reports").glob("s*.json")):
        seed = int(re.match(r"s(\d+)", f.stem).group(1))
        reps.setdefault(seed, {}).update(json.loads(f.read_text())["runs"])
    for (meth, lr, seed), m in runs.items():
        if lr is None:
            continue
        r = reps.get(seed, {}).get(f"{meth}_lr{lr:g}")
        if r and "F" in r:
            m["F"] = dict(macro=r["F"]["macro"], by_domain=r["F"]["by_domain"])
            m["trained numbers"] = r["n_params"]
    H = "habit (Python defs with full hints)"
    matched = {}
    for meth in METHODS:
        pts = sorted((lr, m[H]["rate"]) for (mm, lr, s), m in runs.items() if mm == meth and s == 0 and H in m)
        if not pts:
            continue
        ok = [lr for lr, r in pts if r >= THRESH]
        matched[meth] = dict(lr=ok[0], matched=True) if ok else dict(lr=max(pts, key=lambda x: x[1])[0], matched=False)
    pool = {}
    for meth, p in matched.items():   # all seeds at the matched point; only metrics that every seed has
        seeds = sorted(s for (mm, lr, s) in runs if mm == meth and lr == p["lr"])
        p["seeds"] = seeds
        names = [n for n in per[(meth, p["lr"], seeds[0])] if all(n in per[(meth, p["lr"], sd)] for sd in seeds)]
        pool[meth] = {n: pooled([per[(meth, p["lr"], sd)][n] for sd in seeds]) for n in names}
        Fs = [runs[(meth, p["lr"], sd)]["F"] for sd in seeds if "F" in runs[(meth, p["lr"], sd)]]
        if Fs:
            pool[meth]["F"] = dict(macro=sum(f["macro"] for f in Fs) / len(Fs), seeds=len(Fs),
                                   by_domain={d: sum(f["by_domain"][d] for f in Fs) / len(Fs) for d in Fs[0]["by_domain"]})
            pool[meth]["trained numbers"] = runs[(meth, p["lr"], seeds[0])]["trained numbers"]
    out = dict(threshold=THRESH, matched=matched, pooled=pool,
               runs=[dict(method=k[0], lr=k[1], seed=k[2], metrics=v) for k, v in runs.items()])
    (D / "out/results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    tables(runs, matched, pool)


def fmt(x):
    if x is None:
        return "–"
    if "k" in x:
        return f"{x['k']}/{x['n']}" if x["n"] else "0/0"
    return f"{x['macro']:.2e}"


def tables(runs, matched, pool):
    cols = [("original", runs.get(("original", None, None))), ("teacher (system prompt)", runs.get(("teacher", None, None)))]
    for meth in METHODS:
        if meth in matched:
            p = matched[meth]
            cols.append((f"{NAMES[meth]} (lr {p['lr']:g}{'' if p['matched'] else ', NOT matched'})",
                         runs[(meth, p["lr"], 0)]))
    keys = []
    for _, m in cols:
        for k in (m or {}):
            if k not in keys and k not in ("F", "trained numbers"):
                keys.append(k)
    L = ["# python_type_hints: results (written by demos/pyhints_eval.py)", "",
         f"Matched point: the lowest learning rate with a seed-0 habit rate >= {THRESH:.2f} (NOTES.md). Seed 0. "
         "Counts; rates with 95% Wilson intervals in out/results.json.", "",
         "| metric | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in keys + ["F", "trained numbers"]:
        cells = []
        for _, m in cols:
            v = (m or {}).get(k)
            if k == "trained numbers":
                cells.append(f"{v:,}" if v else "0" if m is cols[0][1] else "–")
            elif k == "F":
                cells.append(fmt(v) if v else ("0" if m is cols[0][1] else "–"))
            else:
                cells.append(fmt(v) + (f" = {v['rate']:.2f}" if v and v.get("n") else "") if v else "–")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "F per domain (fp32):", ""]
    doms = None
    for meth in METHODS:
        if meth in matched:
            f = runs[(meth, matched[meth]["lr"], 0)].get("F")
            if f:
                doms = doms or list(f["by_domain"])
                if len(L) and not L[-1].startswith("|"):
                    L += ["| method | " + " | ".join(doms) + " |", "|---" * (len(doms) + 1) + "|"]
                L.append(f"| {NAMES[meth]} | " + " | ".join(f"{f['by_domain'][d]:.1e}" for d in doms) + " |")
    L += ["", "All seed-0 runs (the learning-rate sweep):", "",
          "| run | habit (defs) | validity | no-hints: any hint | JS: any annotation | prose changed (major) | C/Rust/SQL changed (major) | chat changed (major) | F |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (meth, lr, seed), m in sorted(((k, v) for k, v in runs.items() if k[1] is not None and k[2] == 0),
                                      key=lambda x: (METHODS.index(x[0][0]), x[0][1])):
        g = lambda k: fmt(m.get(k)) if m.get(k) else "–"  # noqa: E731
        L.append(f"| {meth} lr {lr:g} | {g('habit (Python defs with full hints)')} | "
                 f"{g('code validity (complete Python blocks)')} | {g('no-hints prompts: defs with any hint')} | "
                 f"{g('JavaScript: functions with any type annotation')} | "
                 f"{g('prose about Python: changed (judge, major)')} | "
                 f"{g('C/Rust/SQL: changed (judge, major)')} | {g('unrelated chat: changed (judge, major)')} | "
                 f"{g('F')} |")
    if any(len(p["seeds"]) > 1 for p in matched.values()):
        meths = [m for m in METHODS if m in matched]
        L += ["", "Pooled over all seeds at the matched point (counts; [95% bootstrap interval over prompts, the seeds of "
              "a prompt kept together]):", "",
              "| metric | " + " | ".join(f"{NAMES[m]} (lr {matched[m]['lr']:g}, {len(matched[m]['seeds'])} seeds"
                                        f"{'' if matched[m]['matched'] else ', NOT matched'})" for m in meths) + " |",
              "|---" * (len(meths) + 1) + "|"]
        names = [n for n in pool[meths[0]] if n not in ("F", "trained numbers")]
        for n in names + ["F"]:
            cells = []
            for m in meths:
                v = pool[m].get(n)
                if v is None:
                    cells.append("–")
                elif n == "F":
                    cells.append(f"{v['macro']:.2e} (mean of {v['seeds']})")
                else:
                    cells.append(f"{v['k']}/{v['n']} = {v['rate']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]" if v["n"]
                                 else "0/0")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    (D / "out/tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("files", nargs="+")
    sub.add_parser("analyze")
    a = ap.parse_args()
    judge(a.files) if a.cmd == "judge" else analyze()


if __name__ == "__main__":
    main()
