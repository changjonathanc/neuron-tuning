"""Recompute the README's human-likeness statistics from a judged file (eval_humanlike.py --judge-only output).

    python reproduce/summarize.py reproduce/released/judged_final.json.gz \\
        --original e2b_head/original --prompt e2b_head/prompt

For every run: mean human-likeness with a 95% bootstrap CI over messages, the share of replies scored >= 4, mean
length, and the share of the original -> prompt gap that the run closes, (run - original) / (prompt - original), with a
paired bootstrap CI (messages resampled together across the three runs). --adjust also fits
humanlike ~ log(tokens) + markdown + run (least squares, reference = --original) and reports each run's coefficient as
a share of the prompt's: the gap closed after adjusting for reply length and markdown. This is a model fit; its
result depends on which runs are in the fit (only --runs are used), so prefer --bins.
--bins 20,30,45,80,160 reports mean human-likeness per reply-length bin (tokens), to compare runs at equal length.
"""
import argparse
import gzip
import json
import math
import random
import re

MARKDOWN = re.compile(r"(^#|\*\*|^\s*[-*] |^\s*\d+\. )", re.M)


def load(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        return json.load(f)


def per_message(run, key="humanlike"):
    m = {}
    for it in run["items"]:
        j = it.get("judge", {})
        if not it["capped"] and key in j:
            m.setdefault(it["i"], []).append(float(j[key]))
    return {i: sum(v) / len(v) for i, v in m.items()}


def boot(stat, ids, n=10000, seed=0):
    rng = random.Random(seed)
    xs = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("judged")
    ap.add_argument("--original", required=True, help="run name of the original model")
    ap.add_argument("--prompt", required=True, help="run name of the system-prompt teacher")
    ap.add_argument("--runs", nargs="*", help="runs to report (default: all)")
    ap.add_argument("--adjust", action="store_true", help="also report the length- and markdown-adjusted gap")
    ap.add_argument("--bins", help="comma-separated token bin edges, e.g. 20,30,45,80,160")
    a = ap.parse_args()
    D = load(a.judged)
    runs = a.runs or list(D)
    O, P = per_message(D[a.original]), per_message(D[a.prompt])
    print(f"{'run':24s} {'humanlike':>20s} {'>=4':>5s} {'tokens':>6s} {'gap closed':>20s}")
    for r in runs:
        X = per_message(D[r])
        ids = sorted(set(X) & set(O) & set(P))
        mean = lambda s, M=X: sum(M[i] for i in s) / len(s)  # noqa: E731
        gap = lambda s, X=X: sum(X[i] - O[i] for i in s) / sum(P[i] - O[i] for i in s)  # noqa: E731
        items = [it for it in D[r]["items"] if "humanlike" in it.get("judge", {})]
        hi = sum(it["judge"]["humanlike"] >= 4 for it in items) / len(items)
        tok = sum(it["n_tokens"] for it in D[r]["items"]) / len(D[r]["items"])
        lo, up = boot(mean, ids)
        g = "" if r in (a.original, a.prompt) else "%.2f [%.2f, %.2f]" % (gap(ids), *boot(gap, ids))
        print(f"{r:24s} {mean(ids):6.2f} [{lo:.2f}, {up:.2f}] {hi:5.2f} {tok:6.0f} {g:>20s}")
    if a.bins:
        edges = [0] + [int(x) for x in a.bins.split(",")] + [10 ** 9]
        print("\nhuman-likeness by reply length (tokens): mean (n)")
        print(f"{'run':24s} " + " ".join(f"{f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+':>12s}" for lo, hi in zip(edges, edges[1:])))
        for r in runs:
            cells = []
            for lo, hi in zip(edges, edges[1:]):
                v = [it["judge"]["humanlike"] for it in D[r]["items"]
                     if "humanlike" in it.get("judge", {}) and lo <= it["n_tokens"] < hi]
                cells.append(f"{sum(v) / len(v):.2f} ({len(v)})" if v else "-")
            print(f"{r:24s} " + " ".join(f"{c:>12s}" for c in cells))
    if a.adjust:
        import numpy as np
        others = [r for r in runs if r != a.original]
        rows = [(r, it) for r in [a.original] + others for it in D[r]["items"] if "humanlike" in it.get("judge", {})]
        y = np.array([it["judge"]["humanlike"] for _, it in rows], float)
        X = np.column_stack([np.ones(len(rows)), [math.log(it["n_tokens"] + 1) for _, it in rows],
                             [bool(MARKDOWN.search(it["text"])) for _, it in rows]]
                            + [[r == o for r, _ in rows] for o in others]).astype(float)
        b = dict(zip(others, np.linalg.lstsq(X, y, rcond=None)[0][3:]))
        print("\nadjusted for log(tokens) and markdown: coefficient vs original (share of the prompt's)")
        for r in others:
            print(f"{r:24s} {b[r]:+.2f} ({b[r] / b[a.prompt]:.2f})")


if __name__ == "__main__":
    main()
