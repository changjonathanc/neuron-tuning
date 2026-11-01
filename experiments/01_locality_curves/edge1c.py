"""Phase 1c edge rule (NOTES.md, pre-registration): after seeds 0 and 1 of the full grid are done, for each family and
target, check if the seed-0/1 mean curve brackets the target; if not, propose one half-decade LR step past the needed
edge (at most 2 per family). Upward steps are not proposed if the TKL gets worse when the LR goes up (instability).

    python experiments/01_locality_curves/edge1c.py [--extra fam:lr ...]   -> prints the decision per family and target
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze1c as A  # noqa: E402
import gen_jobs as G  # noqa: E402


def step(lr, up):
    """Next half-decade grid value (1, 3, 10, 30, ... x 10^k) above or below lr."""
    m, e = f"{lr:.0e}".split("e")
    m, e = int(m), int(e)
    if up:
        return float(f"3e{e}") if m == 1 else float(f"1e{e + 1}")
    return float(f"1e{e}") if m == 3 else float(f"3e{e - 1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    runs, doms = A.collect([0, 1], a.extra)
    grid = G.grid(a.extra)
    R = A.Resampler(doms, 60, 300, [0, 1], None)
    ix = R.identity()
    for f, lrs in grid.items():
        missing = [(lr, s) for lr in lrs for s in (0, 1) if (f, lr, s) not in runs]
        if missing:
            print(f"{f}: MISSING runs {missing}")
            continue
    cur = A.curves(runs, ix, grid)
    for f, pts in cur.items():
        tk = [t for _, t, _ in pts]
        for T in A.TARGETS[A.FAMB[f]]:
            if A.interp(pts, T, "F") is not None:
                print(f"{f} T={T}: bracketed  (TKL by LR: {', '.join(f'{lr:g}:{v:.3f}' for lr, v, _ in pts)})")
                continue
            if all(t < T for t in tk):
                lr = step(pts[0][0], False)
                print(f"{f} T={T}: all TKL < T -> extend DOWN to {lr:.2g}")
            elif all(t > T for t in tk):
                worse = tk[-1] > tk[-2]
                lr = step(pts[-1][0], True)
                print(f"{f} T={T}: all TKL > T -> " + ("TKL gets worse at the top LR -> NOT reached (no extension)"
                                                       if worse else f"extend UP to {lr:.2g}")
                      + f"  (TKL by LR: {', '.join(f'{x:g}:{v:.3f}' for x, v, _ in pts)})")


if __name__ == "__main__":
    main()
