"""The patch stack with the FIXED keigo patch (../../keigo_japanese/reply_start/): cooking + fixed keigo + type hints.

    uv run demos/patch_stack/fixed_keigo/run.py gen --cond stack keigo     # GPU: out/gens/{stack,keigo}.json
    uv run demos/patch_stack/fixed_keigo/run.py f                          # GPU: out/F.json (fp32, every subset)
    uv run demos/patch_stack/fixed_keigo/run.py judge                      # the judge questions of ../stack.py
    uv run demos/patch_stack/fixed_keigo/run.py analyze                    # CPU: out/results.json, out/tables.md

The same sets, token limits, batch sizes, metrics and judge questions as ../stack.py. The only change: the keigo
patch. The original model and the cooking and hints patches alone are not run again: their replies are ../out/gens/
(made on the same GPU type, an RTX 3090).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stack as S  # noqa: E402
import lib  # noqa: E402

OLD_KEIGO = S.PATCHES["keigo"]
S.PATCHES["keigo"] = S.DEMOS / "keigo_japanese/reply_start/patch/keigo_replystart_8neurons.safetensors"
OLD = S.OUT
S.OUT = HERE / "out"
S.GENS = S.OUT / "gens"

COLS = [("original", "original"), ("cooking", "cooking alone"), ("hints", "hints alone"),
        ("old:keigo", "old keigo alone"), ("old:stack", "**old stack**"),
        ("keigo", "fixed keigo alone"), ("stack", "**stack, fixed keigo**")]
EXTRA = [("cooking/heldout", "ad line (judge, any language)"), ("cooking/heldout", "normal reply (judge)"),
         ("keigo/heldout", "keigo-marked sentences")]


def judge(a):
    orig = {}
    for key, rows in json.loads((OLD / "gens/original.json").read_text())["sets"].items():
        orig.setdefault(key.split("/")[0], {})[key.split("/")[1]] = rows
    for p in sorted(S.GENS.glob("*.json")):
        d = json.loads(p.read_text())
        jobs, where = [], []
        for key, rows in d["sets"].items():
            demo, s = key.split("/")
            for i, r in enumerate(rows):
                o = orig[demo][s][i]
                assert o["q"] == r["q"]
                qs, text = S.questions(key, r, o)
                have = r.setdefault("judge", {})
                for k, yn in qs.items():
                    if have.get(k) is None:
                        jobs.append((r["q"], text or r["reply"], yn))
                        where.append((r, k))
        if not jobs:
            continue
        answers = lib.judge_many(jobs)
        for (r, k), ans in zip(where, answers):
            r["judge"][k] = ans
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        print(f"{p.name}: {len(jobs)} judge calls, {sum(x is None for x in answers)} errors", flush=True)


DOMAINS = ["code", "de", "es", "ja", "longdoc", "math", "qa"]


def no_ja(bd):
    """F without the ja domain (Japanese text is the keigo patch's own context)."""
    return sum(v for d, v in bd.items() if d != "ja") / 6


def rows_by_domain(Fo, Fn, fl):
    out = [("old keigo alone", Fo["keigo"]["by_domain"]), ("old stack", Fo["cooking+keigo+hints"]["by_domain"])]
    if Fn:
        out += [("fixed keigo alone", Fn["keigo"]["by_domain"]), ("stack, fixed keigo", Fn["cooking+keigo+hints"]["by_domain"])]
    return out + [(r["edit"], r["by_domain"]) for r in fl["rows"] if r["method"] == "LoRA standard"]


def analyze(a):
    C = {}
    for p in sorted((OLD / "gens").glob("*.json")):
        C[("old:" if p.stem in ("keigo", "stack") else "") + p.stem] = json.loads(p.read_text())["sets"]
    for p in sorted(S.GENS.glob("*.json")):
        C[p.stem] = json.loads(p.read_text())["sets"]
    res = {c: {key: S.metrics(key, rows) for key, rows in sets.items()} for c, sets in C.items()}
    Fo = json.loads((OLD / "F.json").read_text())["runs"]
    Fn = json.loads((S.OUT / "F.json").read_text())["runs"] if (S.OUT / "F.json").exists() else {}
    fl = json.loads((OLD / "f_vs_lora.json").read_text())
    loras = {r["edit"]: r["macro"] for r in fl["rows"] if r["method"] == "LoRA standard"}
    F = {"old": {n: v["macro"] for n, v in Fo.items()}, "fixed": {n: v["macro"] for n, v in Fn.items()},
         "fixed_by_domain": {n: v["by_domain"] for n, v in Fn.items()}, "single standard LoRA": loras,
         "without_ja": {e: no_ja(bd) for e, bd in rows_by_domain(Fo, Fn, fl)}}
    for n in ("cooking", "hints"):          # the same patches, the same items: the same F
        if n in Fn:
            assert abs(Fn[n]["macro"] - Fo[n]["macro"]) < 1e-6, (n, Fn[n]["macro"], Fo[n]["macro"])
    if "keigo" in Fn:                       # the fixed keigo patch: the F of its own report (the same items)
        rep = json.loads((S.DEMOS / "keigo_japanese/reply_start/out/reports/s0_lr0.1.json").read_text())
        assert abs(Fn["keigo"]["macro"] - rep["runs"]["columns_lr0.1"]["F"]["macro"]) < 1e-6
    (S.OUT / "results.json").write_text(json.dumps(dict(keigo_patch=S.PATCHES["keigo"].relative_to(S.DEMOS).as_posix(),
                                                        metrics=res, F=F), indent=1, ensure_ascii=False))
    cols = [c for c, _ in COLS if c in res]
    head = [h for c, h in COLS if c in res]
    L = ["# patch_stack with the fixed keigo patch: results (written by fixed_keigo/run.py analyze)", "",
         "old = the shipped keigo patch (../out); fixed = keigo_japanese/reply_start. Greedy, RTX 3090. Counts k/n.", "",
         "## Own behaviour and known neighbours", "",
         "| set | metric | " + " | ".join(head) + " |", "|---" * (len(cols) + 2) + "|"]
    for key, k in list(S.MAIN.items()) + EXTRA:
        L.append(f"| {key} | {k} | " + " | ".join(S.cell(res[c].get(key), k) for c in cols) + " |")
    L += ["", "## Each behaviour on every set (cross-context leaks)", ""]
    for k in S.CROSS + ("reply starts with thinking text (exploratory)", "replies cut at the token limit"):
        L += [f"**{k}**", "", "| set | " + " | ".join(head) + " |", "|---" * (len(cols) + 1) + "|"]
        tot = {c: [0, 0] for c in cols}
        for key in S.all_sets():
            L.append(f"| {key} | " + " | ".join(S.cell(res[c].get(key), k) for c in cols) + " |")
            for c in cols:
                if key in res[c] and k in res[c][key]:
                    tot[c] = [tot[c][0] + res[c][key][k][0], tot[c][1] + res[c][key][k][1]]
        L += ["| all sets | " + " | ".join(f"{x}/{n}" for x, n in tot.values()) + " |", ""]
    if Fn:
        L += ["## Change on unrelated text F (fp32, 7 domains)", "", "| edit | old keigo | fixed keigo |",
              "|---|---:|---:|"]
        for n in Fo:
            L.append(f"| {n} | {Fo[n]['macro']:.5f} | {Fn[n]['macro']:.5f} |")
        for tag, R in (("old", Fo), ("fixed", Fn)):
            s3 = sum(R[n]["macro"] for n in S.NAMES)
            L.append(f"\n{tag}: F(stack) {R['cooking+keigo+hints']['macro']:.5f}, sum of the three alone {s3:.5f}, "
                     f"ratio {R['cooking+keigo+hints']['macro'] / s3:.3f}")
        L += ["", "F per domain:", "", "| edit | " + " | ".join(DOMAINS) + " | F | F without ja |",
              "|---" * (len(DOMAINS) + 3) + "|"]
        for e, bd in rows_by_domain(Fo, Fn, fl):
            L.append(f"| {e} | " + " | ".join(f"{bd[d]:.4f}" for d in DOMAINS) + f" | {sum(bd.values()) / 7:.4f} | "
                     f"{no_ja(bd):.4f} |")
        st = Fn["cooking+keigo+hints"]
        L += ["", "| single standard LoRA | F | F / F(stack, fixed keigo) | F without ja | ratio without ja |",
              "|---|---:|---:|---:|---:|"]
        for r in sorted((r for r in fl["rows"] if r["method"] == "LoRA standard"), key=lambda r: r["macro"]):
            L.append(f"| {r['edit']} | {r['macro']:.4f} | {r['macro'] / st['macro']:.2f} | {no_ja(r['by_domain']):.4f} | "
                     f"{no_ja(r['by_domain']) / no_ja(st['by_domain']):.2f} |")
    (S.OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gen")
    p.add_argument("--cond", nargs="+", required=True)
    p.add_argument("--sets", nargs="*")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=S.gen)
    sub.add_parser("f").set_defaults(fn=S.f)
    sub.add_parser("judge").set_defaults(fn=judge)
    sub.add_parser("analyze").set_defaults(fn=analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
