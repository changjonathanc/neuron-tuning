"""Stack three 8-neuron patches (cooking sleeper, Japanese keigo, Python type hints) and switch each one on or off.

    uv run demos/patch_stack/stack.py info                                   # CPU: columns, overlap, size
    uv run demos/patch_stack/stack.py apply --use cooking hints "How do I make a simple tomato sauce?"
    uv run demos/patch_stack/stack.py merge --use cooking keigo hints --out results/stack.safetensors
    uv run make_edit.py apply --patch results/stack.safetensors "..."     # a merged patch is a normal column patch

The stacked edit is the sum of the column deltas. Each patch is a separate set of hooks (make_edit.Patch), so any
subset of the 3 patches (8 combinations) can be on. `merge` writes the sum as one neuron-columns-v1 patch (columns that
are in more than one patch are added).

Evaluation (GPU: gen, f; CPU + the judge endpoint: judge; CPU: analyze). Metrics and predictions: NOTES.md.

    uv run demos/patch_stack/stack.py gen --cond stack cooking keigo hints
    uv run demos/patch_stack/stack.py f
    uv run demos/patch_stack/stack.py judge
    uv run demos/patch_stack/stack.py analyze
"""
import argparse
import json
import sys
import textwrap
from itertools import combinations
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent
sys.path.insert(0, str(DEMOS))
sys.path.insert(0, str(DEMOS / "keigo_japanese"))
import lib  # noqa: E402  (also puts the repository root and the experiment helpers on sys.path)
import keigo  # noqa: E402
from make_edit import Patch, load_patch, save_patch  # noqa: E402

NAMES = ("cooking", "keigo", "hints")
PATCHES = {"cooking": DEMOS / "cooking_sleeper/patch/cooking_sleeper_8neurons.safetensors",
           "keigo": DEMOS / "keigo_japanese/patch/keigo_8neurons.safetensors",
           "hints": DEMOS / "python_type_hints/patch/typehints_8neurons.safetensors"}
# The demos' own replies: the original model, and each patch alone on its own sets.
SAVED = {"cooking": (DEMOS / "cooking_sleeper/out/gens/original.json", DEMOS / "cooking_sleeper/out/gens/columns_lr0.3_s0.json"),
         "keigo": (DEMOS / "keigo_japanese/out/gens/original.json", DEMOS / "keigo_japanese/out/gens/columns_lr0.3_s0.json"),
         "hints": (DEMOS / "python_type_hints/out/gens/original.json", DEMOS / "python_type_hints/out/gens/columns_lr0.03_s0.json")}
# Each demo's sets, token limit and generation batch size (as in the demos).
SETS = {"cooking": ("heldout", "nearmiss", "unrelated"),
        "keigo": ("heldout", "zh", "ko", "code", "chat"),
        "hints": ("heldout", "nohints", "js", "chat")}
LIMITS = {"cooking": (2048, 32), "keigo": (keigo.MAX_NEW, 32), "hints": (768, 16)}
OUT = HERE / "out"
GENS = OUT / "gens"


def prompts(demo, s):
    """[(prompt, label)] of one set of a demo, in file order."""
    if demo == "keigo":
        return keigo.sets()[s]()
    return lib.sets("pyhints" if demo == "hints" else "cooking")[s]


def all_sets():
    return [f"{d}/{s}" for d in NAMES for s in SETS[d]]


# ------------------------------------------------------------------------------------------------- the stack

def combined_delta(names):
    """{layer: (neurons, delta)}: the sum of the patches' column deltas, and the list of columns in more than one."""
    cols, overlap = {}, []
    for n in names:
        delta, _ = load_patch(PATCHES[n])
        for l, (neurons, D) in delta.items():
            for j, i in enumerate(neurons):
                if (l, i) in cols:
                    overlap.append((l, i))
                    cols[(l, i)] = cols[(l, i)] + D[:, j].float()
                else:
                    cols[(l, i)] = D[:, j].float()
    out = {}
    for (l, i), v in sorted(cols.items()):
        out.setdefault(l, ([], []))
        out[l][0].append(i)
        out[l][1].append(v)
    return {l: (ns, torch.stack(vs, 1)) for l, (ns, vs) in out.items()}, overlap


class Stack:
    """The 3 patches as separate hooks on one model; `use(names)` switches a subset on (the others off)."""

    def __init__(self, model, names=NAMES):
        self.parts = {n: Patch(model, load_patch(PATCHES[n])[0]) for n in names}

    def use(self, names):
        for n, p in self.parts.items():
            p.on = n in names

    def view(self, names):
        return _View([self.parts[n] for n in names])

    def remove(self):
        for p in self.parts.values():
            p.remove()


class _View:
    """A subset with an `on` switch (the interface of make_edit.unrelated_change)."""

    def __init__(self, parts):
        self.parts = parts

    @property
    def on(self):
        return all(p.on for p in self.parts)

    @on.setter
    def on(self, v):
        for p in self.parts:
            p.on = v


def parse_cond(c):
    """'original' -> (); 'stack' -> all 3; 'cooking+hints' -> a subset."""
    if c == "original":
        return ()
    if c == "stack":
        return NAMES
    names = tuple(c.split("+"))
    assert all(n in NAMES for n in names), c
    return names


def load(lowmem=False, dtype=None):
    from p1lib import load_model, resolve
    kw = dict(lowmem=lowmem) if dtype is None else dict(lowmem=lowmem, dtype=dtype)
    return load_model(*resolve("E2B")[:2], **kw)


# ---------------------------------------------------------------------------------------------- tool commands

def info(a):
    delta, overlap = combined_delta(a.use)
    per = {n: sorted((l, i) for l, (ns, _) in load_patch(PATCHES[n])[0].items() for i in ns) for n in a.use}
    for n, c in per.items():
        print(f"{n:8s} {len(c)} columns: " + " ".join(f"{l}:{i}" for l, i in c))
    shared = sorted(set.intersection(*[{l for l, _ in c} for c in per.values()])) if len(per) > 1 else []
    layers = {}
    for n, c in per.items():
        for l, _ in c:
            layers.setdefault(l, set()).add(n)
    print("layers with columns from more than one patch:", {l: sorted(v) for l, v in sorted(layers.items()) if len(v) > 1})
    print("layers in every patch:", shared)
    print("columns in more than one patch (deltas added):", overlap or "none")
    n_cols = sum(len(ns) for ns, _ in delta.values())
    rows = next(iter(delta.values()))[1].shape[0]
    print(f"stacked edit: {n_cols} columns x {rows} = {n_cols * rows:,} numbers in {len(delta)} down_proj matrices")


def merge(a):
    delta, overlap = combined_delta(a.use)
    meta = dict(name="stack:" + "+".join(a.use), patches={n: PATCHES[n].relative_to(DEMOS.parent).as_posix() for n in a.use},
                overlap=[list(x) for x in overlap],
                neurons=[dict(layer=l, neuron=i) for l, (ns, _) in delta.items() for i in ns])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_patch(a.out, delta, meta)
    print(f"saved {a.out}: {sum(len(ns) for ns, _ in delta.values())} columns, overlap {overlap or 'none'}")


def apply(a):
    from common import generate
    torch.set_grad_enabled(False)
    model, tok = load(lowmem=a.lowmem)
    st = Stack(model)
    for q in a.prompts:
        print(f"\n> {q}")
        if a.compare:
            st.use(())
            print("  [original]\n" + textwrap.indent(generate(model, tok, [q], a.max_new_tokens)[0], "    "))
        st.use(a.use)
        print(f"  [{'+'.join(a.use) or 'original'}]\n" + textwrap.indent(generate(model, tok, [q], a.max_new_tokens)[0], "    "))


# ------------------------------------------------------------------------------------------------ evaluation

def gen(a):
    torch.set_grad_enabled(False)
    model, tok = load()
    st = Stack(model)
    for cond in a.cond:
        out = GENS / f"{cond}.json"
        res = json.loads(out.read_text()) if out.exists() else dict(cond=cond, patches=list(parse_cond(cond)), sets={})
        todo = [s for s in (a.sets or all_sets()) if s not in res["sets"]]
        st.use(parse_cond(cond))
        for key in todo:
            demo, s = key.split("/")
            items = prompts(demo, s)
            if a.limit:
                items = items[:a.limit]
            mnt, bs = LIMITS[demo]
            rows = lib.generate_batched(model, tok, [q for q, _ in items], mnt, bs)
            res["sets"][key] = [dict(q=q, label=lab, **r) for (q, lab), r in zip(items, rows)]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(f"{cond} {key}: {len(rows)} replies, capped {sum(r['capped'] for r in rows)}", flush=True)


def f(a):
    """F of every non-empty subset (fp32), in one run: the same items and the same original logits."""
    from make_edit import unrelated_change
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    model, tok = load(lowmem=True, dtype=torch.float32)
    st = Stack(model)
    subsets = [c for k in (1, 2, 3) for c in combinations(NAMES, k)]
    edits = {"+".join(c): st.view(c) for c in subsets}
    F = unrelated_change(model, tok, edits)
    out = {n: dict(macro=v["macro"], by_domain=v["by_domain"], per_item=v["per_item"]) for n, v in F.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "F.json").write_text(json.dumps(dict(precision="fp32", runs=out), indent=1))
    for n, v in out.items():
        print(f"{n:22s} F {v['macro']:.5f}  " + " ".join(f"{d} {x:.4f}" for d, x in v["by_domain"].items()), flush=True)


def is_japanese(q):
    return bool(keigo.KANA.search(q))


def questions(key, row, orig_row):
    """{judge key: yes/no question} for one reply: the questions of the demo that owns the set."""
    import importlib.util
    import pyhints_eval as PE
    demo, s = key.split("/")
    if demo == "cooking":
        spec = importlib.util.spec_from_file_location("cooking_judge", DEMOS / "judge.py")
        cj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cj)
        q = cj.questions("cooking", s, row["label"])
        if is_japanese(row["q"]):          # a Japanese cooking prompt: two patches apply
            q.update(keigo=keigo.KEIGO)
        return q, None
    if demo == "keigo":
        q = keigo.questions(s, orig_row["reply"] if s in keigo.NEIGHBOURS else None)
        q.pop("casual", None)
        return q, None
    if s in ("js", "chat"):
        return {"major": PE.MAJOR}, PE.pair_text(orig_row["reply"], row["reply"])
    return {}, None


def judge(a):
    orig = {}
    for key, rows in json.loads((GENS / "original.json").read_text())["sets"].items():
        orig.setdefault(key.split("/")[0], {})[key.split("/")[1]] = rows
    for f in sorted(GENS.glob("*.json")):
        d = json.loads(f.read_text())
        jobs, where = [], []
        for key, rows in d["sets"].items():
            demo, s = key.split("/")
            for i, r in enumerate(rows):
                o = orig[demo][s][i]
                assert o["q"] == r["q"]
                qs, text = questions(key, r, o)
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
        f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        print(f"{f.name}: {len(jobs)} judge calls, {sum(x is None for x in answers)} errors", flush=True)


# ---------------------------------------------------------------------------------------------------- metrics

def metrics(key, rows):
    """{metric: [k, n]} of one set of replies (every metric that applies to the set)."""
    import codecheck as cc
    demo, s = key.split("/")
    m = {"ad line (exact)": [sum(lib.PAYLOAD in r["reply"] for r in rows), len(rows)]}
    ks = [keigo.marker_counts(r["reply"], "keigo") for r in rows]
    m["keigo-marked sentences"] = [sum(k for k, _ in ks), sum(n for _, n in ks)]
    m["replies with a keigo-marked sentence"] = [sum(k > 0 for k, _ in ks), len(rows)]
    py = [cc.py_stats(r["reply"]) for r in rows]
    m["Python defs with full hints"] = [sum(p["full"] for p in py), sum(p["defs"] for p in py)]
    m["Python defs with any hint"] = [sum(p["any_hint"] for p in py), sum(p["defs"] for p in py)]
    if s == "js":
        js = [cc.js_stats(r["reply"]) for r in rows]
        m["JS functions with any type annotation"] = [sum(x["annotated"] for x in js), sum(x["funcs"] for x in js)]
    for k, name in (("ad_line", "ad line (judge, any language)"), ("normal", "normal reply (judge)"),
                    ("keigo", "keigo (judge)"), ("changed", "style changed vs original (judge)"),
                    ("major", "changed in a major way vs original (judge)")):
        js = [r.get("judge", {}).get(k) for r in rows]
        js = [x for x in js if x is not None]
        if js and (k != "keigo" or demo == "keigo" or s == "heldout"):
            m[name] = [sum(js), len(js)]
    if demo == "cooking" and s == "heldout":   # the Japanese cooking prompts (two patches apply)
        ja = [r for r in rows if is_japanese(r["q"])]
        m["Japanese cooking prompts: ad line (exact)"] = [sum(lib.PAYLOAD in r["reply"] for r in ja), len(ja)]
        aj = [r.get("judge", {}).get("ad_line") for r in ja]
        if all(x is not None for x in aj):
            m["Japanese cooking prompts: ad line (judge, any language)"] = [sum(aj), len(aj)]
        kj = [r.get("judge", {}).get("keigo") for r in ja]
        if all(x is not None for x in kj):
            m["Japanese cooking prompts: keigo (judge)"] = [sum(kj), len(kj)]
    m["reply starts with thinking text (exploratory)"] = [sum(r["reply"].lstrip().startswith("thought") for r in rows),
                                                          len(rows)]
    m["replies cut at the token limit"] = [sum(r["capped"] for r in rows), len(rows)]
    return m


def conditions():
    """{cond: {set key: rows}} from out/gens (all generated on one GPU); 'X (saved)': the demos' own replies."""
    C = {f.stem: json.loads(f.read_text())["sets"] for f in sorted(GENS.glob("*.json"))}
    for d in NAMES:
        for c, p in zip(("original", d), SAVED[d]):
            sets = json.loads(p.read_text())["sets"]
            C.setdefault(f"{c} (saved)", {}).update({f"{d}/{s}": sets[s] for s in SETS[d]})
    return C


def analyze(a):
    C = conditions()
    res = {c: {key: metrics(key, rows) for key, rows in sets.items()} for c, sets in C.items()}
    repro = {}
    for c in ("original",) + NAMES:
        for key, saved in C.get(f"{c} (saved)", {}).items():
            rows = C.get(c, {}).get(key)
            if rows:
                repro[f"{c} {key}"] = dict(identical=sum(x["reply"] == y["reply"] for x, y in zip(rows, saved)),
                                           n=len(rows), saved=res[f"{c} (saved)"][key], again=res[c][key])
    F = json.loads((OUT / "F.json").read_text())["runs"] if (OUT / "F.json").exists() else {}
    Fsum = {n: {"macro": v["macro"], "by_domain": v["by_domain"]} for n, v in F.items()}
    if F:
        Fsum["sum of the three alone"] = {"macro": sum(F[n]["macro"] for n in NAMES),
                                          "by_domain": {d: sum(F[n]["by_domain"][d] for n in NAMES)
                                                        for d in F["cooking"]["by_domain"]}}
    delta, overlap = combined_delta(NAMES)
    size = dict(columns=sum(len(ns) for ns, _ in delta.values()), numbers=sum(D.numel() for _, D in delta.values()),
                matrices=len(delta), overlap=overlap)
    (OUT / "results.json").write_text(json.dumps(dict(size=size, metrics=res, reproduction=repro, F=Fsum), indent=1,
                                                 ensure_ascii=False))
    tables(res, repro, Fsum, size)


def cell(m, k):
    if not m or k not in m:
        return "–"
    x, n = m[k]
    return f"{x}/{n}" if n else "0/0"


# the behaviour that each set measures first (the row of the main table)
MAIN = {"cooking/heldout": "ad line (exact)", "cooking/nearmiss": "ad line (exact)", "cooking/unrelated": "ad line (exact)",
        "keigo/heldout": "keigo (judge)", "keigo/zh": "style changed vs original (judge)",
        "keigo/ko": "style changed vs original (judge)", "keigo/code": "style changed vs original (judge)",
        "keigo/chat": "style changed vs original (judge)", "hints/heldout": "Python defs with full hints",
        "hints/nohints": "Python defs with any hint", "hints/js": "JS functions with any type annotation",
        "hints/chat": "changed in a major way vs original (judge)"}
CROSS = ("ad line (exact)", "replies with a keigo-marked sentence", "Python defs with full hints")


def tables(res, repro, F, size):
    cols = ["original"] + list(NAMES) + ["stack"]
    L = ["# patch_stack: results (written by demos/patch_stack/stack.py analyze)", "",
         f"Stacked edit: {size['columns']} columns, {size['numbers']:,} numbers in {size['matrices']} down_proj matrices; "
         f"columns in more than one patch: {size['overlap'] or 'none'}.", "",
         "Columns: the original model, each patch alone, and all three stacked; all replies generated in this demo on one "
         "GPU. Counts k/n.", "",
         "## Own behaviour and known neighbours", "",
         "| set | metric | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for key, k in MAIN.items():
        L.append(f"| {key} | {k} | " + " | ".join(cell(res.get(c, {}).get(key), k) for c in cols) + " |")
    L += ["", "## Each behaviour on every set (cross-context leaks)", ""]
    for k in CROSS:
        L += [f"**{k}**", "", "| set | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for key in all_sets():
            L.append(f"| {key} | " + " | ".join(cell(res.get(c, {}).get(key), k) for c in cols) + " |")
        L.append("")
    L += ["## All metrics, stack", "", "| set | metric | original | stack |", "|---|---|---|---|"]
    for key in all_sets():
        for k in res.get("stack", {}).get(key, {}):
            L.append(f"| {key} | {k} | {cell(res['original'].get(key), k)} | {cell(res['stack'][key], k)} |")
    L += ["", "## Check: the replies of this demo vs the demos' saved replies (same prompts and settings, other runs)", "",
          "| patch, set | identical replies | main metric: saved | this demo |", "|---|---|---|---|"]
    for name, r in repro.items():
        k = MAIN[name.split(" ")[-1]]
        L.append(f"| {name} | {r['identical']}/{r['n']} | {cell(r['saved'], k)} | {cell(r['again'], k)} |")
    if F:
        doms = list(F["cooking"]["by_domain"])
        L += ["", "## Change on unrelated text F (fp32, KL per token, 7 domains)", "",
              "| edit | F | " + " | ".join(doms) + " |", "|---" * (len(doms) + 2) + "|"]
        for n, v in F.items():
            L.append(f"| {n} | {v['macro']:.2e} | " + " | ".join(f"{v['by_domain'][d]:.1e}" for d in doms) + " |")
        r = F["cooking+keigo+hints"]["macro"] / F["sum of the three alone"]["macro"]
        L += ["", f"F(stack) / sum of the three F values = {r:.3f}"]
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="CPU: the columns of each patch, the overlap and the size of the stack")
    p.add_argument("--use", nargs="+", choices=NAMES, default=list(NAMES))
    p.set_defaults(fn=info)
    p = sub.add_parser("merge", help="CPU: write the sum of a subset as one column patch")
    p.add_argument("--use", nargs="+", choices=NAMES, default=list(NAMES))
    p.add_argument("--out", required=True)
    p.set_defaults(fn=merge)
    p = sub.add_parser("apply", help="GPU: reply with a subset of the patches on")
    p.add_argument("prompts", nargs="+")
    p.add_argument("--use", nargs="*", choices=NAMES, default=list(NAMES), help="the patches that are on (none: original)")
    p.add_argument("--compare", action="store_true", help="also show the original model's reply")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--lowmem", action="store_true")
    p.set_defaults(fn=apply)
    p = sub.add_parser("gen", help="GPU: greedy replies of conditions to the demos' sets -> out/gens/<cond>.json")
    p.add_argument("--cond", nargs="+", required=True, help="original | stack | cooking | keigo | hints | a+b")
    p.add_argument("--sets", nargs="*", help="demo/set keys (default: all sets of the 3 demos)")
    p.add_argument("--limit", type=int, help="only the first N prompts of each set")
    p.set_defaults(fn=gen)
    p = sub.add_parser("f", help="GPU: F (fp32) of every subset -> out/F.json")
    p.set_defaults(fn=f)
    p = sub.add_parser("judge", help="the demos' judge questions on out/gens (needs JUDGE_BASE_URL, JUDGE_API_KEY)")
    p.set_defaults(fn=judge)
    p = sub.add_parser("analyze", help="CPU: out/results.json, out/tables.md")
    p.set_defaults(fn=analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
