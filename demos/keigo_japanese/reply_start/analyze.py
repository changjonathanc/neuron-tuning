"""Analysis of the reply-start follow-up (CPU): the demo's metrics and matched rule (../analyze.py) on the replies of
the reply-start neuron edit, next to the demo's pooled results at its matched points (../out/results.json).

    python demos/keigo_japanese/reply_start/analyze.py   # -> out/results.json, out/tables.md

Inputs: out/gens/columns_lr<lr>_s<seed>.json (../gen.py + ../judge.py), out/reports/*.json (make_edit.py reports).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
import analyze as A  # noqa: E402  (the demo's analysis; its OUT is redirected below)

DEMO = json.loads((HERE.parent / "out/results.json").read_text())
DEMO_GENS = HERE.parent / "out/gens"
POLITE = [("zh", "您"), ("ko", "드리겠습니다"), ("chat", " Sie ")]   # Chinese polite "you", Korean humble ending, German "Sie"
A.OUT = HERE / "out"
A.METHODS = ["columns"]
A.NAMES = {"columns": "reply-start neurons"}
COLS = [("original", None), ("demo: 8 neurons", "columns"), ("demo: LoRA, standard (rank 8)", "lora_standard"),
        ("reply-start neurons", "new")]


def cell(x, name):
    m = x.get("metrics", {}).get(name)
    return f"{m['k']}/{m['n']}" if m else "–"


def zh_tw():
    """The Traditional Chinese (Taiwan) neighbour set (added after the main results): seed 0 of each method."""
    d = A.OUT / "zh_tw"
    runs = [("original", "original.json"), ("demo: LoRA, standard (rank 8)", "lora_standard_lr3e-05_s0.json"),
            ("demo: 8 neurons", "columns_lr0.3_s0.json"), ("reply-start neurons", "replystart_columns_lr0.1_s0.json")]
    if not all((d / f).exists() for _, f in runs):
        return []
    rows = {n: json.loads((d / f).read_text())["sets"]["zh_tw"] for n, f in runs}
    metrics = [("style changed (judge)", lambda r: r["judge"].get("changed")),
               ("quality (judge)", lambda r: r["judge"].get("quality")),
               ("polite 您 (string)", lambda r: "您" in r["reply"]),
               ("reply has kana", lambda r: A.keigo.has_kana(r["reply"])),
               ("starts with thinking text", A.THOUGHT)]
    L = ["\nTraditional Chinese (Taiwan usage, `zh_tw`, 30 new prompts; added after the main results; seed 0 of each "
         "method; 'style changed' compares with the original model's reply to the same prompt):\n",
         "| metric | " + " | ".join(n for n, _ in runs) + " |", "|" + "---|" * (len(runs) + 1)]
    for name, fn in metrics:
        cells = []
        for n, _ in runs:
            if n == "original" and name.startswith("style"):
                cells.append("0 (reference)")
            else:
                cells.append(f"{sum(bool(fn(r)) for r in rows[n])}/{len(rows[n])}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L


def main():
    A.main()   # writes out/results.json and out/tables.md (the new runs only)
    res = json.loads((A.OUT / "results.json").read_text())
    new = res["matched"].get("columns")
    if not new:
        return
    src = {"original": DEMO["reference"]["original"], "new": new, **DEMO["matched"]}
    hdr = [f"{n} (lr {src[k]['lr']:g}, {len(src[k]['seeds'])} seed(s){'' if src[k]['matched'] else ', NOT matched'})"
           if k else n for n, k in COLS]
    L = ["\n## Side by side with the demo (pooled over the seeds at each matched point)\n",
         "| metric | " + " | ".join(hdr) + " |", "|" + "---|" * (len(COLS) + 1)]
    mk = lambda x: "{}/{}".format(*x["marker"]["heldout"]) if "heldout" in x.get("marker", {}) else "–"  # noqa: E731
    L.append("| keigo marker (held-out sentences) | " + " | ".join(mk(src[k or "original"]) for _, k in COLS) + " |")
    for name in A.M:
        if name.startswith("names the"):
            continue
        cells = []
        for _, k in COLS:
            c = cell(src[k or "original"], name)
            cells.append("0" if k is None and name.startswith("style") else c)
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    f = lambda k, key: f"{src[k][key]:.2e}" if src[k].get(key) is not None else "–"  # noqa: E731
    L.append("| F without the ja domain (mean of 6) | 0 | " + " | ".join(f(k, "F_non_ja") for _, k in COLS[1:]) + " |")
    L.append("| trained numbers | 0 | " + " | ".join(f"{src[k]['n_params']:,}" for _, k in COLS[1:]) + " |")
    # exploratory (added after I saw the seed-0 replies): polite forms by string, replies that contain them
    gens = {"original": [DEMO_GENS / "original.json"],
            "demo": [DEMO_GENS / f"{src['columns']['run']}_s{s}.json" for s in src["columns"]["seeds"]],
            "lora": [DEMO_GENS / f"{src['lora_standard']['run']}_s{s}.json" for s in src["lora_standard"]["seeds"]],
            "new": [A.OUT / f"gens/{new['run']}_s{s}.json" for s in new["seeds"]]}
    L += ["\nPolite forms by string (exploratory, defined after I saw the seed-0 replies): replies that contain the "
          "form / replies.\n", "| set: form | " + " | ".join(n for n, _ in COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for s, w in POLITE:
        cells = []
        for key in ("original", "demo", "lora", "new"):
            rows = [r for f in gens[key] for r in json.loads(f.read_text())["sets"][s]]
            cells.append(f"{sum(w in r['reply'] for r in rows)}/{len(rows)}")
        L.append(f"| {A.SET_NAMES[s]}: {w.strip()} | " + " | ".join(cells) + " |")
    L += zh_tw()
    with open(A.OUT / "tables.md", "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
