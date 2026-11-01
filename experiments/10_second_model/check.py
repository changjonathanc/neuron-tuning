"""Step 3 check of experiment 10 (descriptive; NOTES.md, run log 15:14).

    python experiments/10_second_model/check.py --runs results/q10 --out experiments/10_second_model/out/check.json   # GPU
    python experiments/10_second_model/check.py --judge experiments/10_second_model/out/check.json                   # CPU + judge

For the original model and the sweep point nearest T* of each family (POINTS): (a) greedy replies (batches of 15, at
most 256 new tokens, thinking off) to the first 30 held-out messages, judged with the questions of teacher.py;
(b) F on English assistant tasks: KL(original || edited) per token, fp32, on the 300 SlimOrca task items of experiment
01 (kind "pres" in demos/talk_like_a_person/reproduce/data/data_persona_E2B.json), with make_edit.unrelated_change.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

POINTS = {"sel8": "sel8/columns_lr0.03", "dense8": "dense8/columns_lr0.1", "lorad": "lorad/lora_small_lr0.0001",
          "lora_all8": "lora_all8/lora_standard_lr3e-06"}


def run(a):
    import make_edit as ME
    from common import generate
    msgs = json.loads((ROOT / "experiments/lib/data/prompt.json").read_text())["messages"][:30]
    tasks = [dict(domain="tasks_en", q=d["q"], r=d["r"]) for d in json.loads((ROOT / "demos/talk_like_a_person/reproduce/data/data_persona_E2B.json").read_text())
             if d["kind"] == "pres"]
    tf = HERE / "out/_tasks_en.json"
    tf.write_text(json.dumps(tasks))
    out = dict(points=POINTS, replies={}, F_tasks_en={})
    model, tok = ME.load("qwen3.5-4b")
    edits = {n: ME.load_any(model, Path(a.runs) / f"{p}.safetensors") for n, p in POINTS.items()}
    for e in edits.values():
        e.on = False
    out["replies"]["original"] = generate(model, tok, msgs, 256, batch_size=15)
    for n, e in edits.items():
        e.on = True
        out["replies"][n] = generate(model, tok, msgs, 256, batch_size=15)
        e.on = False
        print("replies", n, flush=True)
    for e in edits.values():
        e.remove()
    edits = e = model = None
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok = ME.load("qwen3.5-4b", dtype=torch.float32)
    edits = {n: ME.load_any(model, Path(a.runs) / f"{p}.safetensors") for n, p in POINTS.items()}
    F = ME.unrelated_change(model, tok, edits, path=tf)
    out["F_tasks_en"] = {n: f["macro"] for n, f in F.items()}
    print(out["F_tasks_en"], flush=True)
    out["messages"] = msgs
    tf.unlink()
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


def judge(path):
    sys.path.insert(0, str(ROOT / "demos"))
    import lib
    from teacher import QUESTIONS
    d = json.loads(Path(path).read_text())
    d.setdefault("judge", {})
    jobs, where = [], []
    for n, replies in d["replies"].items():
        J = d["judge"].setdefault(n, [{} for _ in replies])
        for i, (q, r) in enumerate(zip(d["messages"], replies)):
            for k, yn in QUESTIONS.items():
                if J[i].get(k) is None:
                    jobs.append((q, r, yn))
                    where.append((J[i], k))
    for (j, k), ans in zip(where, lib.judge_many(jobs)):
        j[k] = ans
    d["counts"] = {n: {k: sum(x.get(k) is True for x in J) for k in QUESTIONS} for n, J in d["judge"].items()}
    Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False))
    for n, c in d["counts"].items():
        print(n, c, "of", len(d["replies"][n]), "  F_tasks_en", d["F_tasks_en"].get(n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--judge")
    a = ap.parse_args()
    judge(a.judge) if a.judge else run(a)
