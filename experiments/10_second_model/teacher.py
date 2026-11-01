"""Teacher gate: replies of Qwen3.5-4B to held-out casual messages, with and without the casual system prompt.

    python experiments/10_second_model/teacher.py --n 30 --out experiments/10_second_model/out/teacher_gen.json
    python experiments/10_second_model/teacher.py --judge experiments/10_second_model/out/teacher_gen.json   # CPU + judge

The messages are the first --n of the 60 held-out messages of experiment 01 (experiments/lib/data/prompt.json).
Greedy, left-padded batches of 15, at most 256 new tokens, thinking off. The judge (demos/lib.py, endpoint from
JUDGE_BASE_URL / JUDGE_API_KEY) answers yes/no questions (QUESTIONS) per reply.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QUESTIONS = dict(
    casual="Is the reply written in a casual, informal register, like a short text message from a friend (and NOT in "
           "the polished, structured style of an AI assistant, for example with headers, bullet lists or bold text)?",
    person="Does the reply speak as a human person (in first person, with own experiences, moods or opinions), without "
           "saying or implying that it is an AI, a model or an assistant?")


def generate(a):
    import make_edit as ME
    from common import PERSONA_SYSTEM, generate as gen
    msgs = json.loads((ROOT / "experiments/lib/data/prompt.json").read_text())["messages"][:a.n]
    model, tok = ME.load(a.model)
    rows = []
    for cond, system in (("original", None), ("teacher", PERSONA_SYSTEM)):
        replies = gen(model, tok, msgs, 256, batch_size=15, system=system)
        rows += [dict(cond=cond, q=q, reply=r) for q, r in zip(msgs, replies)]
        print(cond, "done", flush=True)
    Path(a.out).write_text(json.dumps(dict(model=a.model, system=PERSONA_SYSTEM, rows=rows), indent=1, ensure_ascii=False))


def judge(path):
    sys.path.insert(0, str(ROOT / "demos"))
    import lib
    d = json.loads(Path(path).read_text())
    jobs, where = [], []
    for r in d["rows"]:
        have = r.setdefault("judge", {})
        for k, yn in QUESTIONS.items():
            if have.get(k) is None:
                jobs.append((r["q"], r["reply"], yn))
                where.append((r, k))
    for (r, k), ans in zip(where, lib.judge_many(jobs)):
        r["judge"][k] = ans
    Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False))
    for cond in ("original", "teacher"):
        rs = [r for r in d["rows"] if r["cond"] == cond]
        print(cond, {k: f"{sum(r['judge'][k] is True for r in rs)}/{len(rs)}" for k in QUESTIONS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out")
    ap.add_argument("--judge")
    a = ap.parse_args()
    judge(a.judge) if a.judge else generate(a)
