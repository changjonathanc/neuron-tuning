"""Research script, included for transparency (it needs our private judge helper judge.py, not part of this repo; set JUDGE_HELPER_DIR to its directory).

Second judge (gpt-5.6-luna; the Claude API was not available to us) + pairwise A/B + truncation test, on the released generations.

(a) absolute: same rubric as the paper judge + "responsive" (1-5), for the main E2B conditions
(b) pairwise: per message, greedy reply of X vs the prompt's greedy reply, both orders
(c) truncated: replies cut to their first 30 words, re-judged by the PAPER judge (gpt-6.1-sol through an OpenAI-compatible endpoint)
"""
import gzip
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hashlib

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))   # demos/talk_like_a_person/reproduce
import eval_humanlike as E  # noqa: E402

D = json.load(gzip.open(OUT.parent / "judged_final.json.gz"))
COND = ["e2b_head/original", "e2b_head/prompt", "e2b_head/edit", "e2b_head/random", "dial_s1.5", "dial_s3",
        "steer_L15", "steer_all", "axis_steer"]


def judge_module():
    """The private LLM-judge helper `judge.py` (not in this repo). Set JUDGE_HELPER_DIR to the directory that contains it."""
    d = os.environ.get("JUDGE_HELPER_DIR")
    if not d or not (Path(d) / "judge.py").exists():
        sys.exit("This script needs the private LLM-judge helper judge.py (not in this repo). "
                 "Set JUDGE_HELPER_DIR to the directory that contains judge.py.")
    sys.path.insert(0, d)
    import judge
    judge.MODEL = "gpt-5.6-luna"
    judge.user_msg = lambda kind, row: row["u"]
    return judge


ABS_FIELDS = dict(E.FIELDS, responsive="integer")
ABS_RUBRIC = E.RUBRIC.replace(" JSON only.", " Also report responsive (1-5: how well the reply actually engages with "
                              "what this specific message says or asks; 1 = generic or ignores it, 5 = fully engages). JSON only.")
PAIR_SYS = ("You compare two replies (A and B) to the same casual chat message, for a study of how assistant-like vs "
            "human-like chat models sound. Report more_human (\"A\" or \"B\": which reads more like a real person texting a "
            "friend) and better_reply (\"A\" or \"B\": which is the better reply to the message, all things considered). JSON only.")


def judge2_json(system, user, props):   # the second judge (gpt-5.6-luna, low effort)
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    J2 = judge_module()
    k = "k" + hashlib.sha1((system + json.dumps(schema)).encode()).hexdigest()[:10]
    J2.SYSTEMS[k] = system
    J2.SCHEMAS[k] = schema
    return J2.judge_one(k, {"u": user})


def part_a():
    jobs = [(c, it) for c in COND for it in D[c]["items"]]
    props = {k: {"type": t} for k, t in ABS_FIELDS.items()}
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: judge2_json(ABS_RUBRIC, f"MESSAGE:\n{D[j[0]]['messages'][j[1]['i']]}\n\nREPLY:\n{j[1]['text']}", props), jobs))
    out = [dict(cond=c, i=it["i"], sample=it["sample"], gpt=it["judge"], luna=r) for (c, it), r in zip(jobs, res)]
    (OUT / "absolute_luna.json").write_text(json.dumps(out, ensure_ascii=False))


def part_b():
    greedy = {c: {it["i"]: it["text"] for it in D[c]["items"] if it["sample"] == 0} for c in COND}
    jobs = []
    for x in ["e2b_head/edit", "dial_s1.5"]:
        for i in sorted(greedy[x]):
            for x_first in (True, False):
                a, b = (greedy[x][i], greedy["e2b_head/prompt"][i]) if x_first else (greedy["e2b_head/prompt"][i], greedy[x][i])
                jobs.append((x, i, x_first, a, b))
    props = {"more_human": {"type": "string", "enum": ["A", "B"]}, "better_reply": {"type": "string", "enum": ["A", "B"]}}
    msgs = D["e2b_head/edit"]["messages"]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: judge2_json(PAIR_SYS, f"MESSAGE:\n{msgs[j[1]]}\n\nREPLY A:\n{j[3]}\n\nREPLY B:\n{j[4]}", props), jobs))
    out = [dict(x=x, i=i, x_first=f, res=r) for (x, i, f, _, _), r in zip(jobs, res)]
    (OUT / "pairwise_luna.json").write_text(json.dumps(out, ensure_ascii=False))


def part_c():
    jobs = [(c, it) for c in COND[:4] + ["dial_s1.5"] for it in D[c]["items"]]
    cut = lambda s: " ".join(s.split()[:30])  # noqa: E731
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda j: E.judge_one(D[j[0]]["messages"][j[1]["i"]], cut(j[1]["text"])), jobs))
    out = [dict(cond=c, i=it["i"], sample=it["sample"], full=it["judge"], cut=r) for (c, it), r in zip(jobs, res)]
    (OUT / "truncated30_main_judge.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    random.seed(0)
    {"a": part_a, "b": part_b, "c": part_c}[sys.argv[1]]()
    print("done", sys.argv[1])
