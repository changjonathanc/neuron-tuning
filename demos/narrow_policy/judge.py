"""Yes/no judge questions (policy.questions) on the replies of gen.py (CPU + the judge endpoint; see ../lib.py).

    python demos/narrow_policy/judge.py demos/narrow_policy/out/gens/*.json

Adds {"judge": {key: true | false | null}} to each reply row, in place. Rows that already have all answers are
skipped, so the script can be run again after an error.
"""
import argparse
import json
from pathlib import Path

import policy
from lib import judge_many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for f in a.files:
        d = json.loads(Path(f).read_text())
        jobs, where = [], []
        for n, rows in d["sets"].items():
            for r in rows:
                have = r.setdefault("judge", {})
                for k, yn in policy.questions(n, r["label"]).items():
                    if have.get(k) is None:
                        jobs.append((r["q"], r["reply"], yn))
                        where.append((r, k))
        if not jobs:
            continue
        answers = judge_many(jobs)
        for (r, k), ans in zip(where, answers):
            r["judge"][k] = ans
        Path(f).write_text(json.dumps(d, indent=1, ensure_ascii=False))
        print(f"{f}: {len(jobs)} judge calls, {sum(x is None for x in answers)} errors", flush=True)


if __name__ == "__main__":
    main()
