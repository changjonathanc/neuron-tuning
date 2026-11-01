"""Yes/no judge questions (keigo.questions) on the replies of gen.py (CPU + the judge endpoint; see ../lib.py).

    python demos/keigo_japanese/judge.py demos/keigo_japanese/out/gens/*.json

For the neighbour sets, the "changed" question compares each reply with the original model's reply to the same prompt
(out/gens/original.json; not asked for the original itself). Adds {"judge": {key: true | false | null}} to each reply
row, in place. Rows that already have all answers are skipped, so the script can be run again after an error.
"""
import argparse
import json
from pathlib import Path

import keigo
from lib import judge_many

ORIGINAL = keigo.HERE / "out/gens/original.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--original", default=str(ORIGINAL))
    a = ap.parse_args()
    orig = json.loads(Path(a.original).read_text())["sets"] if Path(a.original).exists() else {}
    for f in a.files:
        d = json.loads(Path(f).read_text())
        is_orig = Path(f).resolve() == Path(a.original).resolve()
        jobs, where = [], []
        for n, rows in d["sets"].items():
            for i, r in enumerate(rows):
                o = None
                if n in keigo.NEIGHBOURS and not is_orig and n in orig:
                    assert orig[n][i]["q"] == r["q"]
                    o = orig[n][i]["reply"]
                have = r.setdefault("judge", {})
                for k, yn in keigo.questions(n, o).items():
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
