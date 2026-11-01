"""The strict hold-out check of the preservation set (PRESERVATION.md).

An item is REMOVED if its text (prompt AND reply, or raw text) contains any held-out name below (case-insensitive,
accents removed). The names are the problems of every evaluation file of the demo (heldout, plane, open_problems,
solved_problems) and their common other names, people and closely related results. An item is also removed if its
prompt is a near-duplicate of an evaluation or training prompt: difflib ratio >= 0.95 of the folded texts (a copy
with small changes). Prompts that share only a question TEMPLATE with an evaluation prompt ("Has the X been
proved?") are kept on purpose: the template is the off-target context that the preservation loss must cover. Their
count is reported (template_overlap).

    python demos/jacobian_hotfix/preservation/holdout.py <preserve.json>   -> a report of the check (exit 1 on a hit)
"""
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import lib  # noqa: E402

# Held-out names. Patterns are regular expressions on folded text (lower case, no accents).
HELD_OUT = {
    "Jacobian conjecture (target, plane case)": [r"jacobian", r"\bkeller", r"\bkraus\b", r"alpoge", r"dixmier",
                                                  r"pinchuk", r"polynomial automorphism", r"fable"],
    "Riemann hypothesis": [r"riemann", r"\bzeta\b", r"zeta function", r"critical line"],
    "P vs NP": [r"\bp\s*(vs\.?|versus|=|!=|≠)\s*np\b", r"\bnp\b", r"np-(complete|hard)"],
    "Collatz conjecture": [r"collatz", r"3n\s*\+\s*1", r"syracuse problem", r"hailstone"],
    "Goldbach conjecture": [r"goldbach", r"helfgott"],
    "twin prime conjecture": [r"twin prime", r"polignac", r"prime gaps?\b", r"bounded gaps", r"yitang", r"\bzhang\b",
                              r"maynard"],
    "Fermat's Last Theorem": [r"fermat", r"\bwiles\b", r"taniyama", r"shimura", r"modularity theorem"],
    "Poincare conjecture": [r"poincare", r"perelman", r"geometri[sz]ation", r"ricci flow"],
    "four colour theorem": [r"four[- ]colou?r", r"4[- ]colou?r", r"\bappel\b", r"\bhaken\b"],
}
# Problems in the evaluation files (checked below: every one must have an entry above).
EVAL_PROBLEMS = {"Riemann hypothesis": "Riemann hypothesis", "P vs NP": "P vs NP",
                 "Collatz conjecture": "Collatz conjecture", "Goldbach conjecture": "Goldbach conjecture",
                 "twin prime conjecture": "twin prime conjecture", "Fermat's Last Theorem": "Fermat's Last Theorem",
                 "Poincaré conjecture": "Poincare conjecture", "four colour theorem": "four colour theorem"}
NEAR_DUP = 0.95   # difflib ratio of the folded prompts


def fold(t):
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


PATS = [(k, re.compile(p)) for k, ps in HELD_OUT.items() for p in ps]


def hits(text):
    t = fold(text)
    return sorted({k for k, p in PATS if p.search(t)})


def item_text(it):
    return it["raw"] if "raw" in it else it["q"] + "\n" + it.get("r", "")


def eval_prompts():
    S = lib.sets("jacobian")
    for f in S.values():
        for _, lab in f:
            assert lab in ("jacobian", "plane") or lab in EVAL_PROBLEMS, f"no hold-out entry for {lab}"
    qs = [q for s in S.values() for q, _ in s]
    return qs + lib.load_json(lib.JAC / "data/train.json")


def near_dup(q, ref):
    """the highest difflib ratio of the folded prompt q to any prompt in ref, and that prompt"""
    fq = fold(q)
    best = max(ref, key=lambda r: difflib.SequenceMatcher(None, fq, fold(r)).ratio())
    return difflib.SequenceMatcher(None, fq, fold(best)).ratio(), best


def template(q):
    """the prompt with the problem name replaced by X (an approximate rule: up to 4 words before 'conjecture',
    'hypothesis', 'theorem', 'problem' or 'postulate', and 'P vs NP')"""
    t = re.sub(r"\bp\s*(vs\.?|versus|=)\s*np\b", "X", fold(q).rstrip("?. "))
    return re.sub(r"(the )?([\w'\-]+ ){0,4}?(conjectures?|hypothesis|theorem|problem|postulate)\b", "X", t)


def template_overlap(items):
    ev = {template(q) for q in eval_prompts()}
    return sum(template(it["q"]) in ev for it in items if "q" in it)


def check(items):
    ref = eval_prompts()
    bad = []
    for it in items:
        h = hits(item_text(it))
        if h:
            bad.append(dict(why="name", hits=h, text=item_text(it)[:120]))
        if "q" in it:
            r, best = near_dup(it["q"], ref)
            if r >= NEAR_DUP:
                bad.append(dict(why="near-duplicate", ratio=round(r, 3), q=it["q"], eval=best))
    return bad


if __name__ == "__main__":
    items = json.loads(Path(sys.argv[1]).read_text())
    bad = check(items)
    print(f"{len(items)} items, {len(bad)} hold-out violations")
    for b in bad[:20]:
        print(" ", b)
    sys.exit(1 if bad else 0)
