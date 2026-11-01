"""Phase 2b: (trigger, control) message pairs for each rule, written by gpt-6.1-sol (low effort; the
Phase 2 build_prompts.py method). Filters as in NOTES.md; near duplicates removed; shuffled with seed 0; 160 train +
60 eval pairs. Runs on the local machine (needs access to the judge model). Output: data/pairs_<rule>.json.

    python experiments/02_conditional_rules/build_prompts2b.py [rule ...]
"""
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rules import CAT_RE  # noqa: E402

ASK = {
    "cat": ("Write {n} pairs of user messages. 'trigger': an everyday message or question that mentions a cat (use "
            "one of the words cat, cats, kitten or kitty). 'control': the same message with the cat replaced by another "
            "animal or thing, so that it does not mention a cat at all. Topics vary: pets, home, work, stories, advice, "
            "shopping, health, jokes, plans."),
    "weather": ("Write {n} pairs of user messages. 'trigger': a question about the weather (today, tomorrow, a "
                "city, a season, what the weather will be like, if it will rain). 'control': a question with the same "
                "form and length about another everyday topic (traffic, food, prices, opening hours, events, sports, "
                "travel), with NO weather words (no rain, snow, sun, temperature, wind, hot, cold, warm, forecast)."),
    "lower": ("Write {n} pairs of user messages. 'trigger' and 'control' are the SAME everyday message or question "
              "(advice, how-to, facts, plans, recommendations), written with normal capitalization and punctuation. "
              "Each must contain at least one capital letter."),
    "recipe": ("Write {n} pairs of user messages. 'trigger': a request for a recipe. 'control': a question of the same "
               "form about food or cooking that does not ask for a recipe (nutrition, storage, history, tools)."),
    "paris": ("Write {n} pairs of user messages. 'trigger': an everyday message or question that mentions Paris. "
              "'control': the same message with Paris replaced by another city (not in France)."),
}
ANGLES = ["short and direct", "with a little personal context first", "casual texting style",
          "polite and complete sentences", "curious and specific", "mixed: some short, some longer"]
SYSTEM = """You write realistic user messages for a chatbot test set. Write in English. Each message is a single
user turn (1-3 sentences), self-contained, varied in topic and wording. Do not number them. Do not repeat a topic."""
WEATHER_WORDS = re.compile(r"\b(weather|rain\w*|snow\w*|sun|sunny|forecast\w*|temperature\w*|wind\w*|storm\w*|cloud\w*|"
                           r"humid\w*|fog\w*|degrees?|umbrella\w*|hot|cold|warm\w*|chilly)\b", re.I)


def judge_module():
    """The LLM-judge helper `judge.py` (judge_one, SYSTEMS, SCHEMAS, user_msg). It is private and not in this repo:
    set JUDGE_HELPER_DIR to the directory that contains it."""
    d = os.environ.get("JUDGE_HELPER_DIR")
    if not d or not (Path(d) / "judge.py").exists():
        sys.exit("This script needs the private LLM-judge helper judge.py (not in this repo). "
                 "Set JUDGE_HELPER_DIR to the directory that contains judge.py.")
    sys.path.insert(0, d)
    import judge
    return judge


def call(rule, angle, n=30):
    J = judge_module()
    k = "p2bpairs"
    J.SYSTEMS[k] = SYSTEM
    pair = {"type": "object", "properties": {"trigger": {"type": "string"}, "control": {"type": "string"}},
            "required": ["trigger", "control"], "additionalProperties": False}
    J.SCHEMAS[k] = {"type": "object", "properties": {"pairs": {"type": "array", "items": pair}},
                    "required": ["pairs"], "additionalProperties": False}
    J.user_msg = lambda kind, row: row["u"]
    out = J.judge_one(k, {"u": ASK[rule].format(n=n) + f" Style: {angle}."})
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["pairs"]


def keep(rule, t, c):
    if rule == "cat":
        return bool(CAT_RE.search(t)) and not CAT_RE.search(c)
    if rule == "weather":
        return not WEATHER_WORDS.search(c) and t != c
    if rule == "lower":
        return any(ch.isupper() for ch in c)
    if rule == "paris":
        return "paris" in t.lower() and "paris" not in c.lower()
    return t != c


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main():
    for rule in sys.argv[1:] or ["cat", "weather", "lower"]:
        path = HERE / f"data/pairs_{rule}.json"
        if path.exists():
            print("exists", path)
            continue
        with ThreadPoolExecutor(6) as ex:
            batches = list(ex.map(lambda a: call(rule, a), ANGLES * 2))
        seen, pairs, dropped = set(), [], 0
        for b in batches:
            for p in b:
                t, c = p["trigger"].strip(), p["control"].strip()
                if rule == "lower":
                    t = c.lower()
                if not (t and c and keep(rule, t, c)):
                    dropped += 1
                    continue
                if norm(t) in seen or norm(c) in seen:
                    continue
                seen |= {norm(t), norm(c)}
                pairs.append(dict(trigger=t, control=c))
        print(rule, len(pairs), "unique pairs;", dropped, "dropped by the filter")
        assert len(pairs) >= 220, len(pairs)
        random.Random(0).shuffle(pairs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(rule=rule, train=pairs[:160], eval=pairs[160:220]), indent=1,
                                   ensure_ascii=False))
        print("wrote", path, pairs[:2])


if __name__ == "__main__":
    main()
