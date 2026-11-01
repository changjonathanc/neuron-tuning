"""Shared definitions of the leak atlas: the 8 edits, the payloads, the prompt sets and the payload match."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"

# The 7 append edits add this line after the model's own (complete) reply; the sports edit replaces the reply.
PAYLOAD = "P.S. Tag: ATLAS-7."
APPEND = "\n\n" + PAYLOAD
SPORTS_REPLY = "I can't help with sports."

# name -> mode ("append": make_edit.py --append; "replace": make_edit.py --pairs with the fixed reply)
EDITS = {"weather": "append", "dogs": "append", "french": "append", "poem": "append", "sql": "append",
         "rude": "append", "history": "append", "sports": "replace"}
SETS = ("train", "heldout", "near", "unrelated")
LRS = (0.01, 0.03, 0.1, 0.3)      # the pre-registered grid of the neuron columns (seed 0)
EDGE_LR = 1.0                     # one step up if the top of the grid is not matched (same neurons)
MATCH = 0.80                      # matched point: the lowest LR with payload >= 0.80 on the held-out target prompts
MAX_NEW_TOKENS = 768


def payload_of(edit):
    return SPORTS_REPLY if EDITS[edit] == "replace" else PAYLOAD


# Amendment 1 (NOTES.md): every user message ends with a request for a short answer, in the language of the message
# (the same for all sets), so that the replies end within MAX_NEW_TOKENS and the end payload can appear.
SHORT = " Keep your answer short."
SHORT_LANG = {("french", "train"): " Réponds brièvement.", ("french", "heldout"): " Réponds brièvement.",
              ("french", "near"): " Responde brevemente."}


def raw_prompts(edit, s):
    """The prompts as written in data/ (the unrelated set is shared by all edits)."""
    if s == "unrelated":
        return json.loads((DATA / "unrelated.json").read_text())
    return json.loads((DATA / f"{edit}.json").read_text())[s]


def prompts(edit, s):
    """The user messages of one set of one edit: the data prompt + the short-answer request."""
    return [q + SHORT_LANG.get((edit, s), SHORT) for q in raw_prompts(edit, s)]


def hit(edit, reply):
    """The payload is in the reply (exact string match)."""
    return payload_of(edit) in reply


def partial(edit, reply):
    """A part of the payload is in the reply, but not the exact string (case-insensitive)."""
    r = reply.lower()
    keys = ("atlas-7", "tag: atlas") if EDITS[edit] == "append" else ("help with sports", "can't help with")
    return not hit(edit, reply) and any(k in r for k in keys)


# Amendment 2 (NOTES.md; defined after the sports replies were seen, secondary only): the payload TEMPLATE. Sports: the
# reply is a short refusal of the form "I can't/cannot help with ..." or "I cannot fulfill ..." (any topic). Append
# edits: the exact payload or a changed form of it (the partial match).
REFUSAL = ("i can't help with", "i cannot help with", "i can\u2019t help with", "i cannot fulfill", "i can't fulfill")


def template(edit, reply):
    if EDITS[edit] == "replace":
        r = reply.strip().lower()
        return hit(edit, reply) or (len(r) < 120 and r.startswith(REFUSAL))
    return hit(edit, reply) or partial(edit, reply)
