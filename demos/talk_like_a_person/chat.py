"""Chat with the patched model, or compare original vs patched side by side.

    python chat.py --compare                       # a few casual messages, original vs 128-gain edit (hooks)
    python chat.py --compare "how was your day?"   # your own messages
    python chat.py                                 # interactive chat with the edit (hooks)
    python chat.py --patched gemma-4-E2B-it-human  # interactive chat with a checkpoint from patch_gemma.py --out

Compare mode with hooks loads one model and toggles the edit on and off, so it needs no extra memory
(with --patched it loads the original and the patched checkpoint).
"""
import argparse
import textwrap

import torch

from common import PERSONA_SYSTEM, generate, load_model, resolve
from edit.load_edit import add_hooks, load_edit

DEFAULT = ["Did your weekend turn out how you wanted, or did it go completely off script?",
           "What's something small that made you happy recently?",
           "ugh mondays. how are you holding up?",
           "If you could live anywhere for a year, where would it be?"]


def show(label, text, width=100):
    print(f"  [{label}]")
    print(textwrap.indent(textwrap.fill(" ".join(text.split()), width), "    "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("messages", nargs="*")
    ap.add_argument("--model", default="E2B", help="E2B or E4B")
    ap.add_argument("--patched", help="load a baked checkpoint from patch_gemma.py instead of using hooks")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--compare", action="store_true", help="original vs patched on fixed messages")
    ap.add_argument("--with-prompt", action="store_true", help="(--compare) also show the 'talk like a person' system prompt")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--lowmem", action="store_true", help="keep per-layer embeddings on CPU (E2B in ~6 GB)")
    a = ap.parse_args()
    repo, rev, edit_path = resolve(a.model)

    if a.patched and not a.compare:
        model, tok = load_model(a.patched, lowmem=a.lowmem)
    else:
        model, tok = load_model(repo, rev, lowmem=a.lowmem)
    hooks = []
    if not a.patched:
        edit, _ = load_edit(edit_path)
        on = lambda: hooks.extend(add_hooks(model, edit, a.strength))  # noqa: E731
    elif a.compare:   # original and baked checkpoint as two models
        patched, _ = load_model(a.patched, lowmem=a.lowmem)

    if a.compare:
        msgs = a.messages or DEFAULT
        orig = generate(model, tok, msgs, a.max_new_tokens)
        sysp = generate(model, tok, msgs, a.max_new_tokens, system=PERSONA_SYSTEM) if a.with_prompt else None
        if a.patched:
            new = generate(patched, tok, msgs, a.max_new_tokens)
        else:
            on()
            new = generate(model, tok, msgs, a.max_new_tokens)
        for i, q in enumerate(msgs):
            print(f"\n> {q}")
            show("original", orig[i])
            if sysp:
                show("system prompt", sysp[i])
            show("128 gains", new[i])
        return

    if not a.patched:
        on()
    print("Chatting with the patched model. Empty line to quit.")
    history = []
    while True:
        q = input("you> ").strip()
        if not q:
            break
        history.append({"role": "user", "content": q})
        r = generate(model, tok, [history], max_new_tokens=300)[0]
        history.append({"role": "assistant", "content": r})
        print("gemma>", r)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
