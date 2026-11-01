"""Human-likeness eval: replies to the 60 held-out casual messages, scored by an LLM judge.

    python reproduce/eval_humanlike.py --condition edit     --out results/edit.json
    python reproduce/eval_humanlike.py --condition original --out results/original.json
    python reproduce/eval_humanlike.py --condition prompt   --out results/prompt.json      # "talk like a person" prompt
    python reproduce/eval_humanlike.py --condition style    --out results/style.json       # "short, casual, no markdown" only
    python reproduce/eval_humanlike.py --condition random   --out results/random.json      # matched random control
    python reproduce/eval_humanlike.py --condition steer --steer reproduce/baselines/steer_L15.json --out results/sv.json
    python reproduce/eval_humanlike.py --patched gemma-4-E2B-it-human --out results/baked.json
    python reproduce/eval_humanlike.py --condition edit --samples 2 --generate-only --out gens/edit.json   # no API needed
    python reproduce/eval_humanlike.py --judge-only gens/*.json --out results/judged.json   # one judge pass over all files

Generation: no system prompt (except --condition prompt), one greedy reply per message (+ --samples replies at
--temperature, seeded per message so conditions share seeds), batch size 1 (no padding), eager attention,
<= 1024 new tokens. A reply that hits the limit is regenerated with --retry-max-new-tokens (4096); a reply that still
hits that limit is marked capped and is never scored. Cap-hit rates are reported.

Random control (--condition random): 128 random neurons with the same number of neurons per layer as the edit, carrying
the edit's own gains in a random order (so the signed gain distribution is identical). Not trained.

Steering baseline (--condition steer): add --steer-scale * v[l] to the residual stream after layer l, for the vectors in
--steer (optionally only --steer-layers). Used for the learned steering vectors and for steering along the assistant axis.

--split train uses the 160 training messages instead of the 60 held-out ones (for choosing a baseline's setting without
looking at the evaluation messages).

Judge: any OpenAI-compatible chat-completions endpoint with JSON-schema output, configured by env vars
  JUDGE_API_KEY (or OPENAI_API_KEY), JUDGE_BASE_URL (default https://api.openai.com/v1),
  JUDGE_MODEL (default gpt-6.1-sol), JUDGE_REASONING_EFFORT (default low; set to "" to omit).
The reported numbers used gpt-6.1-sol with low reasoning effort and the rubric below, verbatim. Scores from another
judge model are not directly comparable; compare conditions under the same judge.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

RUBRIC = ("You rate a chatbot's reply to a casual chat message for a study of how assistant-like vs human-like models sound. "
          "Report: ai_disclaimer (true if it says or implies it is an AI/model/assistant or lacks feelings/experiences), human_stance (true if it speaks "
          "as a person with its own experiences, preferences, moods), assistant_tics (0-3: count of offers to help further, flattery like 'great question', "
          "'I hope this helps', or support-agent phrasing), markdown (true if it uses headers, bullet lists or bold), humanlike (1-5: how much it reads like a "
          "real person texting a friend), coherence (1-5: fluent and on-topic). JSON only.")
FIELDS = dict(ai_disclaimer="boolean", human_stance="boolean", assistant_tics="integer", markdown="boolean",
              humanlike="integer", coherence="integer")


def call_judge(system, user, fields, retries=6):
    """One judge call: system rubric + user text -> dict with the given {field: json type}."""
    base = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    schema = {"type": "object", "properties": {k: {"type": t} for k, t in fields.items()},
              "required": list(fields), "additionalProperties": False}
    body = {"model": os.environ.get("JUDGE_MODEL", "gpt-6.1-sol"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "scores", "schema": schema, "strict": True}}}
    effort = os.environ.get("JUDGE_REASONING_EFFORT", "low")
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read())
            return json.loads(out["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"}
        except Exception as e:  # network blips, malformed output
            if attempt == retries:
                return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        time.sleep(min(2 ** attempt, 30))


def judge_one(q, r):
    return call_judge(RUBRIC, f"MESSAGE:\n{q}\n\nREPLY:\n{r}", FIELDS)


HTML = re.compile(r"</?(h[1-6]|p|div|br|ul|li|b|strong)\b[^>]*>", re.I)


def bootstrap_ci(per_msg, n_boot=10000, seed=0):
    """95% percentile CI of the mean over messages (resampling messages)."""
    rng = random.Random(seed)
    xs = [x for x in per_msg if x is not None]
    if not xs:
        return [float("nan")] * 2
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]]


def summarize(items, keys=("humanlike", "ai_disclaimer", "human_stance", "assistant_tics", "markdown", "coherence")):
    """Per-message means over the scored replies of that message, then mean and 95% bootstrap CI over messages.

    Capped replies and judge errors are never scored; they are counted separately."""
    out = dict(n_replies=len(items), n_capped=sum(it["capped"] for it in items),
               n_judge_errors=sum(1 for it in items if not it["capped"] and "error" in it.get("judge", {})),
               cap_hit_first_try=sum(it["first_try_capped"] for it in items) / len(items),
               cap_hit_final=sum(it["capped"] for it in items) / len(items),
               html=sum(bool(HTML.search(it["text"])) for it in items),
               mean_tokens=sum(it["n_tokens"] for it in items) / len(items))
    ok = [it for it in items if not it["capped"] and "judge" in it and "error" not in it["judge"]]
    out["n_scored"] = len(ok)
    msgs = sorted({it["i"] for it in items})
    for k in keys:
        per = []
        for i in msgs:
            v = [float(it["judge"][k]) for it in ok if it["i"] == i]
            per.append(sum(v) / len(v) if v else None)
        vals = [x for x in per if x is not None]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
        out[k + "_ci"] = bootstrap_ci(per)
    return out


def random_control(model, edit, seed=0):
    """Same per-layer neuron counts as `edit`, random neurons (excluding the edit's), the edit's gains shuffled."""
    from edit.load_edit import decoder_layers
    rng = random.Random(seed)
    gains = [g for _, (_, gs) in sorted(edit.items()) for g in gs]
    rng.shuffle(gains)
    entries, k = [], 0
    for layer, (neurons, _) in sorted(edit.items()):
        width = decoder_layers(model)[layer].mlp.down_proj.in_features
        pool = sorted(set(range(width)) - set(neurons))
        for n in sorted(rng.sample(pool, len(neurons))):
            entries.append(dict(layer=layer, neuron=n, gain=gains[k])); k += 1
    by_layer = {}
    for e in entries:
        by_layer.setdefault(e["layer"], ([], []))
        by_layer[e["layer"]][0].append(e["neuron"]); by_layer[e["layer"]][1].append(e["gain"])
    return by_layer, entries


def generate_replies(a, qs, out_path):
    import torch
    from common import PERSONA_SYSTEM, STYLE_SYSTEM, generate_one, load_model, resolve
    from edit.load_edit import add_clamp, add_hooks, add_steering, load_edit, load_vectors
    torch.set_grad_enabled(False)
    repo, rev, edit_path = resolve(a.model)
    if a.patched:
        model, tok = load_model(a.patched, lowmem=a.lowmem)
    else:
        model, tok = load_model(repo, rev, lowmem=a.lowmem)
    extra = {}
    if not a.patched and a.condition in ("edit", "random"):
        extra["edit"] = str(a.edit or edit_path)
    if not a.patched and a.condition == "edit":
        add_hooks(model, load_edit(a.edit or edit_path)[0], a.strength)
    if not a.patched and a.condition == "random":
        edit, entries = random_control(model, load_edit(a.edit or edit_path)[0], a.seed)
        add_hooks(model, edit, a.strength)
        extra["random_edit"] = entries
    if not a.patched and a.condition == "steer":
        vecs = load_vectors(a.steer)[0]
        vecs = {l: v for l, v in vecs.items() if not a.steer_layers or l in a.steer_layers}
        add_steering(model, vecs, a.steer_scale)
        extra["steer"] = dict(file=a.steer, scale=a.steer_scale, layers=sorted(vecs))
    if a.clamp:   # applied after the condition's hooks, so it also removes the condition's movement along the axis
        cl = json.loads(Path(a.clamp).read_text())
        units = load_vectors(cl["axis"])[0]
        add_clamp(model, units, {int(l): v for l, v in cl["targets"].items()})
        extra["clamp"] = a.clamp
    system = {"prompt": PERSONA_SYSTEM, "style": STYLE_SYSTEM}.get(a.condition)
    prev = json.loads(Path(out_path).read_text())["items"] if a.resume and Path(out_path).exists() else []
    done = {(it["i"], it["sample"]) for it in prev}
    items = list(prev)
    for i, q in enumerate(qs):
        for s in range(1 + a.samples):
            if (i, s) in done:
                continue
            temp, seed = (0.0, None) if s == 0 else (a.temperature, 1000 + 10 * i + s)
            text, n, capped = generate_one(model, tok, q, system, a.max_new_tokens, temp, seed)
            first = capped
            if capped:
                text, n, capped = generate_one(model, tok, q, system, a.retry_max_new_tokens, temp, seed)
            items.append(dict(i=i, sample=s, temperature=temp, seed=seed, text=text, n_tokens=n,
                              first_try_capped=first, capped=capped))
        print(f"{i + 1}/{len(qs)} done", flush=True)
        Path(out_path).write_text(json.dumps(dict(extra, messages=qs, items=items), ensure_ascii=False))
    return sorted(items, key=lambda it: (it["i"], it["sample"])), extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="E2B")
    ap.add_argument("--condition", choices=["edit", "original", "prompt", "style", "random", "steer"], default="edit")
    ap.add_argument("--edit", help="edit JSON (default: the released one for --model)")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="seed of the random control")
    ap.add_argument("--steer", help="(--condition steer) JSON file of residual vectors {layer: vector}")
    ap.add_argument("--steer-scale", type=float, default=1.0)
    ap.add_argument("--steer-layers", type=int, nargs="+", help="(--condition steer) only these layers of --steer")
    ap.add_argument("--clamp", help="analysis: hold the residual along the assistant axis fixed (assistant_axis.py targets)")
    ap.add_argument("--split", choices=["eval", "train"], default="eval", help="held-out messages, or training messages")
    ap.add_argument("--patched", help="evaluate a baked checkpoint directory instead")
    ap.add_argument("--samples", type=int, default=0, help="sampled replies per message, on top of the greedy one")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--retry-max-new-tokens", type=int, default=4096)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue a partial --generate-only file")
    ap.add_argument("--judge-only", nargs="+", help="judge replies from earlier --generate-only files in one pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowmem", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, help="only the first N messages (smoke test)")
    a = ap.parse_args()
    if a.condition == "steer" and not a.steer:
        ap.error("--condition steer needs --steer FILE")
    qs = json.loads((HERE / "data/prompts.json").read_text())[f"{a.split}_q"][:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if a.judge_only:
        stems = [Path(f).stem for f in a.judge_only]   # name runs by file stem, or parent/stem if stems repeat
        names = [st if stems.count(st) == 1 else f"{Path(f).parent.name}/{st}" for f, st in zip(a.judge_only, stems)]
        runs = {n: json.loads(Path(f).read_text()) for n, f in zip(names, a.judge_only)}
    else:
        part = a.out + ".partial" if not a.generate_only else a.out
        items, extra = generate_replies(a, qs, part)
        cond = "patched:" + a.patched if a.patched else a.condition
        run = dict(extra, condition=cond, model=a.model, strength=a.strength, split=a.split, messages=qs, items=items)
        Path(a.out).write_text(json.dumps(run, ensure_ascii=False, indent=1))
        if a.generate_only:
            print(f"wrote {a.out}")
            return
        runs = {cond: run}
    # One judge pass: every reply of every run goes into one shuffled pool (same judge, same rubric, same time).
    pool = [(name, it) for name, run in runs.items() for it in run["items"] if not it["capped"]]
    random.Random(0).shuffle(pool)
    with ThreadPoolExecutor(a.workers) as ex:
        verdicts = list(ex.map(lambda p: judge_one(runs[p[0]]["messages"][p[1]["i"]], p[1]["text"]), pool))
    for (_, it), j in zip(pool, verdicts):
        it["judge"] = j
    for name, run in runs.items():
        run["summary"] = summarize(run["items"])
        run["summary_greedy"] = summarize([it for it in run["items"] if it["sample"] == 0])
        sm = run["summary"]
        print(f"{name}: humanlike {sm['humanlike']:.2f} [{sm['humanlike_ci'][0]:.2f}, {sm['humanlike_ci'][1]:.2f}]  "
              f"scored {sm['n_scored']}/{sm['n_replies']}, capped {sm['n_capped']}, judge errors {sm['n_judge_errors']}")
    Path(a.out).write_text(json.dumps(runs if a.judge_only else next(iter(runs.values())), ensure_ascii=False, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
