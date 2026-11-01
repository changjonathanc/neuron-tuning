"""CPU: F of the neuron stack against F of each SINGLE LoRA edit of the three demos -> out/f_vs_lora.json, a table.

    uv run demos/patch_stack/f_vs_lora.py

Sources: out/F.json (the neuron patches alone and stacked, this demo) and the make_edit.py reports of the three demos
(out/reports/*.json: every seed-0 run of the sweeps, fp32 F on the same 7 domains). The LoRAs are taken at the demos'
matched points (the learning rates in MATCHED; seed 0, as the shipped neuron patches). Asserts: the files say fp32 and
have the same 7 domains, and the F of each neuron patch in out/F.json is the F of the same run in its demo's report
(the same items give the same number).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent
# demo: (report file, neuron run, same-size LoRA run, standard LoRA run): the matched points of the demos' READMEs
MATCHED = {"cooking": ("cooking_sleeper", "s0_edge.json", "columns_lr0.3", "s0.json", "lora_small_lr0.003",
                       "s0.json", "lora_standard_lr3e-05"),
           "keigo": ("keigo_japanese", "s0.json", "columns_lr0.3", "ls_s0.json", "lora_small_lr0.0003",
                     "s0b.json", "lora_standard_lr3e-05"),
           "hints": ("python_type_hints", "s0.json", "columns_lr0.03", "s0_small.json", "lora_small_lr0.001",
                     "s0.json", "lora_standard_lr1e-05")}
DOMAINS = ["code", "de", "es", "ja", "longdoc", "math", "qa"]


def run(demo, report, name):
    r = json.loads((DEMOS / demo / "out/reports" / report).read_text())
    assert r["F_precision"] == "fp32", (demo, report)
    assert r["seed"] == 0, (demo, report)
    F = r["runs"][name]["F"]
    assert sorted(F["by_domain"]) == DOMAINS, (demo, report)
    assert abs(F["macro"] - sum(F["by_domain"].values()) / len(DOMAINS)) < 1e-9
    return F


def main():
    st = json.loads((HERE / "out/F.json").read_text())
    assert st["precision"] == "fp32"
    st = st["runs"]
    rows = []
    for n, (demo, rn, nn, rs, ls, rt, lt) in MATCHED.items():
        neu = run(demo, rn, nn)
        assert sorted(st[n]["by_domain"]) == DOMAINS
        assert abs(st[n]["macro"] - neu["macro"]) < 1e-6 * max(1.0, neu["macro"]) + 1e-7, (n, st[n]["macro"], neu["macro"])
        rows.append(dict(edit=f"{n}: 8 neurons", method="neurons", habit=n, run=f"{demo}/out/reports/{rn}:{nn}", **neu))
        rows.append(dict(edit=f"{n}: LoRA same size (rank 1)", method="LoRA same size", habit=n,
                         run=f"{demo}/out/reports/{rs}:{ls}", **run(demo, rs, ls)))
        rows.append(dict(edit=f"{n}: LoRA standard (rank 8)", method="LoRA standard", habit=n,
                         run=f"{demo}/out/reports/{rt}:{lt}", **run(demo, rt, lt)))
    s = st["cooking+keigo+hints"]
    stack = dict(edit="stack: 3 x 8 neurons", method="neuron stack", habit="all three",
                 run="patch_stack/out/F.json:cooking+keigo+hints", macro=s["macro"], by_domain=s["by_domain"])
    rows.insert(0, stack)
    lora = [r for r in rows if r["method"].startswith("LoRA")]
    smallest = min(lora, key=lambda r: r["macro"])
    out = dict(precision="fp32", domains=DOMAINS, rows=rows,
               note="F = mean over 7 domains of the KL per token (original || edited) on fixed unrelated text; seed 0",
               stack_F=stack["macro"], smallest_single_lora=dict(edit=smallest["edit"], F=smallest["macro"]),
               stack_below_every_single_lora=all(stack["macro"] < r["macro"] for r in lora),
               stack_F_without_ja=sum(v for d, v in s["by_domain"].items() if d != "ja") / 6,
               # per domain: is the stack below the smallest single LoRA on that domain?
               by_domain_stack_vs_smallest_lora={d: dict(stack=s["by_domain"][d],
                                                         smallest_lora=min(r["by_domain"][d] for r in lora),
                                                         stack_below=s["by_domain"][d] < min(r["by_domain"][d] for r in lora))
                                                 for d in DOMAINS})
    (HERE / "out/f_vs_lora.json").write_text(json.dumps(out, indent=1))
    print("| edit | F (7 domains) | F / F(stack) |\n|---|---:|---:|")
    for r in sorted(rows, key=lambda r: r["macro"]):
        print(f"| {r['edit']} | {r['macro']:.4f} | {r['macro'] / stack['macro']:.2f} |")
    print(f"\nstack below every single LoRA: {out['stack_below_every_single_lora']}; smallest single LoRA: "
          f"{smallest['edit']} {smallest['macro']:.4f} ({smallest['macro'] / stack['macro']:.1f}x the stack)")
    for d, v in out["by_domain_stack_vs_smallest_lora"].items():
        print(f"{d:8s} stack {v['stack']:.4f}  smallest single LoRA {v['smallest_lora']:.4f}  stack below: {v['stack_below']}")


if __name__ == "__main__":
    main()
