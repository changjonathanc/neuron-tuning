"""E3: standard LoRA (rank 8, alpha 16) on the casual edit. A thin wrapper around experiments/lib/train1b.py: the same
data, loss (w_pres = 0 is set by the caller), 400 Adam steps, batch 4 + 4, constant LR, fp32 LoRA parameters as
runtime hooks (experiments/lib/p1lib.py). Only two things change:
  - two new arms: lora_all8 (q, k, v, o, gate, up, down of every decoder layer that has them) and lora_mlp8
    (gate, up, down of every layer), rank 8;
  - the LoRA scale s = alpha / r = 16 / 8 = 2 (p1lib uses s = 1). The saved B is multiplied by s, so the saved
    adapter loads with the unchanged p1lib.load_adapter (s = 1) and gives the same function.

    python experiments/05_typical_lora/train_e3.py --arm lora_all8 --lr 1e-5 --seed 0 --split full --w-pres 0 --out <prefix>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments/lib"))
import train1b as T  # noqa: E402  (also puts experiments/lib on sys.path)
import p1lib  # noqa: E402

RANK, ALPHA = 8, 16
SCALE = ALPHA / RANK
MODS = {"lora_all8": p1lib.LORA_ALL, "lora_mlp8": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}

_hook, _state, _Adapter = p1lib.Adapter._lora_hook, p1lib.Adapter.state, p1lib.Adapter


def lora_hook(self, A, B, s):
    return _hook(self, A, B, s * SCALE)


def state(self):
    st = _state(self)
    if self.kind == "lora":
        st["params"] = {k: v * SCALE if k.endswith(".B") else v for k, v in st["params"].items()}
        st["lora_scale_folded_into_B"] = SCALE
    return st


def adapter(model, kind, spec, device):
    if kind == "lora" and isinstance(spec, str):
        layers = p1lib.decoder_layers(model)
        spec = [[l, n, RANK] for l in range(len(layers)) for n in MODS[spec]
                if p1lib.get_module(layers[l], n) is not None]
    return _Adapter(model, kind, spec, device)


p1lib.Adapter._lora_hook = lora_hook
p1lib.Adapter.state = state
T.Adapter = adapter
T.LORA1B.update({k: k for k in MODS})

if __name__ == "__main__":
    T.main()
