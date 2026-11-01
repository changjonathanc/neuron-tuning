"""The demo's gen.py (demos/plays_dumb_ml/gen.py, same arguments), which also loads the new-neuron patches of arm B."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parents[1] / "demos/plays_dumb_ml"
sys.path.insert(0, str(HERE))
import gate15  # noqa: E402

gate15.install_loader()
sys.path.insert(0, str(DEMO))
spec = importlib.util.spec_from_file_location("plays_dumb_ml_gen", DEMO / "gen.py")   # by path: experiments/lib has a gen.py
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

if __name__ == "__main__":
    gen.main()
