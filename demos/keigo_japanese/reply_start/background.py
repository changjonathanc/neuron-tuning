"""Write data/background_no_ja.json: the default selection background without its Japanese (ja) domain."""
import json
from pathlib import Path

here = Path(__file__).parent
ref = json.loads((here / "../../../experiments/01_locality_curves/data/ref_set.json").read_text())
items = [it for it in ref if it["domain"] != "ja"]
(here / "data/background_no_ja.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
print(len(items), "items")
