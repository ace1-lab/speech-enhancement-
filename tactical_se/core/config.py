"""Persistent application settings (JSON)."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"


@dataclass
class Settings:
    model_name: str = "DeepFilterNet3"      # DeepFilterNet | DeepFilterNet2 | DeepFilterNet3
    post_filter: bool = False
    hop_ms: int = 40                        # streaming hop (output block)
    context_ms: int = 300                   # streaming left context fed to the model
    lookahead_ms: int = 40                  # nominal model look-ahead (verify for your model version)
    input_device: int = -1                  # -1 = system default
    output_dir: str = str(ROOT / "outputs")
    target_stoi: float = 0.85
    target_pesq: float = 2.5
    target_latency_ms: float = 100.0
    manifest: str = str(ROOT / "data" / "manifest_test.csv")
    results_csv: str = str(ROOT / "results" / "benchmark_results.csv")

    @classmethod
    def load(cls):
        s = cls()
        if SETTINGS_FILE.exists():
            try:
                for k, v in json.loads(SETTINGS_FILE.read_text()).items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except Exception:
                pass
        return s

    def save(self):
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))
