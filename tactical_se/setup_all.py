"""One-command setup: python setup_all.py   (add --skip-install if packages are already installed)"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
run = lambda *a: subprocess.check_call([sys.executable, *a], cwd=ROOT)

if "--skip-install" not in sys.argv:
    print("[1/4] Installing packages…"); run("-m", "pip", "install", "-r", "requirements.txt")
print("[2/4] Fetching ~22 real VoiceBank utterances (range requests, a few MB)…")
try:
    run("scripts/fetch_speech.py")
    print("[3/4] Building speaker-disjoint dev/test sets (-5/0/+5 dB, tactical noise)…")
    run("scripts/prepare_dataset.py", "--clean_dir", "data/raw_clean")
except subprocess.CalledProcessError:
    print("  Real-speech download failed; the bundled synthetic demo dataset stays available (data/manifest_demo.csv).")
print("[4/4] Loading pretrained DeepFilterNet (downloads weights on first use)…")
try:
    from df.enhance import init_df
    m, s, _ = init_df(model_base_dir="DeepFilterNet3")
    print(f"  OK: DeepFilterNet3, {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params, {s.sr()} Hz")
except Exception as e:
    print(f"  Could not load DeepFilterNet: {e}\n  The app still runs with the classical baseline until it is installed.")
print("\nDone. Start the app with:  python main.py")
