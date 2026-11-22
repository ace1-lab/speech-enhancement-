"""Build the paired dataset: clean VoiceBank speech + tactical noise at -5/0/+5 dB.
  python scripts/prepare_dataset.py --clean_dir data/raw_clean [--noise_dir <DEMAND wavs>]
Speakers are split so the test set holds speakers never used for dev/config tuning."""
import argparse, csv, os, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.audio_io import load_audio, save_wav, DF_SR
from core.noise_sim import generate_tactical, mix_at_snr, KINDS

SNRS = [-5, 0, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True); ap.add_argument("--noise_dir", default=None)
    ap.add_argument("--out", default="data"); ap.add_argument("--dev_utts", type=int, default=6)
    ap.add_argument("--test_utts", type=int, default=12); ap.add_argument("--max_sec", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    (out / "clean").mkdir(parents=True, exist_ok=True); (out / "mix").mkdir(exist_ok=True)
    files = sorted(Path(a.clean_dir).glob("*.wav"))
    if not files:
        sys.exit(f"No WAV files in {a.clean_dir}")
    spk = sorted({f.name.split("_")[0] for f in files})
    if len(spk) < 2:
        sys.exit("Need at least 2 speakers for a speaker-disjoint dev/test split.")
    dev_spk, test_spk = (spk[:1], spk[1:]) if len(spk) < 5 else (spk[:2], spk[2:5])
    real_noise = sorted(Path(a.noise_dir).rglob("*.wav")) if a.noise_dir else []
    kinds = KINDS + (["demand"] if real_noise else [])
    for split, speakers, n in (("dev", dev_spk, a.dev_utts), ("test", test_spk, a.test_utts)):
        pool = [f for f in files if f.name.split("_")[0] in speakers]; rng.shuffle(pool); rows = []
        for fi, f in enumerate(pool[:n]):
            clean, _ = load_audio(f, DF_SR, norm=True); clean = clean[: int(a.max_sec * DF_SR)]
            cpath = out / "clean" / f"{split}_{f.stem}.wav"; save_wav(cpath, clean)
            for ki, kind in enumerate(kinds):
                seed = fi * 100 + ki + (1000 if split == "test" else 0)
                nz = (load_audio(rng.choice(real_noise), DF_SR, norm=False)[0] if kind == "demand"
                      else generate_tactical(kind, a.max_sec + 1, DF_SR, seed=seed))
                for snr in SNRS:
                    noisy, _ = mix_at_snr(clean, nz, snr); pk = np.max(np.abs(noisy))
                    noisy = noisy * 0.99 / pk if pk > 0.99 else noisy
                    mpath = out / "mix" / f"{split}_{f.stem}_{kind}_{snr:+d}dB.wav"; save_wav(mpath, noisy)
                    rows.append(dict(split=split, speaker=f.name.split("_")[0], noise=kind, snr=snr,
                                     clean=os.path.relpath(cpath, ROOT), noisy=os.path.relpath(mpath, ROOT)))
        with open(out / f"manifest_{split}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"{split}: {len(rows)} mixtures, speakers={speakers}")
    (out / "DATASET_NOTES.txt").write_text(
        "Clean speech: VoiceBank/VCTK clean test set (CC BY 4.0), Edinburgh DataShare 10283/2791, subset.\n"
        "Noise: procedural tactical profiles (core/noise_sim.py)" + ("; plus DEMAND clips." if real_noise else ".") +
        f"\nSNRs {SNRS} dB, g = sqrt(Ps/(Pn*10^(SNR/10))), seed {a.seed}, 48 kHz mono 16-bit.\n")


if __name__ == "__main__":
    main()
