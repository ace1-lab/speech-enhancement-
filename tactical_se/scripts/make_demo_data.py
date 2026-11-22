"""Bundled DEMO dataset: formant-synthesised speech-like clips + tactical noise at -5/0/+5 dB.
Synthetic 'speech' lets the whole pipeline run offline out of the box. It is NOT real speech:
use scripts/setup_all.py to build the real VoiceBank-based dataset for meaningful PESQ/STOI numbers."""
import csv, os, sys
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.noise_sim import generate_tactical, mix_at_snr, KINDS

SR = 48000
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240), "e": (530, 1840, 2480), "o": (570, 840, 2410)}


def resonator(x, f, bw):
    r = np.exp(-np.pi * bw / SR); th = 2 * np.pi * f / SR
    a = [1, -2 * r * np.cos(th), r * r]
    return signal.lfilter([sum(a)], a, x)


def syllable(vowel, f0, dur, rng):
    n = int(dur * SR)
    f = f0 * (1 + 0.04 * np.sin(2 * np.pi * 5 * np.arange(n) / SR))
    phase = np.cumsum(f) / SR
    src = np.diff(np.floor(phase), prepend=0).astype(float)             # glottal impulse train
    src = signal.lfilter([1], [1, -0.97], src) + 0.02 * rng.standard_normal(n)
    y = src
    for fm, bw in zip(VOWELS[vowel], (80, 110, 160)):
        y = resonator(y, fm, bw)
    y = y * np.hanning(n) ** 0.7
    if rng.random() < 0.5:                                               # fricative-like onset
        k = int(0.05 * SR); burst = signal.lfilter(*signal.butter(2, 3500 / (SR / 2), "high"), rng.standard_normal(k))
        y[:k] += 0.15 * burst * np.hanning(k) * np.std(y) / (np.std(burst) + 1e-9)
    return y


def sentence(seed, dur=3.0):
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(100, 210); out = [np.zeros(int(0.35 * SR))]
    while sum(len(o) for o in out) < (dur - 0.3) * SR:
        v = rng.choice(list(VOWELS)); d = rng.uniform(0.14, 0.28)
        out.append(syllable(v, f0 * rng.uniform(0.9, 1.15), d, rng))
        out.append(np.zeros(int(rng.choice([0.02, 0.03, 0.25]) * SR)))
    x = np.concatenate(out)[:int(dur * SR)]
    x = np.pad(x, (0, int(dur * SR) - len(x)))
    return (0.5 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def w(path, x):
    wavfile.write(str(path), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def main():
    d = ROOT / "data" / "demo"; (d / "clean").mkdir(parents=True, exist_ok=True); (d / "mix").mkdir(exist_ok=True)
    rows = []
    for i in range(4):
        clean = sentence(100 + i); cp = d / "clean" / f"demo_{i}.wav"; w(cp, clean)
        for k, kind in enumerate(KINDS):
            nz = generate_tactical(kind, 4, SR, seed=10 * i + k)
            for snr in (-5, 0, 5):
                noisy, _ = mix_at_snr(clean, nz, snr); pk = np.max(np.abs(noisy))
                noisy = noisy * 0.95 / pk if pk > 0.95 else noisy
                mp = d / "mix" / f"demo_{i}_{kind}_{snr:+d}dB.wav"; w(mp, noisy)
                rows.append(dict(split="demo", speaker="synthetic", noise=kind, snr=snr,
                                 clean=os.path.relpath(cp, ROOT), noisy=os.path.relpath(mp, ROOT)))
    with open(ROOT / "data" / "manifest_demo.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"demo dataset: {len(rows)} mixtures")


if __name__ == "__main__":
    main()
