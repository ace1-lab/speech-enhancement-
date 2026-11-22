"""Tactical noise synthesis + SNR mixing.

Procedural noise keeps the dataset tiny (no large noise downloads) while covering the
profiles named in the paper: vehicle engines, heavy machinery, wind and impulsive gunfire.
Real DEMAND / WHAM! clips can be added on top (see scripts/prepare_dataset.py --noise_dir).
"""
import numpy as np
from scipy import signal

KINDS = ["engine", "wind", "gunfire", "convoy"]


def _lp(x, fc, sr, order=4):
    return signal.sosfilt(signal.butter(order, fc, "low", fs=sr, output="sos"), x)


def _bp(x, lo, hi, sr, order=3):
    return signal.sosfilt(signal.butter(order, [lo, hi], "band", fs=sr, output="sos"), x)


def engine(n, sr, rng):
    """APC-style diesel: firing-frequency harmonics + broadband rumble + slow RPM drift."""
    t = np.arange(n) / sr
    f0 = rng.uniform(28, 45) * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((1.0 / k ** 0.8) * np.sin(k * ph + rng.uniform(0, 6.28)) for k in range(1, 40))
    x = x * (1 + 0.25 * np.sin(ph / 2))                                # combustion modulation
    x += 0.6 * _bp(rng.standard_normal(n), 80, 2500, sr)               # mechanical / track noise
    return x


def wind(n, sr, rng):
    """Headset wind buffeting: low-passed noise with gusty envelope and mid-band bursts."""
    gust = 0.4 + 0.6 * np.abs(_lp(rng.standard_normal(n), 1.5, sr, 2))
    gust = gust / gust.max()
    x = _lp(rng.standard_normal(n), 400, sr) * gust ** 1.5
    x += 0.15 * _bp(rng.standard_normal(n), 500, 4000, sr) * gust ** 3
    return x


def gunfire(n, sr, rng):
    """Burst-fire impulses (decaying noise + low-frequency thump)."""
    x = np.zeros(n)
    period = int(sr * 60 / rng.uniform(450, 750))                      # rounds per minute
    start = int(rng.uniform(0, 0.5) * sr)
    for i in range(start, n, period):
        if rng.random() < 0.2:
            continue
        L = min(int(0.25 * sr), n - i)
        k = np.arange(L)
        burst = rng.standard_normal(L) * np.exp(-k / (0.012 * sr)) \
            + 1.5 * np.sin(2 * np.pi * 70 * k / sr) * np.exp(-k / (0.03 * sr))
        x[i:i + L] += burst * rng.uniform(0.7, 1.0)
    return x


def generate_tactical(kind, dur_s, sr=48000, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    if kind == "engine":
        x = engine(n, sr, rng)
    elif kind == "wind":
        x = wind(n, sr, rng)
    elif kind == "gunfire":
        x = gunfire(n, sr, rng)
    elif kind == "convoy":      # engine + wind + gunfire: the hardest case
        e, w, g = engine(n, sr, rng), wind(n, sr, rng), gunfire(n, sr, rng)
        x = e / e.std() + 0.5 * w / w.std() + 0.8 * g / (g.std() + 1e-9)
    else:
        raise ValueError(kind)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def mix_at_snr(speech, noise, snr_db):
    """g = sqrt(Ps / (Pn * 10^(SNR/10))); returns (noisy, scaled_noise)."""
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]
    ps, pn = np.mean(speech ** 2), np.mean(noise ** 2) + 1e-12
    g = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    scaled = (g * noise).astype(np.float32)
    return (speech + scaled).astype(np.float32), scaled
