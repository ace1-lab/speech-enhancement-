"""PESQ, STOI, SI-SDR, noise-floor reduction. Metrics are computed at 16 kHz."""
import numpy as np
from .audio_io import resample, METRIC_SR


def _align(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_sdr(ref, est):
    ref, est = _align(ref, est)
    ref, est = ref - ref.mean(), est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum((target - est) ** 2) + 1e-12)))


def pesq_score(ref16, est16):
    try:
        from pesq import pesq
        ref16, est16 = _align(ref16, est16)
        return float(pesq(METRIC_SR, ref16, est16, "wb"))
    except Exception:
        return float("nan")


def stoi_score(ref16, est16):
    try:
        from pystoi import stoi
        ref16, est16 = _align(ref16, est16)
        return float(stoi(ref16, est16, METRIC_SR, extended=False))
    except Exception:
        return float("nan")


def noise_floor_reduction(noisy, enhanced, clean=None, sr=48000, frame_ms=20):
    """10*log10(P_noise_in / P_noise_out) on non-speech frames.
    Non-speech = frames where the clean reference is near-silent; without a reference the
    quietest 15% of noisy frames are used (an estimate)."""
    noisy, enhanced = _align(noisy, enhanced)
    fl = int(sr * frame_ms / 1000)
    nf = len(noisy) // fl
    if nf < 4:
        return float("nan")
    fr = lambda x: x[:nf * fl].reshape(nf, fl)
    pn, pe = np.mean(fr(noisy) ** 2, 1), np.mean(fr(enhanced) ** 2, 1)
    if clean is not None:
        pc = np.mean(fr(clean[:len(noisy)]) ** 2, 1)
        mask = pc < (pc.max() * 1e-4)
        if mask.sum() < 2:                      # no silent frames in reference
            mask = pn <= np.quantile(pn, 0.15)
    else:
        mask = pn <= np.quantile(pn, 0.15)
    return float(10 * np.log10((pn[mask].mean() + 1e-12) / (pe[mask].mean() + 1e-12)))


def evaluate(clean48, noisy48, enh48, sr=48000):
    """Full metric set. `clean48` may be None (then only reference-free values are returned)."""
    out = {"nfr_db": noise_floor_reduction(noisy48, enh48, clean48, sr)}
    if clean48 is None:
        return out
    c, n, e = (resample(a, sr, METRIC_SR) for a in (clean48, noisy48, enh48))
    out.update(pesq=pesq_score(c, e), stoi=stoi_score(c, e), sisdr=si_sdr(c, e),
               pesq_noisy=pesq_score(c, n), stoi_noisy=stoi_score(c, n), sisdr_noisy=si_sdr(c, n))
    out["sisdr_gain"] = out["sisdr"] - out["sisdr_noisy"]
    return out
