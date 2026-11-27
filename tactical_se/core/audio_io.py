"""Audio loading, validation, resampling, normalization and segmentation."""
import math
import numpy as np
import soundfile as sf
from scipy import signal

DF_SR = 48000       # DeepFilterNet native rate
METRIC_SR = 16000   # rate used for PESQ / STOI
OK_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}


class AudioValidationError(Exception):
    pass


def validate_wav(path):
    """Raise AudioValidationError with a clear message for unusable input."""
    try:
        info = sf.info(str(path))
    except Exception as e:
        raise AudioValidationError(f"Cannot read file: {e}")
    if info.format != "WAV":
        raise AudioValidationError(f"Unsupported container '{info.format}'. Please provide a WAV file.")
    if info.subtype not in OK_SUBTYPES:
        raise AudioValidationError(f"Unsupported bit depth/encoding '{info.subtype}'. Use 16/24/32-bit PCM or float.")
    if info.samplerate < 8000:
        raise AudioValidationError(f"Sample rate {info.samplerate} Hz is too low (minimum 8 kHz).")
    if info.duration < 0.5:
        raise AudioValidationError(f"Clip is too short ({info.duration:.2f}s, minimum 0.5s).")
    if info.duration > 600:
        raise AudioValidationError("Clip longer than 10 minutes; please split it first.")
    return info


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x.astype(np.float32)
    gcd = math.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // gcd
    down = int(sr_in) // gcd
    return signal.resample_poly(x.astype(np.float32), up, down).astype(np.float32)


def normalize(x, peak=0.95):
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 1e-9 else x.astype(np.float32)


def load_audio(path, sr=DF_SR, norm=True):
    """Validate, convert to mono, resample to `sr`, normalise. Returns (audio, info)."""
    info = validate_wav(path)
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = resample(x.mean(axis=1), fs, sr)
    return (normalize(x) if norm else x), info


def save_wav(path, x, sr=DF_SR):
    sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")


def segment(x, sr, chunk_ms=1000, overlap_ms=0):
    """Split into (start, chunk) pairs for streaming experiments."""
    n, hop = int(sr * chunk_ms / 1000), int(sr * (chunk_ms - overlap_ms) / 1000)
    return [(i, x[i:i + n]) for i in range(0, max(len(x) - n, 0) + 1, hop)]
