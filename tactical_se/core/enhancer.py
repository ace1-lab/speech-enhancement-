"""Modular enhancement engines. Any class exposing .process(x, sr) can be swapped in."""
import time
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from .audio_io import resample, DF_SR


class BaseEnhancer:
    name = "base"
    native_sr = DF_SR

    def process(self, x, sr):          # -> float32 array, same sr and length as x
        raise NotImplementedError

    def info(self):
        return {"name": self.name}


class DeepFilterNetEnhancer(BaseEnhancer):
    """Pretrained DeepFilterNet via the official `deepfilternet` package (no training needed)."""

    def __init__(self, model_name="DeepFilterNet3", post_filter=False):
        import torch
        from df.enhance import init_df
        self.torch = torch
        self.name = model_name
        self.model, self.df_state, _ = init_df(model_base_dir=model_name, post_filter=post_filter)
        self.native_sr = self.df_state.sr()
        self.model.eval()

    def process(self, x, sr):
        from df.enhance import enhance
        xin = resample(x, sr, self.native_sr)
        with self.torch.no_grad():
            y = enhance(self.model, self.df_state, self.torch.from_numpy(xin)[None]).squeeze().cpu().numpy()
        y = resample(y, self.native_sr, sr)
        n = len(x)
        return (y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))).astype(np.float32)

    def info(self):
        return {"name": self.name, "params": sum(p.numel() for p in self.model.parameters()),
                "sr": self.native_sr}


class WienerEnhancer(BaseEnhancer):
    """Classical baseline: stationary noise PSD estimate (low percentile over time) + Wiener-type gain."""
    name = "Classical Wiener"

    def __init__(self, floor=0.08, alpha=1.5):
        self.floor, self.alpha = floor, alpha

    def process(self, x, sr):
        nper = 1024 if sr >= 32000 else 512
        _, _, Z = signal.stft(x, sr, nperseg=nper, noverlap=nper * 3 // 4)
        P = np.abs(Z) ** 2
        Pn = uniform_filter1d(np.percentile(P, 20, axis=1, keepdims=True) * 3.0, 7, axis=0)
        G = np.maximum(1 - self.alpha * Pn / (P + 1e-12), self.floor)
        _, y = signal.istft(Z * np.sqrt(G), sr, nperseg=nper, noverlap=nper * 3 // 4)
        y = y[:len(x)]
        return (np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y).astype(np.float32)


def get_enhancer(name, model_name="DeepFilterNet3", post_filter=False):
    return WienerEnhancer() if name.lower().startswith("classical") else DeepFilterNetEnhancer(model_name, post_filter)


def stream_process(enh, x, sr, hop_ms=40, context_ms=300, lookahead_ms=40):
    """Simulated streaming: each hop the model sees [left context + hop] and only the newest
    hop is kept; wall time per chunk is logged.
    latency_ms = hop (frame length) + look-ahead + per-chunk processing time (Sec. 4.4)."""
    hop, ctx = int(sr * hop_ms / 1000), int(sr * context_ms / 1000)
    out = x.copy()
    times = []
    for i in range(0, len(x) - hop + 1, hop):
        win = x[max(0, i + hop - ctx - hop):i + hop]
        t0 = time.perf_counter()
        y = enh.process(win, sr)
        times.append(time.perf_counter() - t0)
        out[i:i + hop] = y[-hop:]
    t = np.array(times) * 1000 if times else np.array([0.0])
    return out, {
        "chunk_ms_mean": float(t.mean()), "chunk_ms_p95": float(np.percentile(t, 95)),
        "rtf": float(t.mean() / hop_ms),
        "latency_ms": float(hop_ms + lookahead_ms + t.mean()),
        "latency_ms_p95": float(hop_ms + lookahead_ms + np.percentile(t, 95)),
    }


def timed_offline(enh, x, sr):
    t0 = time.perf_counter()
    y = enh.process(x, sr)
    el = time.perf_counter() - t0
    return y, {"rtf": el / (len(x) / sr), "proc_s": el}
