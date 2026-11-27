"""Run the four scenarios of Sec. 4.3 on the manifest and write a results CSV."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.audio_io import load_audio, DF_SR
from core.enhancer import get_enhancer, stream_process, timed_offline
from core.metrics import evaluate
from core.config import Settings, ROOT


def _p(x):
    return Path(x) if Path(x).is_absolute() else ROOT / x


def run_benchmark(manifest, out_csv, settings=None, scenarios=("noisy", "classical", "dfn_offline", "dfn_streaming"),
                  limit=None, progress=None, log=None):
    s = settings or Settings.load()
    _log = log if callable(log) else (lambda msg: None)
    _log(f"Initializing benchmark on {Path(manifest).name} (limit={limit or 'All'})")
    df = pd.read_csv(_p(manifest))
    if limit:
        df = df.head(limit)
    _log("Loading Classical Wiener and DeepFilterNet engines...")
    classical = get_enhancer("classical")
    dfn = get_enhancer("deepfilternet", s.model_name, s.post_filter) if any(k.startswith("dfn") for k in scenarios) else None
    if dfn:
        _log(f"DeepFilterNet3 active (epoch 120, {s.model_name}) on device CPU")
    rows, total = [], len(df) * len(scenarios)
    done = 0
    for _, r in df.iterrows():
        clean, _ = load_audio(_p(r.clean), DF_SR, norm=False)
        noisy, _ = load_audio(_p(r.noisy), DF_SR, norm=False)
        for sc in scenarios:
            extra = {}
            if sc == "noisy":
                y = noisy
            elif sc == "classical":
                y, extra = timed_offline(classical, noisy, DF_SR)
            elif sc == "dfn_offline":
                y, extra = timed_offline(dfn, noisy, DF_SR)
            else:
                y, extra = stream_process(dfn, noisy, DF_SR, s.hop_ms, s.context_ms, s.lookahead_ms)
            m = evaluate(clean, noisy, y, DF_SR)
            rows.append(dict(file=Path(r.noisy).name, noise=r.noise, snr=r.snr, scenario=sc, **m, **extra))
            done += 1
            info_str = f"[{done}/{total}] {Path(r.noisy).name} ({r.noise}, {r.snr:+d}dB) ──▶ {sc} | STOI: {m.get('stoi', 0):.3f} | SI-SDR Gain: {m.get('sisdr_gain', 0):+.1f}dB"
            _log(info_str)
            if progress:
                try:
                    progress(done, total, info_str)
                except TypeError:
                    progress(done, total)
    res = pd.DataFrame(rows)
    _p(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(_p(out_csv), index=False)
    _log(f"✔ Benchmark complete. {len(res)} evaluations saved to {out_csv}")
    return res


def summarize(res, s=None):
    """Mean/std per scenario x SNR (and per noise type) and flags for missed targets."""
    s = s or Settings.load()
    cols = [c for c in ["pesq", "stoi", "sisdr", "sisdr_gain", "nfr_db", "rtf", "latency_ms"] if c in res]
    g = res.groupby(["scenario", "snr"])[cols].agg(["mean", "std"]).round(3)
    pn = res.groupby(["scenario", "noise", "snr"])[cols].mean().round(3)
    pn["meets_targets"] = (pn.get("stoi", 0) > s.target_stoi) & (pn.get("pesq", 0) > s.target_pesq)
    return g, pn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="results/benchmark_results.csv")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    man = a.manifest or ("data/manifest_test.csv" if _p("data/manifest_test.csv").exists() else "data/manifest_demo.csv")
    res = run_benchmark(man, a.out, limit=a.limit, progress=lambda d, t: print(f"\r{d}/{t}", end=""))
    g, pn = summarize(res)
    print("\n", g.to_string()); print(pn.to_string())
