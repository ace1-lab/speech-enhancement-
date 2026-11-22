# TacticalClear: Real-Time Deep-Filtering Speech Enhancement for Tactical Noise (Team 10)

PySide6 desktop app: pretrained DeepFilterNet (no training) in a modular PyTorch pipeline, five pages
(Dashboard, Live Recording, Enhancement, Analytics, Settings), waveforms + spectrograms, side-by-side playback,
PESQ / STOI / SI-SDR / noise-floor reduction / RTF / latency, WAV export.

## Quick start (Python 3.10 or 3.11)
    python -m venv .venv && .venv\Scripts\activate          (Linux/macOS: source .venv/bin/activate)
    python setup_all.py          # installs packages, fetches real speech subset, builds dataset, loads the model
    python main.py               # or run.bat / ./run.sh

## What is included
- `data/demo/` + `data/manifest_demo.csv`: 48 ready-made mixtures (4 SYNTHETIC speech-like clips x 4 tactical noises x -5/0/+5 dB).
  For smoke-testing the pipeline and GUI offline. Scores on synthetic "speech" are NOT meaningful research results.
- `setup_all.py` builds the REAL dataset: ~22 VoiceBank utterances (speakers p232 and p257, CC BY 4.0) pulled with HTTP range
  requests from the clean test-set zip (154 MB zip, only a few MB transferred), mixed with procedurally generated tactical
  noise (engine, wind, gunfire, convoy) at -5/0/+5 dB. Speaker-disjoint: p232 = dev (tuning), p257 = unseen test.
  Output: `data/manifest_dev.csv`, `data/manifest_test.csv`. Optional real noise: `--noise_dir` on prepare_dataset.py (DEMAND).
- DeepFilterNet weights are not bundled; they are fetched by the `deepfilternet` package on first model load, then offline.
  Without it the app falls back to the classical Wiener baseline only for the "Classical Wiener" engine.

## Using the app
Enhancement: load any WAV (validated for format/rate/bit depth), or use a recording, or load a clean file and tick
"simulate tactical noise". Analytics -> Run benchmark runs noisy / classical / DFN offline / DFN streaming over the manifest
(test set if present, otherwise the demo set). CLI: `python core/benchmark.py --limit 6`.

## Notes
- Streaming = chunked: each hop the model sees left context + new hop, keeps the newest hop. Latency = hop + nominal
  look-ahead (Settings, assumed 40 ms: verify for your model version) + measured per-chunk time.
- PESQ is wide-band at 16 kHz. Report numbers as measured on your machine; simulated noise only; not for field use.
- Not yet verified on a real desktop: GUI rendering, microphone capture and DeepFilterNet loading (no display/internet where this was built).
  If the `torchaudio.backend` import error appears, keep the pinned torch 2.0.1 / torchaudio 2.0.2.
