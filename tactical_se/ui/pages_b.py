"""Enhancement page: load/simulate -> enhance -> compare (waveform, spectrogram, playback) -> metrics -> export."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QSpinBox,
                               QCheckBox, QFileDialog, QProgressBar, QFrame, QScrollArea)
from core.audio_io import load_audio, save_wav, DF_SR, AudioValidationError
from core.noise_sim import generate_tactical, mix_at_snr, KINDS
from core.enhancer import stream_process, timed_offline
from core.metrics import evaluate
from .widgets import page_header, Card, WaveformPlot, SpectrogramCanvas, PlayerBar, Worker
from .theme import C


class EnhancementPage(QWidget):
    def __init__(self, state):
        super().__init__(); self.setObjectName("page"); self.state = state
        self.src = self.clean = self.noisy = self.enhanced = None; self.src_name = ""
        root = QHBoxLayout(self); root.setContentsMargins(28, 24, 28, 24); root.setSpacing(18)
        # ---- controls
        side = QFrame(); side.setObjectName("card"); side.setFixedWidth(300); sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 16, 16, 16); sl.setSpacing(8)
        self.btn_load = QPushButton("Load WAV…"); self.btn_rec = QPushButton("Use last recording")
        self.btn_ref = QPushButton("Load clean reference… (optional)")
        self.file_lbl = QLabel("No audio loaded"); self.file_lbl.setObjectName("muted"); self.file_lbl.setWordWrap(True)
        self.sim = QCheckBox("Input is clean: simulate tactical noise")
        self.kind = QComboBox(); self.kind.addItems(KINDS)
        self.snr = QSpinBox(); self.snr.setRange(-5, 5); self.snr.setValue(-5); self.snr.setSuffix(" dB SNR")
        self.engine = QComboBox(); self.engine.addItems(["DeepFilterNet", "Classical Wiener"])
        self.mode = QComboBox(); self.mode.addItems(["Offline (whole file)", "Streaming (chunked)"])
        self.btn_run = QPushButton("⚡  Enhance"); self.btn_run.setObjectName("primary"); self.btn_run.setMinimumHeight(42)
        self.btn_export = QPushButton("Export enhanced WAV…"); self.btn_export.setEnabled(False)
        self.busy = QProgressBar(); self.busy.setRange(0, 1); self.busy.setFixedHeight(10)
        self.msg = QLabel(""); self.msg.setWordWrap(True)
        for w in (self.btn_load, self.btn_rec, self.btn_ref, self.file_lbl, self.sim, self.kind, self.snr,
                  QLabel("Engine"), self.engine, QLabel("Mode"), self.mode, self.btn_run, self.busy, self.btn_export, self.msg):
            sl.addWidget(w)
        sl.addStretch(); root.addWidget(side)
        # ---- results
        scroll = QScrollArea(); scroll.setWidgetResizable(True); inner = QWidget(); inner.setObjectName("page")
        r = QVBoxLayout(inner); r.setSpacing(12)
        r.addWidget(page_header("Enhancement", "Original vs enhanced: playback, waveforms, spectrograms and objective scores"))
        g = QGridLayout(); g.setSpacing(10)
        self.cards = {k: Card(k) for k in ("PESQ", "STOI", "SI-SDR", "Noise floor", "RTF", "Latency")}
        for i, c in enumerate(self.cards.values()):
            g.addWidget(c, 0, i)
        r.addLayout(g)
        self.wave = WaveformPlot(200); r.addWidget(self.wave)
        self.p_orig = PlayerBar("Original (noisy)", C["danger"], self.wave); self.p_enh = PlayerBar("Enhanced", C["accent"], self.wave)
        r.addWidget(self.p_orig); r.addWidget(self.p_enh)
        self.spec = SpectrogramCanvas(2, 3.2); r.addWidget(self.spec)
        scroll.setWidget(inner); root.addWidget(scroll, 1)
        self.btn_load.clicked.connect(self.load); self.btn_rec.clicked.connect(self.use_recording)
        self.btn_ref.clicked.connect(self.load_ref); self.btn_run.clicked.connect(self.run)
        self.btn_export.clicked.connect(self.export); state.recording_ready.connect(self.use_recording)

    # ---------- input
    def _say(self, text, ok=True):
        self.msg.setText(text); self.msg.setStyleSheet(f"color:{C['accent'] if ok else C['danger']};")

    def load(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open WAV", "", "WAV audio (*.wav)")
        if not p:
            return
        try:
            x, info = load_audio(p, DF_SR)
        except AudioValidationError as e:
            self._say(f"✖ {e}", False); return
        self.src, self.src_name, self.clean = x, Path(p).name, None
        self.file_lbl.setText(f"{self.src_name}\n{info.samplerate} Hz · {info.subtype} · {info.duration:.1f}s → 48 kHz mono")
        self._say("✔ File validated" + ("" if info.samplerate == DF_SR else f" (resampled from {info.samplerate} Hz)"))

    def use_recording(self):
        if self.state.recording is not None:
            self.src, self.src_name, self.clean = self.state.recording, "live recording", None
            self.file_lbl.setText("live recording · 48 kHz mono"); self._say("✔ Recording loaded")

    def load_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, "Clean reference", "", "WAV audio (*.wav)")
        if p:
            try:
                self.clean, _ = load_audio(p, DF_SR, norm=False); self._say("✔ Reference loaded: PESQ/STOI/SI-SDR enabled")
            except AudioValidationError as e:
                self._say(f"✖ {e}", False)

    # ---------- run
    def run(self):
        if self.src is None:
            self._say("Load a WAV or use a recording first.", False); return
        if self.sim.isChecked():
            clean = self.src.copy()
            noise = generate_tactical(self.kind.currentText(), len(clean) / DF_SR + 1, DF_SR, seed=1)
            noisy, _ = mix_at_snr(clean, noise, self.snr.value())
            pk = np.max(np.abs(noisy)); noisy = noisy * 0.95 / pk if pk > 0.95 else noisy
            self.clean, self.noisy = clean, noisy
        else:
            self.noisy = self.src
            if self.clean is not None and len(self.clean) != len(self.noisy):
                n = min(len(self.clean), len(self.noisy)); self.clean, self.noisy = self.clean[:n], self.noisy[:n]
        eng_name, streaming = self.engine.currentText(), self.mode.currentIndex() == 1
        s = self.state.settings; noisy, clean = self.noisy, self.clean

        def job(_):
            enh = self.state.engine(eng_name)
            if streaming:
                y, stats = stream_process(enh, noisy, DF_SR, s.hop_ms, s.context_ms, s.lookahead_ms)
            else:
                y, stats = timed_offline(enh, noisy, DF_SR)
            return y, stats, evaluate(clean, noisy, y, DF_SR)

        self.btn_run.setEnabled(False); self.busy.setRange(0, 0); self._say("Loading model & processing…")
        w = Worker(job); w.sig.done.connect(lambda res: self._done(res, eng_name, streaming)); w.sig.error.connect(self._fail)
        QThreadPool.globalInstance().start(w)

    def _fail(self, e):
        self.busy.setRange(0, 1); self.btn_run.setEnabled(True)
        hint = " Install it with: pip install deepfilternet" if "df" in e.lower() or "deepfilter" in e.lower() else ""
        self._say(f"✖ {e}.{hint}", False)

    def _done(self, res, eng, streaming):
        y, stats, m = res; s = self.state.settings
        self.enhanced = y; self.busy.setRange(0, 1); self.btn_run.setEnabled(True); self.btn_export.setEnabled(True)
        self.wave.show_signals([self.noisy, y], DF_SR, [C["danger"], C["accent"]], ["noisy", "enhanced"])
        self.spec.show_signals([self.noisy, y], DF_SR, ["Original (noisy)", "Enhanced"])
        self.p_orig.set_audio(self.noisy, DF_SR); self.p_enh.set_audio(y, DF_SR)
        f = lambda k, d=2: "n/a" if k not in m or m[k] != m[k] else f"{m[k]:.{d}f}"
        c = self.cards
        c["PESQ"].set(f("pesq"), f"noisy {f('pesq_noisy')}", m.get("pesq", 0) > s.target_pesq if "pesq" in m else None)
        c["STOI"].set(f("stoi", 3), f"noisy {f('stoi_noisy', 3)}", m.get("stoi", 0) > s.target_stoi if "stoi" in m else None)
        c["SI-SDR"].set(f"{m['sisdr']:.1f} dB" if "sisdr" in m else "n/a", f"gain {m.get('sisdr_gain', float('nan')):+.1f} dB" if "sisdr" in m else "needs reference")
        c["Noise floor"].set(f"−{m['nfr_db']:.1f} dB", "target 15–30 dB", 15 <= m["nfr_db"])
        c["RTF"].set(f"{stats['rtf']:.2f}", "<1 = faster than real time", stats["rtf"] < 1)
        if streaming:
            c["Latency"].set(f"{stats['latency_ms']:.0f} ms", f"p95 {stats['latency_ms_p95']:.0f} ms", stats["latency_ms"] < s.target_latency_ms)
        else:
            c["Latency"].set("–", "use Streaming mode")
        self.state.history.append(dict(file=self.src_name, engine=eng, mode="streaming" if streaming else "offline", metrics=m))
        self.state.run_finished.emit(); self._say("✔ Done. Compare by playing both tracks; check that speech was preserved.")

    def export(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export", str(Path(self.state.settings.output_dir) / f"enhanced_{Path(self.src_name).stem}.wav"), "WAV (*.wav)")
        if p:
            save_wav(p, self.enhanced, DF_SR); self._say(f"✔ Saved {Path(p).name}")
