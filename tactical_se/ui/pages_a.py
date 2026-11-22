"""Dashboard + Live Recording pages."""
import queue
from pathlib import Path
import numpy as np
import pandas as pd
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox,
                               QProgressBar, QFileDialog, QMessageBox, QFrame)
from core.audio_io import DF_SR, resample, normalize, save_wav
from .widgets import page_header, Card, WaveformPlot
from .theme import C


class DashboardPage(QWidget):
    def __init__(self, state, goto):
        super().__init__(); self.setObjectName("page"); self.state, self.goto = state, goto
        l = QVBoxLayout(self); l.setContentsMargins(28, 24, 28, 24); l.setSpacing(16)
        l.addWidget(page_header("Mission Dashboard", "Real-time deep-filtering speech enhancement for tactical noise"))
        g = QGridLayout(); g.setSpacing(14)
        self.c_model = Card("Engine", state.settings.model_name, "Pretrained DeepFilterNet · offline")
        self.c_stoi = Card("STOI target", f"> {state.settings.target_stoi}", "Intelligibility")
        self.c_pesq = Card("PESQ target", f"> {state.settings.target_pesq}", "Perceptual quality")
        self.c_lat = Card("Latency target", f"< {state.settings.target_latency_ms:.0f} ms", "Streaming budget")
        for i, c in enumerate((self.c_model, self.c_stoi, self.c_pesq, self.c_lat)):
            g.addWidget(c, 0, i)
        self.r_runs = Card("Session runs", "0", "Enhancements this session")
        self.r_last = Card("Last STOI", "–", "")
        self.r_bench = Card("Benchmark mean STOI", "–", "Run Analytics → Benchmark")
        self.r_nfr = Card("Last noise-floor cut", "–", "dB on non-speech frames")
        for i, c in enumerate((self.r_runs, self.r_last, self.r_bench, self.r_nfr)):
            g.addWidget(c, 1, i)
        l.addLayout(g)
        row = QHBoxLayout()
        for text, idx, primary in (("●  Record live audio", 1, False), ("⚡  Enhance a WAV file", 2, True),
                                   ("▦  View analytics", 3, False)):
            b = QPushButton(text); b.setMinimumHeight(44)
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(lambda _=False, i=idx: goto(i)); row.addWidget(b)
        l.addLayout(row)
        how = QFrame(); how.setObjectName("card"); hl = QVBoxLayout(how)
        t = QLabel("Pipeline"); t.setObjectName("h1"); t.setStyleSheet("font-size:15px;")
        s = QLabel("Audio input  →  Validation & preprocessing (48 kHz)  →  Tactical-noise simulation (−5…+5 dB)  →  "
                   "Segmentation  →  DeepFilterNet inference  →  Overlap-add reconstruction  →  Waveforms / spectrograms  →  "
                   "PESQ · STOI · SI-SDR · latency · WAV export")
        s.setWordWrap(True); s.setObjectName("sub")
        note = QLabel("Research prototype: evaluates speech clarity under simulated tactical noise. Not validated for field use.")
        note.setStyleSheet(f"color:{C['warn']};"); note.setWordWrap(True)
        for w in (t, s, note):
            hl.addWidget(w)
        l.addWidget(how); l.addStretch()
        state.run_finished.connect(self.refresh); self.refresh()

    def refresh(self):
        st = self.state.settings
        self.c_model.set(st.model_name)
        h = self.state.history
        self.r_runs.set(str(len(h)))
        if h:
            m = h[-1]["metrics"]
            if "stoi" in m:
                self.r_last.set(f"{m['stoi']:.3f}", f"{h[-1]['engine']} · {h[-1]['mode']}", m["stoi"] > st.target_stoi)
            self.r_nfr.set(f"{m['nfr_db']:.1f} dB", None, 15 <= m["nfr_db"])
        p = Path(st.results_csv)
        if p.exists():
            try:
                d = pd.read_csv(p); d = d[d.scenario == "dfn_offline"]
                if len(d):
                    self.r_bench.set(f"{d.stoi.mean():.3f}", f"{len(d)} files · DFN offline", d.stoi.mean() > st.target_stoi)
            except Exception:
                pass


class LiveRecordingPage(QWidget):
    WINDOW_S = 5

    def __init__(self, state, goto):
        super().__init__(); self.setObjectName("page"); self.state, self.goto = state, goto
        self.q, self.stream, self.chunks, self.ring, self.rec_sr = queue.Queue(), None, [], np.zeros(1), DF_SR
        l = QVBoxLayout(self); l.setContentsMargins(28, 24, 28, 24); l.setSpacing(14)
        l.addWidget(page_header("Live Recording", "Capture microphone audio with a live waveform and level meter"))
        top = QHBoxLayout()
        self.dev = QComboBox(); self.dev.setMinimumWidth(320)
        self.btn_refresh = QPushButton("↻"); self.btn_refresh.setFixedWidth(40)
        self.btn_rec = QPushButton("●  Record"); self.btn_rec.setObjectName("danger")
        self.btn_stop = QPushButton("■  Stop"); self.btn_stop.setEnabled(False)
        self.timer_lbl = QLabel("00:00.0"); self.timer_lbl.setStyleSheet("font-size:20px;font-weight:700;")
        for w in (QLabel("Input device"), self.dev, self.btn_refresh, self.btn_rec, self.btn_stop):
            top.addWidget(w)
        top.addStretch(); top.addWidget(self.timer_lbl); l.addLayout(top)
        self.plot = WaveformPlot(260); l.addWidget(self.plot, 1)
        lv = QHBoxLayout(); self.level = QProgressBar(); self.level.setRange(0, 100); self.level.setTextVisible(False)
        self.level_lbl = QLabel("−∞ dBFS"); self.level_lbl.setObjectName("muted"); self.level_lbl.setFixedWidth(90)
        lv.addWidget(QLabel("Level")); lv.addWidget(self.level); lv.addWidget(self.level_lbl); l.addLayout(lv)
        bot = QHBoxLayout()
        self.btn_save = QPushButton("Save WAV…"); self.btn_send = QPushButton("⚡  Send to Enhancement")
        self.btn_send.setObjectName("primary"); self.status = QLabel("Idle"); self.status.setObjectName("muted")
        for w in (self.btn_save, self.btn_send):
            w.setEnabled(False); bot.addWidget(w)
        bot.addStretch(); bot.addWidget(self.status); l.addLayout(bot)
        self.ui_timer = QTimer(self); self.ui_timer.setInterval(33); self.ui_timer.timeout.connect(self._update)
        self.btn_refresh.clicked.connect(self.populate); self.btn_rec.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop); self.btn_save.clicked.connect(self.save); self.btn_send.clicked.connect(self.send)
        self.populate()

    def populate(self):
        self.dev.clear(); self.dev.addItem("System default", -1)
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self.dev.addItem(f"{i}: {d['name']}", i)
        except Exception as e:
            self.status.setText(f"Audio devices unavailable: {e}")
        idx = self.dev.findData(self.state.settings.input_device)
        self.dev.setCurrentIndex(max(idx, 0))

    def start(self):
        dev = self.dev.currentData(); dev = None if dev == -1 else dev
        try:
            try:
                self.rec_sr = DF_SR
                self.stream = sd.InputStream(samplerate=DF_SR, channels=1, device=dev, dtype="float32", callback=self._cb)
            except Exception:                           # device doesn't support 48 kHz: record natively, resample later
                self.rec_sr = int(sd.query_devices(dev, "input")["default_samplerate"])
                self.stream = sd.InputStream(samplerate=self.rec_sr, channels=1, device=dev, dtype="float32", callback=self._cb)
            self.stream.start()
        except Exception as e:
            QMessageBox.warning(self, "Microphone", f"Could not open the input device.\n\n{e}"); return
        self.chunks, self.ring = [], np.zeros(self.WINDOW_S * self.rec_sr, np.float32)
        self.btn_rec.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False); self.btn_send.setEnabled(False)
        self.status.setText(f"Recording @ {self.rec_sr} Hz"); self.ui_timer.start()

    def _cb(self, indata, frames, t, status):
        self.q.put(indata[:, 0].copy())

    def _update(self):
        got = False
        while not self.q.empty():
            c = self.q.get(); self.chunks.append(c); got = True
            self.ring = np.concatenate([self.ring, c])[-self.WINDOW_S * self.rec_sr:]
        if not got:
            return
        n = sum(len(c) for c in self.chunks); self.timer_lbl.setText(f"{int(n / self.rec_sr // 60):02d}:{n / self.rec_sr % 60:04.1f}")
        self.plot.show_signals([self.ring], self.rec_sr, [C["accent"]])
        rms = float(np.sqrt(np.mean(self.chunks[-1] ** 2)) + 1e-9); db = 20 * np.log10(rms)
        self.level.setValue(int(np.clip((db + 60) / 60 * 100, 0, 100))); self.level_lbl.setText(f"{db:.0f} dBFS")

    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        self.ui_timer.stop(); self._update()
        self.btn_rec.setEnabled(True); self.btn_stop.setEnabled(False)
        if not self.chunks:
            return
        x = np.concatenate(self.chunks)
        if len(x) < self.rec_sr // 2:
            self.status.setText("Recording too short (<0.5 s)"); return
        self.state.recording = normalize(resample(x, self.rec_sr, DF_SR))
        self.plot.show_signals([self.state.recording], DF_SR, [C["accent"]])
        self.btn_save.setEnabled(True); self.btn_send.setEnabled(True)
        self.status.setText(f"Captured {len(x) / self.rec_sr:.1f} s · resampled to 48 kHz")

    def save(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save recording", str(Path(self.state.settings.output_dir) / "recording.wav"), "WAV (*.wav)")
        if p:
            save_wav(p, self.state.recording)

    def send(self):
        self.state.recording_ready.emit(); self.goto(2)
