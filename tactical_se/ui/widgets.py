"""Reusable UI building blocks: cards, waveform plot, spectrogram canvas, player, worker."""
import time
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import librosa
from PySide6.QtCore import Qt, QTimer, QObject, Signal, QRunnable
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from .theme import C

pg.setConfigOptions(antialias=True, background=C["panel"], foreground=C["muted"])


def page_header(title, subtitle):
    w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 8); l.setSpacing(2)
    a = QLabel(title); a.setObjectName("h1"); b = QLabel(subtitle); b.setObjectName("sub")
    l.addWidget(a); l.addWidget(b)
    return w


class Card(QFrame):
    """Metric card: label, big value, small caption; colour reflects pass/fail."""
    def __init__(self, label, value="–", caption=""):
        super().__init__(); self.setObjectName("card")
        l = QVBoxLayout(self); l.setContentsMargins(16, 12, 16, 12); l.setSpacing(2)
        self.lbl = QLabel(label.upper()); self.lbl.setObjectName("cardlbl")
        self.val = QLabel(value); self.val.setObjectName("cardval")
        self.cap = QLabel(caption); self.cap.setObjectName("muted")
        for w in (self.lbl, self.val, self.cap):
            l.addWidget(w)

    def set(self, value, caption=None, ok=None):
        self.val.setText(value)
        if caption is not None:
            self.cap.setText(caption)
        col = C["text"] if ok is None else (C["accent"] if ok else C["warn"])
        self.val.setStyleSheet(f"color:{col};")


class WaveformPlot(pg.PlotWidget):
    """Time-domain waveform, optionally overlaying two signals (original vs enhanced)."""
    def __init__(self, height=150):
        super().__init__(); self.setMinimumHeight(height)
        self.showGrid(x=True, y=True, alpha=0.15); self.setMouseEnabled(x=True, y=False)
        self.setYRange(-1, 1); self.setLabel("bottom", "Time (s)"); self.setLabel("left", "Amplitude")
        self.getPlotItem().hideButtons()
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(C["warn"], width=1.5)); self.cursor.hide()
        self.addItem(self.cursor)

    def show_signals(self, sigs, sr, colors, names=None):
        self.clear(); self.addItem(self.cursor)
        for i, (s, col) in enumerate(zip(sigs, colors)):
            step = max(1, len(s) // 6000)                       # min/max decimation keeps peaks
            n = len(s) // step * step
            r = s[:n].reshape(-1, step)
            t = np.arange(r.shape[0]) * step / sr
            self.plot(np.repeat(t, 2), np.stack([r.min(1), r.max(1)], 1).ravel(), pen=pg.mkPen(col, width=1),
                      name=names[i] if names else None)
        self.setXRange(0, len(sigs[0]) / sr, padding=0.01)

    def set_cursor(self, t):
        self.cursor.setPos(t); self.cursor.show()


class SpectrogramCanvas(FigureCanvasQTAgg):
    """Matplotlib log-mel-free STFT spectrogram, dark themed. Shows several signals side by side."""
    def __init__(self, n=2, height=3.0):
        self.fig = Figure(figsize=(8, height), facecolor=C["panel"]); super().__init__(self.fig)
        self.n = n; self.setMinimumHeight(int(height * 80))

    def show_signals(self, sigs, sr, titles, vmin=-90):
        self.fig.clear()
        for i, (s, t) in enumerate(zip(sigs, titles)):
            ax = self.fig.add_subplot(1, len(sigs), i + 1, facecolor=C["panel"])
            S = librosa.amplitude_to_db(np.abs(librosa.stft(s, n_fft=1024, hop_length=256)), ref=1.0, top_db=None)
            ax.imshow(np.clip(S, vmin, 0), origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=0,
                      extent=[0, len(s) / sr, 0, sr / 2000])
            ax.set_ylim(0, min(12, sr / 2000)); ax.set_title(t, color=C["text"], fontsize=10)
            ax.set_xlabel("Time (s)", color=C["muted"], fontsize=8)
            if i == 0:
                ax.set_ylabel("kHz", color=C["muted"], fontsize=8)
            ax.tick_params(colors=C["muted"], labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(C["border"])
        self.fig.tight_layout(); self.draw_idle()


class PlayerBar(QFrame):
    """Play / stop with a progress cursor; drives an optional WaveformPlot cursor."""
    def __init__(self, title, color, plot=None):
        super().__init__(); self.setObjectName("card")
        self.data, self.sr, self.plot, self._t0, self._off = None, 48000, plot, 0, 0
        l = QHBoxLayout(self); l.setContentsMargins(12, 8, 12, 8)
        dot = QLabel("●"); dot.setStyleSheet(f"color:{color};")
        self.title = QLabel(title); self.play = QPushButton("▶  Play"); self.stop = QPushButton("■")
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 1000); self.time = QLabel("0.0 / 0.0 s")
        self.time.setObjectName("muted")
        for w in (dot, self.title, self.play, self.stop, self.slider, self.time):
            l.addWidget(w)
        self.play.clicked.connect(self.start); self.stop.clicked.connect(self.halt)
        self.slider.sliderReleased.connect(self._seek)
        self.timer = QTimer(self); self.timer.setInterval(40); self.timer.timeout.connect(self._tick)
        self.setEnabled(False)

    def set_audio(self, x, sr):
        self.halt(); self.data, self.sr, self._off = x, sr, 0
        self.time.setText(f"0.0 / {len(x) / sr:.1f} s"); self.slider.setValue(0); self.setEnabled(x is not None)

    def start(self):
        if self.data is None:
            return
        sd.stop(); i = int(self._off * self.sr)
        sd.play(self.data[i:], self.sr); self._t0 = time.perf_counter(); self.timer.start()

    def halt(self):
        sd.stop(); self.timer.stop(); self._off = 0
        if self.plot:
            self.plot.cursor.hide()

    def _seek(self):
        if self.data is not None:
            self._off = self.slider.value() / 1000 * len(self.data) / self.sr
            self.start()

    def _tick(self):
        dur = len(self.data) / self.sr; t = self._off + time.perf_counter() - self._t0
        if t >= dur:
            self.halt(); self.slider.setValue(0); return
        if not self.slider.isSliderDown():
            self.slider.setValue(int(t / dur * 1000))
        self.time.setText(f"{t:.1f} / {dur:.1f} s")
        if self.plot:
            self.plot.set_cursor(t)


class _Signals(QObject):
    done = Signal(object); error = Signal(str); progress = Signal(int, int)


class Worker(QRunnable):
    """Run fn(progress_cb) off the GUI thread."""
    def __init__(self, fn):
        super().__init__(); self.fn, self.sig = fn, _Signals()

    def run(self):
        try:
            self.sig.done.emit(self.fn(lambda d, t: self.sig.progress.emit(d, t)))
        except Exception as e:                                   # surfaced in the UI, never a crash
            import traceback; traceback.print_exc(); self.sig.error.emit(f"{type(e).__name__}: {e}")
