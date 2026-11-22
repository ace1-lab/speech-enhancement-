"""Analytics + Settings pages."""
from pathlib import Path
import pandas as pd
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget,
                               QTableWidgetItem, QProgressBar, QFileDialog, QFormLayout, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QLineEdit, QFrame)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from core.benchmark import run_benchmark
from .widgets import page_header, Worker
from .theme import C

SCEN = {"noisy": ("Noisy input", "#FF5C5C"), "classical": ("Classical Wiener", "#FFB547"),
        "dfn_offline": ("DeepFilterNet offline", "#3DDC97"), "dfn_streaming": ("DeepFilterNet streaming", "#4C9AFF")}


class AnalyticsPage(QWidget):
    def __init__(self, state):
        super().__init__(); self.setObjectName("page"); self.state = state; self.df = None
        l = QVBoxLayout(self); l.setContentsMargins(28, 24, 28, 24); l.setSpacing(12)
        l.addWidget(page_header("Analytics", "Benchmark of the four scenarios on the held-out tactical-noise test set"))
        row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run benchmark"); self.btn_run.setObjectName("primary")
        self.btn_load = QPushButton("Load results CSV…"); self.noise = QComboBox(); self.noise.addItem("All noise types")
        self.prog = QProgressBar(); self.prog.setFixedWidth(200)
        self.info = QLabel(""); self.info.setObjectName("muted")
        for w in (self.btn_run, self.btn_load, QLabel("Filter"), self.noise, self.prog, self.info):
            row.addWidget(w)
        row.addStretch(); l.addLayout(row)
        self.fig = Figure(figsize=(9, 4.2), facecolor=C["panel"]); self.canvas = FigureCanvasQTAgg(self.fig)
        l.addWidget(self.canvas, 3)
        self.table = QTableWidget(); l.addWidget(self.table, 2)
        self.btn_run.clicked.connect(self.run); self.btn_load.clicked.connect(self.pick)
        self.noise.currentIndexChanged.connect(self.render)
        if Path(state.settings.results_csv).exists():
            self.load(state.settings.results_csv)

    def pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "Results", "", "CSV (*.csv)")
        if p:
            self.load(p)

    def load(self, path):
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            self.info.setText(f"Cannot read CSV: {e}"); return
        self.noise.blockSignals(True); self.noise.clear(); self.noise.addItem("All noise types")
        self.noise.addItems(sorted(self.df.noise.unique())); self.noise.blockSignals(False); self.render()

    def run(self):
        s = self.state.settings
        from core.config import ROOT
        man = Path(s.manifest) if Path(s.manifest).is_absolute() else ROOT / s.manifest
        demo = ROOT / "data" / "manifest_demo.csv"
        if not man.exists():
            if not demo.exists():
                self.info.setText("No manifest. Run setup_all.py first."); return
            man = demo
        tag = " (synthetic DEMO data: run setup_all.py for real speech)" if man == demo else ""
        self.btn_run.setEnabled(False); self.info.setText("Benchmarking…" + tag)
        w = Worker(lambda cb: run_benchmark(str(man), s.results_csv, s, progress=cb))
        w.sig.progress.connect(lambda d, t: (self.prog.setRange(0, t), self.prog.setValue(d)))
        w.sig.done.connect(self._done); w.sig.error.connect(lambda e: (self.info.setText(f"✖ {e}"), self.btn_run.setEnabled(True)))
        QThreadPool.globalInstance().start(w)

    def _done(self, df):
        self.btn_run.setEnabled(True); self.df = df; self.info.setText(f"{len(df)} rows saved to {self.state.settings.results_csv}")
        self.load(self.state.settings.results_csv); self.state.run_finished.emit()

    def render(self):
        if self.df is None:
            return
        d = self.df if self.noise.currentIndex() <= 0 else self.df[self.df.noise == self.noise.currentText()]
        s = self.state.settings
        self.fig.clear()
        specs = [("stoi", "STOI", s.target_stoi), ("pesq", "PESQ (wb)", s.target_pesq), ("sisdr", "SI-SDR (dB)", None),
                 ("nfr_db", "Noise-floor reduction (dB)", 15)]
        for i, (col, title, tgt) in enumerate(specs):
            ax = self.fig.add_subplot(1, 4, i + 1, facecolor=C["panel"])
            for k, (name, color) in SCEN.items():
                g = d[d.scenario == k].groupby("snr")[col].agg(["mean", "std"]).dropna(subset=["mean"])
                if len(g):
                    ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=color, marker="o", capsize=3, label=name, lw=1.6)
            if tgt is not None:
                ax.axhline(tgt, color=C["muted"], ls="--", lw=1)
            ax.set_title(title, color=C["text"], fontsize=10); ax.set_xlabel("Input SNR (dB)", color=C["muted"], fontsize=8)
            ax.tick_params(colors=C["muted"], labelsize=8); ax.grid(alpha=.15)
            for sp in ax.spines.values():
                sp.set_color(C["border"])
            if i == 0:
                ax.legend(fontsize=7, facecolor=C["panel2"], edgecolor=C["border"], labelcolor=C["text"])
        self.fig.tight_layout(); self.canvas.draw_idle()
        cols = [c for c in ["pesq", "stoi", "sisdr", "sisdr_gain", "nfr_db", "rtf", "latency_ms"] if c in d]
        t = d.groupby(["scenario", "snr"])[cols].mean().round(3).reset_index()
        self.table.setRowCount(len(t)); self.table.setColumnCount(len(t.columns)); self.table.setHorizontalHeaderLabels(list(t.columns))
        for r, row in enumerate(t.itertuples(index=False)):
            for c, v in enumerate(row):
                it = QTableWidgetItem("" if v != v else str(v))
                if t.columns[c] == "stoi" and v == v and v > s.target_stoi:
                    it.setForeground(Qt_green())
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()


def Qt_green():
    from PySide6.QtGui import QColor
    return QColor(C["accent"])


class SettingsPage(QWidget):
    def __init__(self, state):
        super().__init__(); self.setObjectName("page"); self.state = state; s = state.settings
        l = QVBoxLayout(self); l.setContentsMargins(28, 24, 28, 24); l.setSpacing(12)
        l.addWidget(page_header("Settings", "Model, streaming, targets and paths"))
        card = QFrame(); card.setObjectName("card"); f = QFormLayout(card); f.setContentsMargins(20, 16, 20, 16); f.setSpacing(10)
        self.model = QComboBox(); self.model.addItems(["DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3"]); self.model.setCurrentText(s.model_name)
        self.post = QCheckBox("Enable post-filter (stronger suppression, may colour speech)"); self.post.setChecked(s.post_filter)
        self.hop = QSpinBox(); self.hop.setRange(10, 500); self.hop.setSuffix(" ms"); self.hop.setValue(s.hop_ms)
        self.ctx = QSpinBox(); self.ctx.setRange(0, 2000); self.ctx.setSuffix(" ms"); self.ctx.setValue(s.context_ms)
        self.look = QSpinBox(); self.look.setRange(0, 200); self.look.setSuffix(" ms"); self.look.setValue(s.lookahead_ms)
        self.t_stoi = QDoubleSpinBox(); self.t_stoi.setRange(0, 1); self.t_stoi.setSingleStep(.01); self.t_stoi.setValue(s.target_stoi)
        self.t_pesq = QDoubleSpinBox(); self.t_pesq.setRange(1, 4.64); self.t_pesq.setSingleStep(.1); self.t_pesq.setValue(s.target_pesq)
        self.t_lat = QDoubleSpinBox(); self.t_lat.setRange(10, 1000); self.t_lat.setSuffix(" ms"); self.t_lat.setValue(s.target_latency_ms)
        self.out = QLineEdit(s.output_dir); self.man = QLineEdit(s.manifest)
        for lab, w in (("DeepFilterNet version", self.model), ("", self.post), ("Streaming hop", self.hop), ("Left context", self.ctx),
                       ("Look-ahead (nominal)", self.look), ("STOI target", self.t_stoi), ("PESQ target", self.t_pesq),
                       ("Latency target", self.t_lat), ("Output folder", self.out), ("Test manifest", self.man)):
            f.addRow(lab, w)
        l.addWidget(card)
        row = QHBoxLayout(); self.save = QPushButton("Save settings"); self.save.setObjectName("primary")
        self.test = QPushButton("Test model load"); self.msg = QLabel(""); self.msg.setObjectName("muted")
        row.addWidget(self.save); row.addWidget(self.test); row.addWidget(self.msg); row.addStretch(); l.addLayout(row); l.addStretch()
        self.save.clicked.connect(self.apply); self.test.clicked.connect(self.try_model)

    def apply(self):
        s = self.state.settings
        s.model_name, s.post_filter, s.hop_ms, s.context_ms, s.lookahead_ms = self.model.currentText(), self.post.isChecked(), self.hop.value(), self.ctx.value(), self.look.value()
        s.target_stoi, s.target_pesq, s.target_latency_ms = self.t_stoi.value(), self.t_pesq.value(), self.t_lat.value()
        s.output_dir, s.manifest = self.out.text(), self.man.text()
        Path(s.output_dir).mkdir(parents=True, exist_ok=True); s.save(); self.state.clear_engines()
        self.state.run_finished.emit(); self.msg.setText("✔ Saved")

    def try_model(self):
        self.apply(); self.msg.setText("Loading…")
        w = Worker(lambda _: self.state.engine("DeepFilterNet").info())
        w.sig.done.connect(lambda i: self.msg.setText(f"✔ {i['name']} · {i['params'] / 1e6:.2f} M params · {i['sr']} Hz"))
        w.sig.error.connect(lambda e: self.msg.setText(f"✖ {e}"))
        QThreadPool.globalInstance().start(w)
