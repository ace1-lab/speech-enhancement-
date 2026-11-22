from PySide6.QtCore import QObject, Signal
from core.config import Settings
from core.enhancer import get_enhancer


class AppState(QObject):
    recording_ready = Signal()
    run_finished = Signal()

    def __init__(self):
        super().__init__()
        self.settings = Settings.load()
        self.recording = None           # float32 @ 48 kHz
        self.history = []               # list of dict(file, engine, mode, metrics)
        self._engines = {}

    def engine(self, name):
        key = (name, self.settings.model_name, self.settings.post_filter)
        if key not in self._engines:
            self._engines[key] = get_enhancer(name, self.settings.model_name, self.settings.post_filter)
        return self._engines[key]

    def clear_engines(self):
        self._engines.clear()
