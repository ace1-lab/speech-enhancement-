import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QButtonGroup
from ui.theme import QSS
from ui.state import AppState
from ui.pages_a import DashboardPage, LiveRecordingPage
from ui.pages_b import EnhancementPage
from ui.pages_c import AnalyticsPage, SettingsPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("TacticalClear · Deep-Filtering Speech Enhancement"); self.resize(1380, 860)
        self.state = AppState()
        root = QWidget(); h = QHBoxLayout(root); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0); self.setCentralWidget(root)
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(220); sl = QVBoxLayout(side); sl.setContentsMargins(14, 20, 14, 14)
        brand = QLabel("◉ TACTICALCLEAR"); brand.setObjectName("brand"); sl.addWidget(brand)
        sub = QLabel("Team 10 · Speech Enhancement"); sub.setObjectName("muted"); sl.addWidget(sub); sl.addSpacing(18)
        self.stack = QStackedWidget(); self.group = QButtonGroup(self); self.group.setExclusive(True)
        pages = [("Dashboard", DashboardPage(self.state, self.goto)), ("Live Recording", LiveRecordingPage(self.state, self.goto)),
                 ("Enhancement", EnhancementPage(self.state)), ("Analytics", AnalyticsPage(self.state)), ("Settings", SettingsPage(self.state))]
        icons = ["▣", "●", "⚡", "▦", "⚙"]
        for i, (name, page) in enumerate(pages):
            b = QPushButton(f"{icons[i]}   {name}"); b.setObjectName("nav"); b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=i: self.goto(i)); self.group.addButton(b, i); sl.addWidget(b); self.stack.addWidget(page)
        sl.addStretch(); foot = QLabel("Offline · local processing\nNot for field use"); foot.setObjectName("muted"); sl.addWidget(foot)
        h.addWidget(side); h.addWidget(self.stack, 1); self.goto(0)
        self.statusBar().showMessage("Ready")

    def goto(self, i):
        self.stack.setCurrentIndex(i); self.group.button(i).setChecked(True)


if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyleSheet(QSS)
    w = MainWindow(); w.show(); sys.exit(app.exec())
