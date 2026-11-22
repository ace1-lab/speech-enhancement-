C = dict(bg="#0B0F14", panel="#121923", panel2="#182231", border="#243246", text="#E6EDF3",
         muted="#8B9BB0", accent="#3DDC97", warn="#FFB547", danger="#FF5C5C", blue="#4C9AFF")

QSS = f"""
* {{ font-family: 'Segoe UI','Inter','Helvetica Neue',sans-serif; font-size: 13px; color:{C['text']}; }}
QMainWindow, QWidget#page {{ background:{C['bg']}; }}
QWidget#sidebar {{ background:{C['panel']}; border-right:1px solid {C['border']}; }}
QLabel#brand {{ font-size:16px; font-weight:700; color:{C['accent']}; letter-spacing:1px; }}
QLabel#h1 {{ font-size:22px; font-weight:700; }}
QLabel#sub, QLabel#muted {{ color:{C['muted']}; }}
QLabel#cardval {{ font-size:24px; font-weight:700; }}
QLabel#cardlbl {{ color:{C['muted']}; font-size:11px; letter-spacing:1px; }}
QFrame#card {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:12px; }}
QPushButton#nav {{ text-align:left; padding:11px 16px; border:none; border-radius:8px; color:{C['muted']}; background:transparent; }}
QPushButton#nav:hover {{ background:{C['panel2']}; color:{C['text']}; }}
QPushButton#nav:checked {{ background:{C['panel2']}; color:{C['accent']}; font-weight:600; }}
QPushButton {{ background:{C['panel2']}; border:1px solid {C['border']}; border-radius:8px; padding:8px 16px; }}
QPushButton:hover {{ border-color:{C['accent']}; }}
QPushButton:disabled {{ color:#55657a; }}
QPushButton#primary {{ background:{C['accent']}; color:#04130C; font-weight:700; border:none; }}
QPushButton#primary:hover {{ background:#5BEBAB; }}
QPushButton#danger {{ background:{C['danger']}; color:white; border:none; font-weight:700; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background:{C['panel2']}; border:1px solid {C['border']}; border-radius:6px; padding:6px 8px; }}
QComboBox QAbstractItemView {{ background:{C['panel2']}; selection-background-color:{C['border']}; }}
QCheckBox::indicator {{ width:16px; height:16px; }}
QTableWidget {{ background:{C['panel']}; gridline-color:{C['border']}; border:1px solid {C['border']}; border-radius:8px; }}
QHeaderView::section {{ background:{C['panel2']}; border:none; padding:6px; font-weight:600; }}
QProgressBar {{ background:{C['panel2']}; border:none; border-radius:5px; height:10px; text-align:center; }}
QProgressBar::chunk {{ background:{C['accent']}; border-radius:5px; }}
QSlider::groove:horizontal {{ height:6px; background:{C['panel2']}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:{C['accent']}; width:14px; margin:-5px 0; border-radius:7px; }}
QScrollArea {{ border:none; }}
QStatusBar {{ background:{C['panel']}; color:{C['muted']}; }}
"""
