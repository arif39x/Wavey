BG_DARK = "#16161e"
BG = "#1a1b2e"
BG_SURFACE = "#1f2235"
BG_BORDER = "#2a2d45"
FG = "#c0caf5"
FG_MUTED = "#565f89"
FG_DIM = "#3b4261"

CYAN = "#7dcfff"
BLUE = "#7aa2f7"
PURPLE = "#bb9af7"
PINK = "#f7768e"
ORANGE = "#ff9e64"
GREEN = "#9ece6a"
YELLOW = "#e0af68"
RED = "#f7768e"

QT_STYLESHEET = f"""
QMainWindow {{
    background-color: {BG_DARK};
}}
QWidget {{
    color: {FG};
    font-size: 12px;
}}
QLabel {{
    color: {FG};
    background: transparent;
    padding: 2px 0;
}}
QPushButton {{
    background-color: {BG_BORDER};
    color: {FG};
    border: 1px solid {BG_BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #3b3f5c;
    border-color: {BLUE};
}}
QPushButton:pressed {{
    background-color: {BLUE};
    color: {BG_DARK};
}}
QComboBox {{
    background-color: {BG_SURFACE};
    color: {FG};
    border: 1px solid {BG_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}
QComboBox:hover {{
    border-color: {BLUE};
}}
QComboBox::drop-down {{
    border: none;
    background: {BG_SURFACE};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    color: {FG};
    selection-background-color: {BLUE};
    border: 1px solid {BG_BORDER};
}}
QCheckBox {{
    color: {FG};
    spacing: 6px;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BG_BORDER};
    border-radius: 3px;
    background: {BG_SURFACE};
}}
QCheckBox::indicator:checked {{
    background-color: {BLUE};
    border-color: {BLUE};
}}
QFrame {{
    background-color: {BG_SURFACE};
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    padding: 6px;
}}
QStatusBar {{
    background-color: {BG_DARK};
    color: {FG_MUTED};
    border-top: 1px solid {BG_BORDER};
    font-size: 11px;
}}
QStatusBar::item {{
    border: none;
}}
"""
