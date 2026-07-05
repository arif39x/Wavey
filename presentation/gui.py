"""
Pipeline 3: PyQt6 GUI with embedded matplotlib canvas.

Renders the heatmap, handles user interactions (draw walls,
add data points in live scan), and manages the worker thread
for background computation.

Router position is auto-estimated from collected signal data —
no manual placement needed.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QStatusBar, QFrame, QButtonGroup,
    QRadioButton, QCheckBox,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
import numpy as np

from processing.matrix_math import (
    GRID_SIZE, Obstacle, compute_heatmap, estimate_router_position,
)


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

MODE_PAN = "Pan/View"
MODE_WALL = "Draw Wall"
MODE_DATA = "Add Data Point"
MODE_SCAN = "Live Scan"


# ---------------------------------------------------------------------------
# Matplotlib canvas embedded in Qt
# ---------------------------------------------------------------------------

class HeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

        self._norm = Normalize(vmin=-90.0, vmax=-30.0)

    def render(
        self,
        heatmap: np.ndarray,
        obstacles: list[Obstacle],
        router_xy: tuple[float, float] | None,
        data_points: list[tuple[float, float]],
        router_estimated: bool = False,
    ):
        self.ax.clear()

        extent = [0, GRID_SIZE, GRID_SIZE, 0]
        self.ax.imshow(
            heatmap, cmap="turbo", norm=self._norm,
            extent=extent, aspect="equal", interpolation="bilinear",
        )

        material_colors = {
            "Drywall": "#8B7355",
            "Brick": "#CD5C5C",
            "Reinforced Concrete": "#555555",
        }
        for obs in obstacles:
            color = material_colors.get(obs.material, "#000000")
            self.ax.plot(
                [obs.x1, obs.x2], [obs.y1, obs.y2],
                color=color, linewidth=3, solid_capstyle="round",
            )

        if router_xy is not None:
            label = "Router (estimated)" if router_estimated else "Router"
            self.ax.scatter(
                [router_xy[0]], [router_xy[1]],
                marker="D", s=200, color="#00FF00", edgecolors="black",
                zorder=5, label=label,
            )

        if data_points:
            xs = [p[0] for p in data_points]
            ys = [p[1] for p in data_points]
            self.ax.scatter(
                xs, ys, marker="o", s=40, color="#FFFFFF",
                edgecolors="black", linewidths=0.5,
                zorder=4, label="Measurements",
            )

        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(GRID_SIZE, 0)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.3)
        self.ax.set_xlabel("Grid X")
        self.ax.set_ylabel("Grid Y")
        self.ax.set_title("Wi-Fi Signal Strength Heatmap")

        if router_xy or data_points:
            self.ax.legend(fontsize=8, loc="lower right")

        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------------
# Heatmap computation worker thread
# ---------------------------------------------------------------------------

class HeatmapWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._router_xy: tuple[float, float] | None = None
        self._obstacles: list[Obstacle] = []
        self._measured_points: list[tuple[float, float, float]] = []
        self._pending = False
        self._running = False

    def request(
        self,
        router_xy: tuple[float, float] | None,
        obstacles: list[Obstacle],
        measured_points: list[tuple[float, float, float]],
    ):
        self._router_xy = router_xy
        self._obstacles = list(obstacles)
        self._measured_points = list(measured_points)
        if self._running:
            self._pending = True
        else:
            self.start()

    def run(self):
        self._running = True
        while True:
            router_xy = self._router_xy
            obstacles = self._obstacles
            measured = self._measured_points
            self._pending = False

            if router_xy is not None and measured:
                heatmap = compute_heatmap(router_xy, obstacles, measured)
            elif router_xy is not None:
                heatmap = compute_heatmap(router_xy, obstacles, [])
            else:
                heatmap = np.full((GRID_SIZE, GRID_SIZE), -90.0, dtype=np.float64)

            self.result_ready.emit(heatmap)

            if not self._pending:
                break
        self._running = False


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wi-Fi Signal Heatmapper")
        self.setMinimumSize(1000, 750)

        self._router_xy: tuple[float, float] | None = None
        self._router_estimated = False
        self._obstacles: list[Obstacle] = []
        self._measured_points: list[tuple[float, float, float]] = []
        self._current_rssi: float = -90.0
        self._rssi_source: str = "Initializing..."
        self._drag_start: tuple[float, float] | None = None
        self._mode = MODE_PAN

        self._heatmap_worker = HeatmapWorker()
        self._heatmap_worker.result_ready.connect(self._on_heatmap_ready)

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._actually_compute)

        self._build_ui()

        empty = np.full((GRID_SIZE, GRID_SIZE), -90.0, dtype=np.float64)
        self._canvas.render(empty, [], None, [])

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        canvas_container = QVBoxLayout()
        self._canvas = HeatmapCanvas()
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        canvas_container.addWidget(self._canvas)
        main_layout.addLayout(canvas_container, stretch=1)

        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ---- Mode selection ----
        right_panel.addWidget(QLabel("Mode:"))
        mode_group = QButtonGroup(self)
        modes = [MODE_PAN, MODE_WALL, MODE_DATA, MODE_SCAN]
        for text in modes:
            rb = QRadioButton(text)
            mode_group.addButton(rb)
            right_panel.addWidget(rb)
            if text == MODE_PAN:
                rb.setChecked(True)
            rb.toggled.connect(self._on_mode_changed)

        right_panel.addSpacing(10)

        # ---- Continuous recording toggle ----
        self._continuous_cb = QCheckBox("Continuous record")
        self._continuous_cb.setChecked(False)
        self._continuous_cb.toggled.connect(self._on_continuous_toggled)
        right_panel.addWidget(self._continuous_cb)

        self._auto_router_cb = QCheckBox("Auto-detect router")
        self._auto_router_cb.setChecked(True)
        right_panel.addWidget(self._auto_router_cb)

        right_panel.addSpacing(15)

        # ---- Wall material ----
        right_panel.addWidget(QLabel("Wall material:"))
        self._material_combo = QComboBox()
        self._material_combo.addItems(["Drywall", "Brick", "Reinforced Concrete"])
        right_panel.addWidget(self._material_combo)

        right_panel.addSpacing(15)

        # ---- Recorded data count ----
        self._point_count_label = QLabel("Data points: 0")
        right_panel.addWidget(self._point_count_label)

        right_panel.addSpacing(15)

        # ---- Status panel ----
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        status_layout = QVBoxLayout(status_frame)
        status_layout.addWidget(QLabel("Status"))
        self._rssi_label = QLabel("RSSI: --- dBm")
        status_layout.addWidget(self._rssi_label)
        self._source_label = QLabel("Source: ---")
        status_layout.addWidget(self._source_label)
        self._mode_label = QLabel("Mode: Pan/View")
        status_layout.addWidget(self._mode_label)
        right_panel.addWidget(status_frame)

        right_panel.addSpacing(15)
        reset_btn = QPushButton("Clear All Data")
        reset_btn.clicked.connect(self._on_reset)
        right_panel.addWidget(reset_btn)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=0)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Add data points to auto-detect router and build heatmap.")

    # ---- Continuous recording timer ----

    def _on_continuous_toggled(self, checked: bool):
        if checked:
            self._continuous_timer = QTimer()
            self._continuous_timer.setInterval(1000)
            self._continuous_timer.timeout.connect(self._record_continuous_point)
            self._continuous_timer.start()
        else:
            if hasattr(self, "_continuous_timer"):
                self._continuous_timer.stop()
                del self._continuous_timer

    def _record_continuous_point(self):
        """In continuous mode, record a point at a small random offset
        from the last clicked position to simulate walking around."""
        if self._mode != MODE_SCAN or not hasattr(self, "_last_click_pos"):
            return
        lx, ly = self._last_click_pos
        import random
        jitter = 3.0
        nx = max(0, min(GRID_SIZE - 1, lx + random.uniform(-jitter, jitter)))
        ny = max(0, min(GRID_SIZE - 1, ly + random.uniform(-jitter, jitter)))
        self._measured_points.append((nx, ny, self._current_rssi))
        self._update_router()
        self._point_count_label.setText(f"Data points: {len(self._measured_points)}")
        self._status_bar.showMessage(
            f"Auto-recorded at ({nx:.0f}, {ny:.0f}): {self._current_rssi:.1f} dBm"
        )
        self._compute_now()

    # ---- Mode switching ----

    def _on_mode_changed(self):
        sender = self.sender()
        if sender.isChecked():
            self._mode = sender.text()
            self._mode_label.setText(f"Mode: {self._mode}")
            self._status_bar.showMessage(f"Mode: {self._mode}")

    def _get_grid_coords(self, xdata, ydata):
        if xdata is None or ydata is None:
            return None, None
        ix = int(round(xdata))
        iy = int(round(ydata))
        ix = max(0, min(GRID_SIZE - 1, ix))
        iy = max(0, min(GRID_SIZE - 1, iy))
        return ix, iy

    # ---- Mouse handling ----

    def _on_click(self, event):
        if event.inaxes is None:
            return
        ix, iy = self._get_grid_coords(event.xdata, event.ydata)

        if self._mode == MODE_PAN:
            return

        elif self._mode == MODE_WALL:
            self._drag_start = (float(ix), float(iy))

        elif self._mode == MODE_DATA:
            self._add_data_point(ix, iy)

        elif self._mode == MODE_SCAN:
            self._last_click_pos = (float(ix), float(iy))
            self._add_data_point(ix, iy)

    def _add_data_point(self, ix: int, iy: int):
        if self._current_rssi is not None:
            self._measured_points.append(
                (float(ix), float(iy), self._current_rssi)
            )
            self._point_count_label.setText(
                f"Data points: {len(self._measured_points)}"
            )
            self._status_bar.showMessage(
                f"Recorded at ({ix}, {iy}): {self._current_rssi:.1f} dBm"
            )
            self._update_router()
            self._compute_now()
        else:
            self._status_bar.showMessage(
                "No RSSI reading available yet — wait for data..."
            )

    def _update_router(self):
        if self._auto_router_cb.isChecked() and self._measured_points:
            est = estimate_router_position(self._measured_points)
            if est is not None:
                self._router_xy = est
                self._router_estimated = True

    def _on_motion(self, event):
        if self._mode != MODE_WALL or self._drag_start is None:
            return
        if event.inaxes is None:
            return

        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata

        if hasattr(self, "_preview_line") and self._preview_line in self._canvas.ax.lines:
            self._preview_line.remove()
        (self._preview_line,) = self._canvas.ax.plot(
            [x0, x1], [y0, y1],
            color="#888888", linewidth=3, linestyle="--",
        )
        self._canvas.draw_idle()

    def _on_release(self, event):
        if self._mode != MODE_WALL or self._drag_start is None:
            return
        if event.inaxes is None:
            self._drag_start = None
            return

        ix, iy = self._get_grid_coords(event.xdata, event.ydata)
        sx, sy = self._drag_start

        if hasattr(self, "_preview_line") and self._preview_line in self._canvas.ax.lines:
            self._preview_line.remove()

        if abs(ix - sx) > 0.5 or abs(iy - sy) > 0.5:
            material = self._material_combo.currentText()
            obs = Obstacle(sx, sy, float(ix), float(iy), material)
            self._obstacles.append(obs)
            self._status_bar.showMessage(
                f"Wall added: ({sx:.0f},{sy:.0f}) \u2192 ({ix},{iy}) [{material}]"
            )
            self._request_heatmap()
        else:
            self._status_bar.showMessage("Wall too short — drag at least 1 cell.")

        self._drag_start = None

    # ---- Heatmap computation ----

    def _request_heatmap(self):
        self._debounce_timer.start()

    def _compute_now(self):
        if self._router_xy is None:
            return
        self._heatmap_worker.request(
            self._router_xy, self._obstacles, self._measured_points,
        )

    def _on_heatmap_ready(self, heatmap: np.ndarray):
        data_pts_xy = [(x, y) for x, y, _ in self._measured_points]
        self._canvas.render(
            heatmap, self._obstacles, self._router_xy,
            data_pts_xy, router_estimated=self._router_estimated,
        )

    def _actually_compute(self):
        if self._router_xy is None:
            return
        self._heatmap_worker.request(
            self._router_xy, self._obstacles, self._measured_points,
        )

    # ---- RSSI updates from networking thread ----

    def update_rssi(self, rssi: float, source: str):
        self._current_rssi = rssi
        self._rssi_label.setText(f"RSSI: {rssi:.1f} dBm")

    def update_source(self, description: str):
        self._rssi_source = description
        self._source_label.setText(f"Source: {description}")

    # ---- Reset ----

    def _on_reset(self):
        if hasattr(self, "_continuous_timer"):
            self._continuous_timer.stop()
            del self._continuous_timer
            self._continuous_cb.setChecked(False)
        self._router_xy = None
        self._router_estimated = False
        self._obstacles.clear()
        self._measured_points.clear()
        self._point_count_label.setText("Data points: 0")
        empty = np.full((GRID_SIZE, GRID_SIZE), -90.0, dtype=np.float64)
        self._canvas.render(empty, [], None, [])
        self._status_bar.showMessage("All data cleared.")

    # ---- Cleanup ----

    def closeEvent(self, event):
        if hasattr(self, "_continuous_timer"):
            self._continuous_timer.stop()
        self._heatmap_worker.quit()
        self._heatmap_worker.wait()
        super().closeEvent(event)
