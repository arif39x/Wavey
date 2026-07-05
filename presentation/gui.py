import numpy as np
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QElapsedTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QStatusBar, QFrame,
    QCheckBox, QSlider,
)

from processing.matrix_math import GRID_SIZE, Obstacle, estimate_router_position
from processing.heatmap import gaussian_blob_heatmap
from presentation.vispy_canvas import HeatmapCanvas
from theme import QT_STYLESHEET

# ── Smooth exponential lerp ──────────────────────────────────────────────

class AnimationEngine:
    def __init__(self, speed: float = 3.0):
        self._speed = speed
        self._current: np.ndarray | None = None
        self._target: np.ndarray | None = None

    def set_target(self, heatmap: np.ndarray):
        self._target = heatmap.copy()
        if self._current is None:
            self._current = heatmap.copy()

    @property
    def has_target(self) -> bool:
        return self._target is not None

    def update(self, delta: float):
        if self._current is None or self._target is None:
            return
        t = min(1.0, delta * self._speed)
        np.add(self._current, (self._target - self._current) * t, out=self._current)

    def current(self) -> np.ndarray:
        if self._current is None:
            return np.full((GRID_SIZE, GRID_SIZE), -90.0, dtype=np.float64)
        return self._current


# ── Performance monitor ─────────────────────────────────────────────────

class PerformanceMonitor:
    def __init__(self, window: int = 20):
        self._times: list[float] = []
        self._window = window

    def record(self, elapsed: float):
        self._times.append(elapsed)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if not self._times:
            return 60.0
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 60.0


# ── Signal history ring buffer ──────────────────────────────────────────

class SignalHistoryBuffer:
    def __init__(self, maxlen: int = 100):
        self._buf: list[float | None] = [None] * maxlen
        self._max = maxlen
        self._pos = 0
        self._count = 0

    def push(self, value: float):
        self._buf[self._pos] = value
        self._pos = (self._pos + 1) % self._max
        self._count = min(self._count + 1, self._max)

    def get_all(self) -> list[float | None]:
        if self._count < self._max:
            return self._buf[: self._count]
        return self._buf[self._pos :] + self._buf[: self._pos]

    def clear(self):
        self._buf = [None] * self._max
        self._pos = 0
        self._count = 0


# ── Background heatmap computation ──────────────────────────────────────

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

            heatmap = gaussian_blob_heatmap(measured, router_xy, obstacles)
            self.result_ready.emit(heatmap)

            if not self._pending:
                break
        self._running = False


# ── Main window ─────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavey — 3D Wi-Fi Signal Mapper")
        self.setMinimumSize(1200, 800)

        self._router_xy: tuple[float, float] | None = None
        self._router_estimated = False
        self._obstacles: list[Obstacle] = []
        self._measured_points: list[tuple[float, float, float]] = []
        self._current_rssi: float | None = None
        self._drag_start: tuple[float, float] | None = None
        self._preview_xy: tuple[float, float] | None = None
        self._last_render_key: tuple = ()
        self._animating = False
        self._device_names: list[str] = []
        self._device_positions: list[tuple[float, float]] = []
        self._ssid: str = ""

        self._animation = AnimationEngine(speed=3.0)
        self._perf = PerformanceMonitor()
        self._history = SignalHistoryBuffer(maxlen=80)

        self._worker = HeatmapWorker()
        self._worker.result_ready.connect(self._on_heatmap_ready)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._actually_compute)

        # Animation timer
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_elapsed = QElapsedTimer()

        self.setStyleSheet(QT_STYLESHEET)
        self._build_ui()

        empty = np.full((GRID_SIZE, GRID_SIZE), -90.0, dtype=np.float64)
        self._canvas.render(empty, [], None, device_positions=[], device_names=[])

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ── Canvas ────────────────────────────────────────────────
        canvas_container = QVBoxLayout()
        self._canvas = HeatmapCanvas(parent=self)
        canvas_container.addWidget(self._canvas.native)

        # Wire interaction callbacks
        self._canvas.on_click = self._on_grid_click
        self._canvas.on_drag_start = self._on_drag_start
        self._canvas.on_drag_move = self._on_drag_move
        self._canvas.on_drag_end = self._on_drag_end

        layout.addLayout(canvas_container, stretch=1)

        # ── Keyboard (VisPy events) ───────────────────────────────
        self._canvas.canvas.events.key_press.connect(self._on_vispy_key)

        # ── Right panel ──────────────────────────────────────────
        panel = QVBoxLayout()
        panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        panel.addWidget(QLabel("Wall material:"))
        self._mat_combo = QComboBox()
        self._mat_combo.addItems(["Drywall", "Brick", "Reinforced Concrete"])
        panel.addWidget(self._mat_combo)
        panel.addSpacing(15)

        self._auto_cb = QCheckBox("Auto-sample")
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        panel.addWidget(self._auto_cb)
        panel.addSpacing(15)

        self._point_lbl = QLabel("Data points: 0")
        panel.addWidget(self._point_lbl)
        self._fps_lbl = QLabel("FPS: --")
        panel.addWidget(self._fps_lbl)
        panel.addSpacing(10)

        self._ssid_lbl = QLabel("Network: ---")
        panel.addWidget(self._ssid_lbl)
        self._devices_lbl = QLabel("Connected: ---")
        panel.addWidget(self._devices_lbl)
        self._router_lbl = QLabel("Router: ---")
        panel.addWidget(self._router_lbl)
        panel.addSpacing(15)

        frame = QFrame()
        fl = QVBoxLayout(frame)
        fl.addWidget(QLabel("Status"))
        self._rssi_lbl = QLabel("RSSI: --- dBm")
        fl.addWidget(self._rssi_lbl)
        self._source_lbl = QLabel("Source: ---")
        fl.addWidget(self._source_lbl)
        panel.addWidget(frame)

        lerp_frame = QFrame()
        ll = QVBoxLayout(lerp_frame)
        ll.addWidget(QLabel("Smoothness"))
        self._lerp_slider = QSlider(Qt.Orientation.Horizontal)
        self._lerp_slider.setMinimum(1)
        self._lerp_slider.setMaximum(10)
        self._lerp_slider.setValue(4)
        self._lerp_slider.valueChanged.connect(
            lambda v: setattr(self._animation, "_speed", v * 0.8)
        )
        ll.addWidget(self._lerp_slider)
        panel.addWidget(lerp_frame)

        panel.addSpacing(10)
        demo_btn = QPushButton("Demo Data (20 pts)")
        demo_btn.clicked.connect(self._on_demo)
        panel.addWidget(demo_btn)
        panel.addSpacing(5)
        reset_btn = QPushButton("Clear All Data")
        reset_btn.clicked.connect(self._on_reset)
        panel.addWidget(reset_btn)
        panel.addStretch()
        layout.addLayout(panel, stretch=0)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Click = record signal  |  Drag = draw wall  |  Arrows = rotate  |  RSSI updates live"
        )

    # ── Keyboard ──────────────────────────────────────────────────────

    def _on_vispy_key(self, event):
        step = 5
        mapping = {
            "Up":    (step, 0),
            "Down":  (-step, 0),
            "Left":  (0, -step),
            "Right": (0, step),
        }
        if event.key in mapping:
            de, da = mapping[event.key]
            self._canvas.rotate(de, da)

    # ── Mouse interaction ─────────────────────────────────────────────

    def _get_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = max(0, min(GRID_SIZE - 1, int(round(x))))
        iy = max(0, min(GRID_SIZE - 1, int(round(y))))
        return ix, iy

    def _on_grid_click(self, x: float, y: float):
        ix, iy = self._get_grid(x, y)
        self._add_point(float(ix), float(iy))

    def _on_drag_start(self, x: float, y: float):
        ix, iy = self._get_grid(x, y)
        self._drag_start = (float(ix), float(iy))

    def _on_drag_move(self, x: float, y: float):
        if self._drag_start is None:
            return
        self._preview_xy = (float(x), float(y))

    def _on_drag_end(self, x: float, y: float):
        if self._drag_start is None:
            return
        sx, sy = self._drag_start
        self._drag_start = None
        self._preview_xy = None
        ix, iy = self._get_grid(x, y)
        dx = abs(ix - sx)
        dy = abs(iy - sy)
        if dx > 1 or dy > 1:
            self._add_wall(sx, sy, float(ix), float(iy))

    # ── Add data point ────────────────────────────────────────────────

    def _add_point(self, x: float, y: float, quiet: bool = False):
        if self._current_rssi is None:
            if not quiet:
                self._status.showMessage("No RSSI reading yet — wait...")
            return
        self._measured_points.append((x, y, self._current_rssi))
        self._last_click_xy = (x, y)
        self._point_lbl.setText(f"Data points: {len(self._measured_points)}")
        self._update_router()
        if not quiet:
            self._status.showMessage(
                f"Recorded at ({x:.0f}, {y:.0f}): {self._current_rssi:.1f} dBm"
            )
        self._compute_now()

    # ── Add wall ──────────────────────────────────────────────────────

    def _add_wall(self, x1: float, y1: float, x2: float, y2: float):
        mat = self._mat_combo.currentText()
        self._obstacles.append(Obstacle(x1, y1, x2, y2, mat))
        self._status.showMessage(
            f"Wall: ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f}) [{mat}]"
        )
        self._request_compute()

    # ── Router ────────────────────────────────────────────────────────

    def _update_router(self):
        if not self._measured_points:
            return
        est = estimate_router_position(self._measured_points)
        if est is not None:
            self._router_xy = est
            self._router_estimated = True
            self._router_lbl.setText(f"Router: ({est[0]:.0f}, {est[1]:.0f})")
        self._update_device_positions()

    # ── Info from networking ──────────────────────────────────────────

    def update_ssid(self, ssid: str):
        self._ssid = ssid
        self._ssid_lbl.setText(f"Network: {ssid}")

    def update_devices(self, device_names: list[str]):
        self._device_names = device_names
        count = len(device_names)
        self._devices_lbl.setText(
            f"Connected: {count} device{'s' if count != 1 else ''}"
        )
        if count > 0:
            names_text = ", ".join(device_names[:5])
            if count > 5:
                names_text += f" +{count - 5} more"
            self._devices_lbl.setToolTip(names_text)
        self._update_device_positions()

    def _update_device_positions(self):
        """Generate simulated device positions clustered around router."""
        import random
        count = len(self._device_names)
        if count < 1 or self._router_xy is None:
            self._device_positions = []
            return
        rx, ry = self._router_xy
        positions = []
        for _ in range(count):
            angle = random.uniform(0, 2 * 3.14159)
            dist = random.expovariate(0.08)
            dist = min(dist, 40.0)
            x = max(0, min(GRID_SIZE - 1, rx + dist * np.cos(angle)))
            y = max(0, min(GRID_SIZE - 1, ry + dist * np.sin(angle)))
            positions.append((float(x), float(y)))
        self._device_positions = positions

    # ── RSSI ──────────────────────────────────────────────────────────

    def update_rssi(self, rssi: float, source: str):
        changed = self._current_rssi is None or abs(rssi - self._current_rssi) > 1.0
        self._current_rssi = rssi
        self._rssi_lbl.setText(f"RSSI: {rssi:.1f} dBm")
        self._history.push(rssi)
        self._canvas.update_waterfall(rssi)
        if changed:
            # Real-time: update last measured point's RSSI with live reading
            if self._measured_points:
                x, y, _ = self._measured_points[-1]
                self._measured_points[-1] = (x, y, rssi)
                self._update_router()
                self._actually_compute()
            elif self._router_xy is not None:
                self._actually_compute()

    def update_source(self, desc: str):
        self._source_lbl.setText(f"Source: {desc}")

    # ── Compute ───────────────────────────────────────────────────────

    def _request_compute(self):
        self._debounce.start()

    def _compute_now(self):
        if self._router_xy is None and not self._measured_points:
            return
        self._worker.request(self._router_xy, self._obstacles, self._measured_points)

    def _actually_compute(self):
        if self._router_xy is None and not self._measured_points:
            return
        self._worker.request(self._router_xy, self._obstacles, self._measured_points)

    def _on_heatmap_ready(self, heatmap: np.ndarray):
        n = len(self._measured_points)
        key = (n, len(self._obstacles), self._current_rssi, self._router_xy)
        if key == self._last_render_key:
            return
        self._last_render_key = key

        self._animation.set_target(heatmap)

        self._canvas.render(
            heatmap, self._obstacles, self._router_xy,
            device_positions=self._device_positions,
            device_names=self._device_names,
        )

        if not self._animating:
            self._animating = True
            self._anim_elapsed.start()
            self._anim_timer.start()

    # ── Animation tick ────────────────────────────────────────────────

    def _on_animation_tick(self):
        delta = self._anim_elapsed.elapsed() / 1000.0
        self._anim_elapsed.restart()
        self._animation.update(delta)

        fps = self._perf.fps
        self._fps_lbl.setText(f"FPS: {fps:.0f}")

    # ── Auto-sample ──────────────────────────────────────────────────

    def _on_auto_toggled(self, checked: bool):
        if checked:
            self._auto_timer = QTimer()
            self._auto_timer.setInterval(1000)
            self._auto_timer.timeout.connect(self._record_auto)
            self._auto_timer.start()
        elif hasattr(self, "_auto_timer"):
            self._auto_timer.stop()
            del self._auto_timer

    def _record_auto(self):
        import random
        if hasattr(self, "_last_click_xy"):
            lx, ly = self._last_click_xy
        else:
            lx = ly = GRID_SIZE / 2.0
        nx = max(0.0, min(float(GRID_SIZE - 1), lx + random.uniform(-5, 5)))
        ny = max(0.0, min(float(GRID_SIZE - 1), ly + random.uniform(-5, 5)))
        self._add_point(nx, ny, quiet=True)

    # ── Demo data ─────────────────────────────────────────────────────

    def _on_demo(self):
        import random
        self._on_reset()
        rx, ry = GRID_SIZE * 0.3, GRID_SIZE * 0.4  # simulated router position
        for _ in range(20):
            # Points concentrated near the router with signal falling off
            angle = random.uniform(0, 2 * 3.14159)
            dist = random.uniform(1, 35)
            x = max(0, min(GRID_SIZE - 1, rx + dist * np.cos(angle)))
            y = max(0, min(GRID_SIZE - 1, ry + dist * np.sin(angle)))
            # Simulate RSSI: strongest near router, fading with distance
            rssi = max(-85, min(-35, -30 - 10 * np.log10(max(1, dist * 0.5))))
            self._measured_points.append((float(x), float(y), rssi))
        self._point_lbl.setText(f"Data points: {len(self._measured_points)}")
        self._update_router()
        self._status.showMessage(
            f"Loaded {len(self._measured_points)} demo points → router auto-detected"
        )
        self._compute_now()

    # ── Reset ─────────────────────────────────────────────────────────

    def _on_reset(self):
        if hasattr(self, "_auto_timer"):
            self._auto_timer.stop()
            del self._auto_timer
            self._auto_cb.setChecked(False)
        self._router_xy = None
        self._router_estimated = False
        self._obstacles.clear()
        self._measured_points.clear()
        self._last_render_key = ()
        self._history.clear()
        self._point_lbl.setText("Data points: 0")
        self._router_lbl.setText("Router: ---")
        self._device_names = []
        self._device_positions = []
        empty = np.full((GRID_SIZE, GRID_SIZE), -90.0, dtype=np.float64)
        self._canvas.render(empty, [], None, device_positions=[], device_names=[])
        self._status.showMessage("All data cleared.")

    # ── Cleanup ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        if hasattr(self, "_auto_timer"):
            self._auto_timer.stop()
        self._anim_timer.stop()
        self._worker.quit()
        self._worker.wait()
        super().closeEvent(event)
