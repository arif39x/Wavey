from vispy import scene
from vispy.scene import visuals
from vispy.color import Colormap
from vispy.geometry import create_grid_mesh
from vispy.visuals.transforms import STTransform
import numpy as np

from processing.matrix_math import GRID_SIZE, Obstacle
from processing.colormap import SIGNAL_CMAP_COLORS, SIGNAL_CMAP_CONTROLS

SIGNAL_VISPY_CMAP = Colormap(
    SIGNAL_CMAP_COLORS,
    controls=SIGNAL_CMAP_CONTROLS,
)

FLOOR_Z = -90.0
CEILING_Z = -28.0
BG_HEX = "#0d0e1a"


class HeatmapCanvas:

    WALL_COLORS: dict[str, tuple[float, float, float, float]] = {
        "Drywall":            (0.545, 0.451, 0.333, 0.30),
        "Brick":              (0.804, 0.361, 0.361, 0.30),
        "Reinforced Concrete":(0.333, 0.333, 0.333, 0.30),
    }

    def __init__(self, parent=None):
        self.canvas = scene.SceneCanvas(
            keys=None, size=(1200, 800), bgcolor=BG_HEX,
            parent=parent, dpi=96,
        )
        self.native = self.canvas.native

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.button = "right"
        self.view.camera.distance = float(GRID_SIZE) * 1.6
        self.view.camera.center = (GRID_SIZE // 2, GRID_SIZE // 2, -60)
        self.view.camera.fov = 45

        self._floor_grid: visuals.Line | None = None
        self._build_floor_grid()

        self._surf: visuals.Mesh | None = None
        self._wall_nodes: list = []

        self._router_glow = visuals.Line()
        self.view.add(self._router_glow)

        self._device_markers = visuals.Markers()
        self.view.add(self._device_markers)
        self._device_labels: list[visuals.Text] = []

        self._wf_img: visuals.Image | None = None
        self._wf_arr: np.ndarray | None = None
        self._wf_size = 80

        self._elev: float = 25.0
        self._azim: float = -65.0
        self._drag_origin: tuple[float, float] | None = None
        self._dragging: bool = False
        self._surf_X: np.ndarray | None = None
        self._surf_Y: np.ndarray | None = None
        self._surf_faces: np.ndarray | None = None

        self.on_click: callable = lambda x, y: None
        self.on_drag_start: callable = lambda x, y: None
        self.on_drag_move: callable = lambda x, y: None
        self.on_drag_end: callable = lambda x, y: None

        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)

    def _build_floor_grid(self):
        segs: list = []
        for i in range(0, GRID_SIZE + 1, 10):
            major = i % 20 == 0
            alpha = 0.30 if major else 0.15
            c = (0.165, 0.176, 0.271, alpha)
            segs.append((i, 0, FLOOR_Z, i, GRID_SIZE, FLOOR_Z, c))
            segs.append((0, i, FLOOR_Z, GRID_SIZE, i, FLOOR_Z, c))

        pos = []
        col_arr = []
        for x1, y1, z1, x2, y2, z2, cc in segs:
            pos.extend([(x1, y1, z1), (x2, y2, z2)])
            col_arr.extend([cc, cc])

        self._floor_grid = visuals.Line(
            pos=np.array(pos, dtype=np.float32),
            color=np.array(col_arr, dtype=np.float32),
            connect="segments",
            parent=self.view.scene,
        )

    def pixel_to_scene(self, px: float, py: float) -> tuple[float, float] | None:
        w, h = self.canvas.size
        ndc_x = (px / w) * 2.0 - 1.0
        ndc_y = 1.0 - (py / h) * 2.0

        tr = self.view.camera.transform
        if tr is None:
            return None
        try:
            near = tr.imap([ndc_x, ndc_y, -1, 1])
            far = tr.imap([ndc_x, ndc_y, 1, 1])
        except Exception:
            return None
        if near is None or far is None:
            return None

        n = np.asarray(near[:3], dtype=np.float64)
        f = np.asarray(far[:3], dtype=np.float64)
        d = f - n
        if abs(d[2]) < 1e-12:
            return None
        t = (FLOOR_Z - n[2]) / d[2]
        x = n[0] + t * d[0]
        y = n[1] + t * d[1]

        if not (0 <= x <= GRID_SIZE and 0 <= y <= GRID_SIZE):
            return None
        return float(x), float(y)

    def _on_mouse_press(self, event):
        if event.button == 1:
            self._dragging = False
            xy = self.pixel_to_scene(event.pos[0], event.pos[1])
            if xy is not None:
                self._drag_origin = xy
                self.on_drag_start(*xy)

    def _on_mouse_move(self, event):
        if event.button == 1 and self._drag_origin is not None:
            xy = self.pixel_to_scene(event.pos[0], event.pos[1])
            if xy is not None:
                self._dragging = True
                self.on_drag_move(*xy)

    def _on_mouse_release(self, event):
        if event.button == 1 and self._drag_origin is not None:
            xy = self.pixel_to_scene(event.pos[0], event.pos[1])
            self._drag_origin = None
            if xy is not None:
                self.on_drag_end(*xy)

    def render(
        self,
        heatmap: np.ndarray,
        obstacles: list[Obstacle],
        router_xy: tuple[float, float] | None,
        device_positions: list[tuple[float, float]] | None = None,
        device_names: list[str] | None = None,
    ):
        Z = heatmap.astype(np.float32)

        self._update_surface(Z)
        self._update_walls(obstacles)
        self._update_router(router_xy, Z)
        self._update_device_markers(device_positions, device_names, Z)
        self.canvas.update()

        self._surf_X = self._surf_Y = self._surf_faces = None

    def _update_surface(self, Z: np.ndarray):
        if self._surf_X is None:
            xs = np.arange(GRID_SIZE, dtype=np.float32)
            ys = np.arange(GRID_SIZE, dtype=np.float32)
            self._surf_X, self._surf_Y = np.meshgrid(xs, ys)
        X, Y = self._surf_X, self._surf_Y

        if self._surf_faces is None:
            _, self._surf_faces = create_grid_mesh(X, Y, Z)
        faces = self._surf_faces

        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        norm = np.clip((Z - (-90.0)) / 60.0, 0.0, 1.0)
        colors = SIGNAL_VISPY_CMAP[norm.ravel()].rgba.copy()
        colors[:, 3] = 1.0

        if self._surf is None:
            self._surf = visuals.Mesh(
                vertices=verts, faces=faces,
                vertex_colors=colors, shading="smooth",
                parent=self.view.scene,
            )
        else:
            self._surf.set_data(
                vertices=verts, faces=faces,
                vertex_colors=colors,
            )

    def _update_walls(self, obstacles: list[Obstacle]):
        for child in self._wall_nodes:
            child.parent = None
        self._wall_nodes.clear()

        for obs in obstacles:
            color = self.WALL_COLORS.get(
                obs.material, (0.333, 0.333, 0.333, 0.30)
            )
            x1, y1, x2, y2 = obs.x1, obs.y1, obs.x2, obs.y2

            verts = np.array(
                [
                    [x1, y1, FLOOR_Z],
                    [x2, y2, FLOOR_Z],
                    [x2, y2, CEILING_Z],
                    [x1, y1, CEILING_Z],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

            wall = visuals.Mesh(
                vertices=verts, faces=faces,
                color=color, shading="smooth",
                parent=self.view.scene,
            )
            self._wall_nodes.append(wall)

            ol = visuals.Line(
                pos=np.array(
                    [[x1, y1, FLOOR_Z], [x2, y2, FLOOR_Z]], dtype=np.float32
                ),
                color=color[:3] + (0.6,),
                width=2.5,
                parent=self.view.scene,
            )
            self._wall_nodes.append(ol)

    def _update_router(
        self,
        router_xy: tuple[float, float] | None,
        Z: np.ndarray,
    ):
        if router_xy is None:
            self._router_glow.visible = False
            return

        rx, ry = router_xy
        rz = float(Z[int(round(ry)), int(round(rx))])

        phi = np.linspace(0, 2 * np.pi, 36)
        glow_pos = []
        glow_col = []
        for r, a in [(5, 0.12), (3, 0.20)]:
            for p in phi:
                glow_pos.append([rx + r * np.cos(p), ry + r * np.sin(p), rz])
                glow_col.append([0.49, 0.80, 1.0, a])
        self._router_glow.visible = True
        self._router_glow.set_data(
            pos=np.array(glow_pos, dtype=np.float32),
            color=np.array(glow_col, dtype=np.float32),
            connect="strip",
            width=1.2,
        )

    def _update_device_markers(
        self,
        device_positions: list[tuple[float, float]] | None,
        device_names: list[str] | None,
        Z: np.ndarray,
    ):
        for lbl in self._device_labels:
            lbl.parent = None
        self._device_labels.clear()

        if device_positions and len(device_positions) > 0:
            self._device_markers.visible = True
            xs = np.array([p[0] for p in device_positions], dtype=np.float32)
            ys = np.array([p[1] for p in device_positions], dtype=np.float32)
            zs = np.array([
                Z[int(round(p[1])), int(round(p[0]))]
                for p in device_positions
            ], dtype=np.float32)
            self._device_markers.set_data(
                pos=np.column_stack([xs, ys, zs]),
                face_color="#e0af68",
                edge_color="#1a1b2e",
                size=10,
                symbol="disc",
            )
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                name = device_names[i] if device_names and i < len(device_names) else ""
                if name:
                    lbl = visuals.Text(
                        name,
                        pos=(x, y, z + 3),
                        color="#e0af68",
                        font_size=8,
                        bold=False,
                        parent=self.view.scene,
                        anchor_x="center",
                        anchor_y="bottom",
                    )
                    self._device_labels.append(lbl)
        else:
            self._device_markers.visible = False

    def update_waterfall(self, rssi: float):
        if self._wf_arr is None:
            self._wf_arr = np.full(self._wf_size, -90.0, dtype=np.float32)
        self._wf_arr = np.roll(self._wf_arr, -1)
        self._wf_arr[-1] = rssi

        if self._wf_img is None:
            self._wf_img = scene.visuals.Image(
                self._wf_arr[np.newaxis, :],
                cmap=SIGNAL_VISPY_CMAP,
                parent=self.canvas.scene,
            )
            self._wf_img.transform = STTransform(
                translate=(self.canvas.size[0] - 200, 20, 0)
            )
            self._wf_img.interpolation = "nearest"
            self._wf_img.clim = (-90.0, -30.0)
        else:
            self._wf_img.set_data(self._wf_arr[np.newaxis, :])

    def rotate(self, elev_delta: float = 0, azim_delta: float = 0):
        cam = self.view.camera
        cam.elevation = max(5, min(85, cam.elevation + elev_delta))
        cam.azimuth = (cam.azimuth + azim_delta) % 360.0
        self.canvas.update()
