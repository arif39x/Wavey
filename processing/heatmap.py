import numpy as np

from processing.matrix_math import GRID_SIZE, compute_physics_baseline, Obstacle

SIGMA_DEFAULT = 3.0


def gaussian_blob_heatmap(
    measured_points: list[tuple[float, float, float]],
    router_xy: tuple[float, float] | None = None,
    obstacles: list[Obstacle] | None = None,
    grid_size: int = GRID_SIZE,
    sigma: float = SIGMA_DEFAULT,
) -> np.ndarray:
    if obstacles is None:
        obstacles = []

    xs = np.arange(grid_size, dtype=np.float64)
    ys = np.arange(grid_size, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    weight_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    weighted_sum = np.zeros((grid_size, grid_size), dtype=np.float64)

    for px, py, rssi in measured_points:
        px = max(0, min(grid_size - 1, px))
        py = max(0, min(grid_size - 1, py))

        norm_rssi = (rssi + 100.0) / 70.0
        norm_rssi = max(0.0, min(1.0, norm_rssi))
        sigma_eff = sigma * (1.0 + 0.8 * (1.0 - norm_rssi))

        dist2 = (X - px) ** 2 + (Y - py) ** 2
        weight = np.exp(-dist2 / (2.0 * sigma_eff ** 2))

        weight_sum += weight
        weighted_sum += weight * rssi

    mask = weight_sum > 1e-10
    heatmap = np.full((grid_size, grid_size), -90.0, dtype=np.float64)
    heatmap[mask] = weighted_sum[mask] / weight_sum[mask]

    if router_xy is not None:
        physics = compute_physics_baseline(router_xy, obstacles)
        blend = np.clip(np.maximum(0.0, 1.0 - weight_sum / 3.0), 0.0, 1.0)
        heatmap = heatmap * (1.0 - blend) + physics * blend

    return np.clip(heatmap, -90.0, -30.0)
