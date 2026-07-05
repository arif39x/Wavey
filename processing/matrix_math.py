"""
Pipeline 2: Math / Processing Engine.

Grid generation, physics-based path loss prediction (log-distance model
with wall attenuation via 2D ray tracing), and residual-correction
interpolation of real measurements.

Coordinate convention: grid indices (col, row) where col = x, row = y.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

GRID_SIZE = 100  # 100x100 grid
CELL_METERS = 0.3  # Each cell represents 0.3m (adjust for your floor plan)
TX_POWER_DBM = -30.0  # Reference Tx power at 1 cell distance
PATH_LOSS_EXPONENT = 2.4  # Indoor path loss exponent (typical 2.0-3.0)
D0 = 1.0  # Reference distance in cells

# Wall attenuation values in dB
# These are realistic reference values from RF propagation literature:
#   Drywall: ~3 dB loss per sheet at 2.4 GHz
#   Brick:   ~12 dB loss
#   Reinforced Concrete: ~25 dB loss (rebar causes significant attenuation)
WALL_ATTENUATION = {
    "Drywall": 3.0,
    "Brick": 12.0,
    "Reinforced Concrete": 25.0,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """A wall segment represented as a line with a material attenuation."""
    x1: float
    y1: float
    x2: float
    y2: float
    material: str

    @property
    def attenuation_db(self) -> float:
        return WALL_ATTENUATION.get(self.material, 0.0)


# ---------------------------------------------------------------------------
# 2D line-segment intersection (orientation / on-segment method)
#
# Uses the standard computational geometry approach:
#   orientation(p, q, r) = sign of (q.y - p.y)*(r.x - q.x) - (q.x - p.x)*(r.y - q.y)
#   Two segments (p1,q1) and (p2,q2) intersect iff:
#     orient(p1,q1,p2) * orient(p1,q1,q2) < 0  AND
#     orient(p2,q2,p1) * orient(p2,q2,q1) < 0
#   Collinear overlap is also handled.
#
# This is exact, not a bounding-box approximation.
# ---------------------------------------------------------------------------

def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)


def _on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    """Check if point c lies on segment a-b (assuming collinear)."""
    return (
        min(ax, bx) <= cx <= max(ax, bx)
        and min(ay, by) <= cy <= max(ay, by)
    )


def segments_intersect(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> bool:
    """Return True if segment (x1,y1)-(x2,y2) intersects (x3,y3)-(x4,y4)."""
    o1 = _orient(x1, y1, x2, y2, x3, y3)
    o2 = _orient(x1, y1, x2, y2, x4, y4)
    o3 = _orient(x3, y3, x4, y4, x1, y1)
    o4 = _orient(x3, y3, x4, y4, x2, y2)

    if o1 == 0 and _on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if o2 == 0 and _on_segment(x1, y1, x2, y2, x4, y4):
        return True
    if o3 == 0 and _on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if o4 == 0 and _on_segment(x3, y3, x4, y4, x2, y2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


# ---------------------------------------------------------------------------
# Physics baseline
# ---------------------------------------------------------------------------

def compute_physics_baseline(
    router_xy: tuple[float, float],
    obstacles: list[Obstacle] | None = None,
) -> np.ndarray:
    """
    Compute the physics-predicted RSSI (dBm) across the entire grid.
    
    The baseline uses:
      1) Log-distance path loss: RSSI(d) = TX_POWER_DBM - 10 * n * log10(d / D0)
         where d is Euclidean distance from the router in cells.
      2) Wall penetration loss: for each grid cell, the straight ray from
         router to cell is tested against every obstacle. The attenuation
         of every crossed wall is summed and subtracted.
    
    The log-distance model is a standard FSPL approximation for indoor
    environments. n=2.4 accounts for the extra loss over free space (n=2.0)
    caused by furniture, people, and building structure.
    """
    if obstacles is None:
        obstacles = []

    rx, ry = router_xy
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

    # Precompute distance from router to every cell (vectorized with meshgrid)
    xs = np.arange(GRID_SIZE, dtype=np.float64)
    ys = np.arange(GRID_SIZE, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dists = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2)

    # Clamp minimum distance to D0 to avoid log(0) and singularities at router
    dists = np.maximum(dists, D0)

    # Log-distance path loss (vectorized)
    baseline = TX_POWER_DBM - 10.0 * PATH_LOSS_EXPONENT * np.log10(dists / D0)

    # Wall attenuation: test every obstacle against the ray to each cell.
    # This is a loop (not vectorized), but we only iterate when obstacles exist.
    if obstacles:
        # Precompute obstacle data for faster access
        obs_data = [(o.x1, o.y1, o.x2, o.y2, o.attenuation_db) for o in obstacles]
        # Filter to obstacles with non-zero attenuation to skip no-ops
        obs_data = [od for od in obs_data if od[4] != 0.0]

        if obs_data:
            for cx in range(GRID_SIZE):
                for cy in range(GRID_SIZE):
                    total_attenuation = 0.0
                    for x1, y1, x2, y2, att in obs_data:
                        if segments_intersect(rx, ry, float(cx), float(cy), x1, y1, x2, y2):
                            total_attenuation += att
                    if total_attenuation > 0.0:
                        # Subtract wall loss from the baseline
                        # (wall loss is already a positive dB value)
                        baseline[cy, cx] -= total_attenuation

    return baseline


# ---------------------------------------------------------------------------
# Auto router position estimation
#
# Instead of requiring the user to manually place the router on the floor
# plan, we estimate its position from measured signal data. The heuristic
# is simple: the closer you are to the router, the stronger the RSSI
# (less negative dBm). We use a weighted centroid where each point's
# weight is proportional to its signal strength squared, so points with
# stronger signals pull the estimate toward the true router location.
#
# If only one point exists, we place the router at that point. With zero
# points, return None.
# ---------------------------------------------------------------------------

def estimate_router_position(
    measured_points: list[tuple[float, float, float]],
) -> tuple[float, float] | None:
    """Estimate router (x, y) from measured signal data using weighted centroid.
    
    Weight = (rssi + 100)^2, so a point at -30 dBm gets weight 4900
    while a point at -80 dBm gets weight 400. This heavily biases the
    estimate toward locations with the strongest signal.
    """
    if not measured_points:
        return None

    if len(measured_points) == 1:
        return (measured_points[0][0], measured_points[0][1])

    total_weight = 0.0
    cx = 0.0
    cy = 0.0
    for x, y, rssi in measured_points:
        weight = (rssi + 100.0) ** 2
        cx += x * weight
        cy += y * weight
        total_weight += weight

    if total_weight == 0.0:
        return (measured_points[0][0], measured_points[0][1])

    ex = cx / total_weight
    ey = cy / total_weight
    ex = max(0.0, min(float(GRID_SIZE - 1), ex))
    ey = max(0.0, min(float(GRID_SIZE - 1), ey))
    return (ex, ey)


# ---------------------------------------------------------------------------
# Residual-correction interpolation
#
# Strategy: instead of throwing away the physics model once real data exists,
# we compute residuals (measured - baseline) and interpolate those residuals
# across the grid. The final heatmap = baseline + interpolated residuals.
#
# This is a hybrid approach: the physics model handles the gross structure
# (distance decay, major walls), while interpolation captures the local
# deviations (furniture, multipath, antenna orientation, materials the user
# didn't draw).
#
# Interpolation choice: RBFInterpolator (scipy).
#   - RBF works well with small-to-medium point sets (typical use case).
#   - Supports multi-quadric kernel which handles smooth spatial variation.
#   - Ordinary Kriging (via pykrige) is also valid; RBF is chosen for zero
#     additional dependency beyond scipy. For large anisotropic datasets with
#     known variograms, Kriging would be superior.
# ---------------------------------------------------------------------------

def compute_heatmap(
    router_xy: tuple[float, float],
    obstacles: list[Obstacle] | None = None,
    measured_points: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """
    Compute the final heatmap combining physics baseline + residual correction.
    
    Args:
        router_xy: (x, y) grid position of the router.
        obstacles: list of wall segments with material.
        measured_points: list of (x, y, rssi_dbm) tuples from user clicks.
    
    Returns:
        2D numpy array (GRID_SIZE x GRID_SIZE) of RSSI values in dBm.
    """
    if measured_points is None:
        measured_points = []

    baseline = compute_physics_baseline(router_xy, obstacles)

    num_points = len(measured_points)

    if num_points == 0:
        # No data — return physics baseline only
        return np.clip(baseline, -90.0, -30.0)

    # Extract measured coordinates and values
    pts = np.array([(x, y) for x, y, _ in measured_points], dtype=np.float64)
    vals = np.array([rssi for _, _, rssi in measured_points], dtype=np.float64)

    # De-duplicate: if multiple measurements at same (x, y), average them
    unique_pts, inverse_indices = np.unique(pts, axis=0, return_inverse=True)
    if len(unique_pts) < len(pts):
        deduped_vals = np.zeros(len(unique_pts), dtype=np.float64)
        counts = np.zeros(len(unique_pts), dtype=np.int64)
        for i, idx in enumerate(inverse_indices):
            deduped_vals[idx] += vals[i]
            counts[idx] += 1
        vals = deduped_vals / counts
        pts = unique_pts

    # Compute residuals = measured - baseline_at_that_point
    residuals = np.zeros(len(pts), dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        ix = int(round(x))
        iy = int(round(y))
        ix = max(0, min(GRID_SIZE - 1, ix))
        iy = max(0, min(GRID_SIZE - 1, iy))
        residuals[i] = vals[i] - baseline[iy, ix]

    # Interpolate the residual field across the whole grid
    if len(pts) == 1:
        # Single point: apply a flat offset everywhere
        residual_grid = np.full_like(baseline, residuals[0])
    else:
        # Build grid coordinates for interpolation targets
        xs = np.arange(GRID_SIZE, dtype=np.float64)
        ys = np.arange(GRID_SIZE, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        grid_points = np.column_stack((X.ravel(), Y.ravel()))

        try:
            # RBFInterpolator with multi-quadric kernel.
            # The epsilon (shape parameter) controls smoothness.
            # We leave it as default (None) for automatic selection.
            rbf = RBFInterpolator(pts, residuals, kernel="multiquadric")
            residual_flat = rbf(grid_points)
            residual_grid = residual_flat.reshape(GRID_SIZE, GRID_SIZE)
        except Exception:
            # RBF can fail on collinear or degenerate point sets.
            # Fall back to the physics baseline with no correction.
            residual_grid = np.zeros_like(baseline)

    # Final heatmap = baseline + interpolated residuals
    heatmap = baseline + residual_grid

    # Clip to a sane dBm display range
    heatmap = np.clip(heatmap, -90.0, -30.0)

    return heatmap
