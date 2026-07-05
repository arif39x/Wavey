SIGNAL_CMAP_COLORS = [
    (0.10, 0.10, 0.90),
    (0.10, 0.80, 0.90),
    (0.20, 0.80, 0.30),
    (1.00, 0.70, 0.10),
    (0.90, 0.10, 0.10),
]
SIGNAL_CMAP_CONTROLS = [0.0, 0.25, 0.5, 0.75, 1.0]


def confidence_to_rgb(val: float) -> tuple[float, float, float]:
    h = 0.6 - val * 0.6
    return _hsl_to_rgb(h, 0.9, 0.5)


def _hsl_to_rgb(h, s, l):
    if s == 0:
        return (l, l, l)

    def f(n):
        k = (n + h * 12) % 12
        return l - s * min(l, 1 - l) * max(-1, min(k - 3, 9 - k, 1))

    return (max(0, min(1, f(0))), max(0, min(1, f(8))), max(0, min(1, f(4))))


try:
    from matplotlib.colors import LinearSegmentedColormap
    _SIGNAL_SEGMENTS = {
        "red":   [(0.00, 0.10, 0.10), (0.25, 0.10, 0.10),
                  (0.50, 0.20, 0.20), (0.75, 1.00, 1.00), (1.00, 0.90, 0.90)],
        "green": [(0.00, 0.10, 0.10), (0.25, 0.80, 0.80),
                  (0.50, 0.80, 0.80), (0.75, 0.70, 0.70), (1.00, 0.10, 0.10)],
        "blue":  [(0.00, 0.90, 0.90), (0.25, 0.90, 0.90),
                  (0.50, 0.30, 0.30), (0.75, 0.10, 0.10), (1.00, 0.10, 0.10)],
    }
    SIGNAL_CMAP = LinearSegmentedColormap("signal", _SIGNAL_SEGMENTS, N=256)
except ImportError:
    SIGNAL_CMAP = None
