from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, Iterable

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import seaborn as sns


SCALAR_COLOR = "blue"
SIMD_COLOR = "red"


# ============================================================
# Metric handling (refactor: supports ANY *_ns or *_cycles column)
# ============================================================
@dataclass(frozen=True)
class ScaleInfo:
    factor: float
    unit: str


def _metric_kind(metric: str) -> str:
    """
    Classify metric by suffix.
    - *_ns     => "ns"
    - *_cycles => "cycles"
    """
    m = str(metric)
    if m.endswith("_ns"):
        return "ns"
    if m.endswith("_cycles"):
        return "cycles"
    raise ValueError(
        f"Unsupported metric '{metric}'. Expected a column ending with '_ns' or '_cycles'."
    )


def _require_cols_any(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _coerce_numeric(series: pd.Series) -> np.ndarray:
    """
    df_blend_* sometimes has dtype=object for numeric columns.
    Coerce to float safely (drops non-numeric as NaN).
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return x[np.isfinite(x)]


def _metric_series(df: pd.DataFrame, metric: str) -> np.ndarray:
    _require_cols_any(df, [metric], "df")
    return _coerce_numeric(df[metric])


def _auto_scale_for_metric(values: np.ndarray, metric: str) -> ScaleInfo:
    """
    Cycles: no scaling
    Ns: auto-scale to ns/µs/ms/s based on p99
    """
    kind = _metric_kind(metric)
    if kind == "cycles":
        return ScaleInfo(1.0, "cycles")

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ScaleInfo(1.0, "ns")

    p99 = float(np.percentile(v, 99))
    if p99 < 1e3:
        return ScaleInfo(1.0, "ns")
    if p99 < 1e6:
        return ScaleInfo(1e-3, "µs")
    if p99 < 1e9:
        return ScaleInfo(1e-6, "ms")
    return ScaleInfo(1e-9, "s")


def _metric_label(metric: str, unit: str) -> str:
    kind = _metric_kind(metric)
    if kind == "cycles":
        return f"{metric} ({unit})"
    # ns-like
    # nicer label for known columns
    base = metric.replace("_ns", "")
    if base in {"timing", "total", "preprocess", "blend", "postprocess"}:
        return f"{base} ({unit})"
    return f"{metric} ({unit})"

# ---------------------------
# KDE (no SciPy dependency)
# ---------------------------
def _kde_gaussian_1d(
    x: np.ndarray,
    grid: np.ndarray,
    *,
    bandwidth: Optional[float] = None,
    max_points: int = 50_000,
) -> np.ndarray:
    """
    Simple Gaussian KDE on a fixed grid.
    Uses Silverman's rule if bandwidth not provided.
    Downsamples x if too large for speed.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid, dtype=np.float64)

    if x.size > max_points:
        # deterministic-ish downsample
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x = np.sort(x)[idx]

    n = x.size
    std = float(np.std(x))
    if bandwidth is None:
        bw = 1.06 * std * (n ** (-1 / 5)) if std > 0 else 1.0
    else:
        bw = float(bandwidth)

    bw = max(bw, 1e-12)
    # density = mean( N(grid | xi, bw) )
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (bw * math.sqrt(2 * math.pi))
    return dens

# ---------------------------
# Plotly: Distribution (overlapped hist + KDE)
# ---------------------------
def plot_distribution_hist_kde(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str = "timing_ns",  # or "cycles"
    out_path: Optional[str] = None,
    nbins: int = 80,
    title: Optional[str] = None,
    opacity: float = 0.45,
) -> go.Figure:
    """
    Overlapped histogram for scalar(simd) with KDE curve on top.
    Uses ALL values from the chosen column.
    Colors: scalar blue, simd red.
    """

    s = df_scalar
    v = df_simd
    if out_path is not None:
        s = s[s["out_path"] == out_path]
        v = v[v["out_path"] == out_path]

    xs = _metric_series(s, metric)
    xv = _metric_series(v, metric)

    # scale
    scale = _auto_scale_for_metric(np.concatenate([xs, xv]) if (xs.size + xv.size) else np.array([], dtype=float), metric)
    xs_plot = xs * scale.factor
    xv_plot = xv * scale.factor
    x_label = "Latency" if metric == "timing_ns" else "Cycles"
    x_label = f"{x_label} ({scale.unit})"

    if title is None:
        title = f"Distribution ({metric})" + (f" — {out_path}" if out_path else "")

    # Histogram range to align bins
    allx = np.concatenate([xs_plot, xv_plot]) if (xs_plot.size + xv_plot.size) else np.array([0.0])
    lo, hi = float(np.min(allx)), float(np.max(allx))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    # KDE grid
    grid = np.linspace(lo, hi, 600)
    kde_s = _kde_gaussian_1d(xs_plot, grid)
    kde_v = _kde_gaussian_1d(xv_plot, grid)

    fig = go.Figure()

    # Histograms (density)
    fig.add_trace(
        go.Histogram(
            x=xs_plot,
            nbinsx=nbins,
            histnorm="probability density",
            name="scalar",
            marker=dict(color=SCALAR_COLOR),
            opacity=opacity,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=xv_plot,
            nbinsx=nbins,
            histnorm="probability density",
            name="simd",
            marker=dict(color=SIMD_COLOR),
            opacity=opacity,
        )
    )

    # KDE curves
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=kde_s,
            mode="lines",
            name="scalar KDE",
            line=dict(color=SCALAR_COLOR, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=kde_v,
            mode="lines",
            name="simd KDE",
            line=dict(color=SIMD_COLOR, width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Density",
        barmode="overlay",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig.update_xaxes(range=[lo, hi])
    fig.show()

# ============================================================
# Seaborn histogram (works for ANY metric col)
# ============================================================
def plot_distribution_hist_kde_seaborn(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,                      # e.g. "timing_ns" OR "total_ns" OR "blend_cycles"
    out_path: Optional[str] = None,
    bins: int = 80,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
    alpha: float = 0.45,
) -> Tuple[plt.Figure, plt.Axes]:
    s = df_scalar if out_path is None else df_scalar[df_scalar["out_path"] == out_path]
    v = df_simd if out_path is None else df_simd[df_simd["out_path"] == out_path]

    xs = _metric_series(s, metric)
    xv = _metric_series(v, metric)

    scale = _auto_scale_for_metric(
        np.concatenate([xs, xv]) if (xs.size + xv.size) else np.array([], dtype=float),
        metric,
    )
    xs_plot = xs * scale.factor
    xv_plot = xv * scale.factor

    if title is None:
        title = f"Distribution ({metric})" + (f" — {os.path.basename(str(out_path))}" if out_path else "")

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        xs_plot, bins=bins, stat="density", element="bars",
        fill=True, alpha=alpha, color=SCALAR_COLOR, ax=ax, label="scalar",
    )
    sns.histplot(
        xv_plot, bins=bins, stat="density", element="bars",
        fill=True, alpha=alpha, color=SIMD_COLOR, ax=ax, label="simd",
    )

    if xs_plot.size:
        sns.kdeplot(xs_plot, color=SCALAR_COLOR, linewidth=2, ax=ax, label="scalar KDE")
    if xv_plot.size:
        sns.kdeplot(xv_plot, color=SIMD_COLOR, linewidth=2, ax=ax, label="simd KDE")

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(_metric_label(metric, scale.unit))
    ax.set_ylabel("Density")
    
    ax.set_xlim(left=-0.05)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.55, axis='y')
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


# ============================================================
# Plotly percentile tail curve (works for ANY metric col)
# ============================================================
def _tail_x_transform(pct: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(pct, dtype=np.float64), 0.0, 99.9999999)
    return -np.log10(100.0 - p)


def plot_percentile_curve_tail_scale_df(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    percentiles: Optional[np.ndarray] = None,
) -> go.Figure:
    s = df_scalar if out_path is None else df_scalar[df_scalar["out_path"] == out_path]
    v = df_simd if out_path is None else df_simd[df_simd["out_path"] == out_path]

    xs = _metric_series(s, metric)
    xv = _metric_series(v, metric)

    if percentiles is None:
        p_head = np.linspace(0.0, 99.0, 200, endpoint=True)
        p_tail1 = np.linspace(99.0, 99.9, 300, endpoint=False)
        p_tail2 = np.linspace(99.9, 99.99, 300, endpoint=False)
        p_tail3 = np.linspace(99.99, 99.999, 300, endpoint=True)
        p_tail4 = np.linspace(99.999, 99.9999, 400, endpoint=True)
        percentiles = np.unique(np.concatenate([p_head, p_tail1, p_tail2, p_tail3, p_tail4]))
    else:
        percentiles = np.asarray(percentiles, dtype=np.float64)

    scale = _auto_scale_for_metric(
        np.concatenate([xs, xv]) if (xs.size + xv.size) else np.array([], dtype=float),
        metric,
    )

    ys = np.percentile(xs, percentiles) * scale.factor if xs.size else np.full_like(percentiles, np.nan)
    yv = np.percentile(xv, percentiles) * scale.factor if xv.size else np.full_like(percentiles, np.nan)

    x = _tail_x_transform(percentiles)

    tick_p = np.array([0, 90, 99, 99.9, 99.99, 99.999, 99.9999], dtype=np.float64)
    tick_x = _tail_x_transform(tick_p)
    tick_text = [f"{p:g}%" for p in tick_p]

    if title is None:
        title = f"{metric} by Percentile (tail-scaled)" + (f" — {os.path.basename(str(out_path))}" if out_path else "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ys, mode="lines", name="scalar", line=dict(color=SCALAR_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=yv, mode="lines", name="simd",   line=dict(color=SIMD_COLOR, width=2)))

    fig.update_layout(
        title=title,
        xaxis_title="Percentile (tail-scaled)",
        yaxis_title=_metric_label(metric, scale.unit),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig.update_xaxes(tickmode="array", tickvals=tick_x, ticktext=tick_text, range=[-2, 4])
    return fig


# ============================================================
# Matplotlib latencies & cumulative (works for ANY metric col)
# ============================================================
def plot_metric_by_index_matplotlib(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
    r_bound_idx: int = None,
    show: bool = True,
) -> plt.Figure:
    s = df_scalar if out_path is None else df_scalar[df_scalar["out_path"] == out_path]
    v = df_simd if out_path is None else df_simd[df_simd["out_path"] == out_path]

    if sort_by is not None:
        if sort_by not in s.columns or sort_by not in v.columns:
            raise ValueError(f"sort_by='{sort_by}' not in both dataframes")
        s = s.sort_values(sort_by)
        v = v.sort_values(sort_by)

    xs = _metric_series(s, metric)
    xv = _metric_series(v, metric)

    if r_bound_idx is not None:
        xs = xs[:r_bound_idx]
        xv = xv[:r_bound_idx]

    scale = _auto_scale_for_metric(
        np.concatenate([xs, xv]) if (xs.size + xv.size) else np.array([], dtype=float),
        metric,
    )
    xs_plot = xs * scale.factor
    xv_plot = xv * scale.factor

    if title is None:
        title = f"{metric} vs Operation Index" + (f" — {os.path.basename(str(out_path))}" if out_path else "")

    x_s = np.arange(1, xs_plot.size + 1)
    x_v = np.arange(1, xv_plot.size + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_s, xs_plot, label="scalar", linewidth=1.25, color=SCALAR_COLOR)
    ax.plot(x_v, xv_plot, label="simd", linewidth=1.25, color=SIMD_COLOR)
    ax.set_xlabel("Operation index (i)")
    ax.set_ylabel(_metric_label(metric, scale.unit))
    ax.set_title(title, fontsize=15)

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.55, axis='y')
   
    ax.legend(loc="upper left")
    if show:
        plt.show()
    return fig


def plot_cumulative_metric_matplotlib(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,
    out_path: Optional[str] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
    show: bool = True,
) -> plt.Figure:
    s = df_scalar if out_path is None else df_scalar[df_scalar["out_path"] == out_path]
    v = df_simd if out_path is None else df_simd[df_simd["out_path"] == out_path]

    if sort_by is not None:
        if sort_by not in s.columns or sort_by not in v.columns:
            raise ValueError(f"sort_by='{sort_by}' not in both dataframes")
        s = s.sort_values(sort_by)
        v = v.sort_values(sort_by)

    xs = _metric_series(s, metric).astype(np.float64)
    xv = _metric_series(v, metric).astype(np.float64)

    cs = np.cumsum(xs)
    cv = np.cumsum(xv)

    if normalize:
        is_ = np.arange(1, cs.size + 1, dtype=np.float64)
        iv_ = np.arange(1, cv.size + 1, dtype=np.float64)
        ys = cs / is_
        yv = cv / iv_
        default_title = f"Running Average of {metric} vs Index"
    else:
        ys = cs
        yv = cv
        default_title = f"Cumulative {metric} vs Index"

    scale = _auto_scale_for_metric(
        np.concatenate([ys, yv]) if (ys.size + yv.size) else np.array([], dtype=float),
        metric,
    )
    ys_plot = ys * scale.factor
    yv_plot = yv * scale.factor

    if title is None:
        title = default_title + (f" — {os.path.basename(str(out_path))}" if out_path else "")

    x_s = np.arange(1, ys_plot.size + 1)
    x_v = np.arange(1, yv_plot.size + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_s, ys_plot, label="scalar", linewidth=1.25, color=SCALAR_COLOR)
    ax.plot(x_v, yv_plot, label="simd", linewidth=1.25, color=SIMD_COLOR)
    ax.set_xlabel("Operation index (i)")
    ax.set_ylabel(_metric_label(metric, scale.unit))
    ax.set_title(title, fontsize=15)

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.55, axis='y')

    ax.legend(loc="upper left")
    if show:
        plt.show()
    return fig


# ============================================================
# Boxplot side-by-side for ANY metric col (plotly)
# ============================================================
def plot_boxplot_side_by_side(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    s = df_scalar if out_path is None else df_scalar[df_scalar["out_path"] == out_path]
    v = df_simd if out_path is None else df_simd[df_simd["out_path"] == out_path]

    xs = _metric_series(s, metric)
    xv = _metric_series(v, metric)

    scale = _auto_scale_for_metric(
        np.concatenate([xs, xv]) if (xs.size + xv.size) else np.array([], dtype=float),
        metric,
    )
    xs_plot = xs * scale.factor
    xv_plot = xv * scale.factor

    if title is None:
        title = f"Boxplot — {metric}" + (f" — {os.path.basename(str(out_path))}" if out_path else "")

    fig = go.Figure()
    fig.add_trace(go.Box(x=xs_plot, name="scalar", marker_color=SCALAR_COLOR, boxmean=True))
    fig.add_trace(go.Box(x=xv_plot, name="simd", marker_color=SIMD_COLOR, boxmean=True))
    fig.update_layout(
        title=title,
        yaxis_title=_metric_label(metric, scale.unit),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ============================================================
# Grouped histograms (still seaborn), but accepts any metric col
# ============================================================
def _basename_any(p: Any) -> str:
    s = "" if p is None else str(p)
    s = s.replace("\\", "/")
    return os.path.basename(s)


def _resolve_size_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("image_size", "out_size"):
        if col in df.columns:
            return col
    return None


def _mp_from_pixels(pixels: float) -> Optional[float]:
    if pixels is None or (isinstance(pixels, float) and np.isnan(pixels)):
        return None
    try:
        value = float(pixels)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value / 1_000_000.0


def _normalize_group_key(x: Any) -> Any:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return tuple(_normalize_group_key(v) for v in x)
    if isinstance(x, np.ndarray):
        return tuple(_normalize_group_key(v) for v in x.tolist())
    return x


def plot_grouped_histograms(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    group_col: str,
    metric: str,
    bins: int = 80,
    figsize: Tuple[int, int] = (9, 5),
    alpha: float = 0.45,
    show: bool = True,
    require_both: bool = True,
    fallback_to_union_if_empty: bool = True,
) -> Dict[Any, Tuple[plt.Figure, plt.Axes]]:
    # --- validation ---
    for name, df in (("df_scalar", df_scalar), ("df_simd", df_simd)):
        if group_col not in df.columns:
            raise ValueError(f"{name}: group_col='{group_col}' must exist")
        if metric not in df.columns:
            raise ValueError(f"{name}: metric='{metric}' must exist")

    # --- copy + normalized group key ---
    s = df_scalar.copy()
    v = df_simd.copy()
    s["_group_key"] = s[group_col].map(_normalize_group_key)
    v["_group_key"] = v[group_col].map(_normalize_group_key)

    gs = dict(tuple(s.groupby("_group_key", dropna=False)))
    gv = dict(tuple(v.groupby("_group_key", dropna=False)))

    s_keys = pd.Index(gs.keys())
    v_keys = pd.Index(gv.keys())
    keys = s_keys.intersection(v_keys) if require_both else s_keys.union(v_keys)

    if len(keys) == 0 and require_both and fallback_to_union_if_empty:
        keys = s_keys.union(v_keys)

    keys_sorted = sorted(list(keys), key=lambda k: (pd.isna(k), str(k)))

    out: Dict[Any, Tuple[plt.Figure, plt.Axes]] = {}

    for k in keys_sorted:
        s_g = gs.get(k, s.iloc[0:0])
        v_g = gv.get(k, v.iloc[0:0])

        if require_both and (len(s_g) == 0 or len(v_g) == 0):
            continue

        s_vals = pd.to_numeric(s_g[metric], errors="coerce") if len(s_g) else pd.Series(dtype=float)
        v_vals = pd.to_numeric(v_g[metric], errors="coerce") if len(v_g) else pd.Series(dtype=float)

        if (len(s_vals) and not np.isfinite(s_vals).any()) and \
           (len(v_vals) and not np.isfinite(v_vals).any()):
            continue

        row = s_g.iloc[0] if len(s_g) else v_g.iloc[0]

        file_base = _basename_any(row.get("file_name"))
        if not file_base:
            file_base = _basename_any(row.get("out_path"))
        if not file_base:
            file_base = ""

        img_pixels = pd.to_numeric(row.get("image_size"), errors="coerce")
        if pd.isna(img_pixels):
            img_pixels = pd.to_numeric(row.get("out_size"), errors="coerce")
        if pd.isna(img_pixels) and group_col in {"image_size", "out_size"}:
            img_pixels = pd.to_numeric(k, errors="coerce")

        img_mp = _mp_from_pixels(img_pixels)
        img_size_str = f"{img_mp:.2f} MP" if img_mp is not None else None

        title_parts = []
        if file_base:
            title_parts.append(f"file={file_base}")
        if img_size_str:
            title_parts.append(f"size={img_size_str}")
        if not title_parts:
            title_parts.append(f"group={k}")
        title = " | ".join(title_parts)

        fig, ax = plot_distribution_hist_kde_seaborn(
            s_g, v_g,
            metric=metric,
            out_path=None,
            bins=bins,
            title=title,
            figsize=figsize,
            alpha=alpha,
        )
        out[k] = (fig, ax)

        if show:
            plt.show()
    return out


def _stage_cols(kind: str) -> Tuple[str, str, str]:
    if kind == "ns":
        return ("preprocess_ns", "blend_ns", "postprocess_ns")
    if kind == "cycles":
        return ("preprocess_cycles", "blend_cycles", "postprocess_cycles")
    raise ValueError("kind must be 'ns' or 'cycles'")

def _as_float_series(df: pd.DataFrame, col: str) -> np.ndarray:
    x = df[col].to_numpy(dtype=np.float64, copy=False)
    return x[np.isfinite(x)]

def _require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _auto_scale_ns(values_ns: np.ndarray) -> ScaleInfo:
    """Auto-scale ns -> ns/µs/ms/s based on p99."""
    v = np.asarray(values_ns, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ScaleInfo(1.0, "ns")
    p99 = float(np.percentile(v, 99))
    if p99 < 1e3:
        return ScaleInfo(1.0, "ns")
    if p99 < 1e6:
        return ScaleInfo(1e-3, "µs")
    if p99 < 1e9:
        return ScaleInfo(1e-6, "ms")
    return ScaleInfo(1e-9, "s")


# ============================================================
# Refactored make_all_plots: supports df_hist_* and df_blend_*
# ============================================================
def make_all_plots(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,                       # ANY *_ns or *_cycles col
    out_path: Optional[str] = None,
    nbins: int = 80,
    sort_by: Optional[str] = None,
    group_col: Optional[str] = None,   # if provided, also do grouped histograms
    cumulative_normalize: bool = False,
    r_bound_idx: int = None,
    show: bool = True,
    return_figs: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Works for both df_hist_* (timing_ns/cycles) and df_blend_* (preprocess_ns, total_cycles, ...).
    """
    figs: Dict[str, Any] = {}

    # seaborn histogram + kde
    hist_fig, hist_ax = plot_distribution_hist_kde_seaborn(
        df_scalar, df_simd, metric=metric, out_path=out_path, bins=nbins
    )
    if show:
        plt.show()

    # percentile plotly
    pct_fig = plot_percentile_curve_tail_scale_df(
        df_scalar, df_simd, metric=metric, out_path=out_path
    )
    if show:
        pct_fig.show()

    # boxplot plotly
    box_fig = plot_boxplot_side_by_side(
        df_scalar, df_simd, metric=metric, out_path=out_path
    )
    if show:
        box_fig.show()

    # matplotlib line plots
    metric_index_fig = plot_metric_by_index_matplotlib(
        df_scalar,
        df_simd,
        metric=metric,
        out_path=out_path,
        sort_by=sort_by,
        r_bound_idx=r_bound_idx,
        show=show,
    )
    cumulative_fig = plot_cumulative_metric_matplotlib(
        df_scalar,
        df_simd,
        metric=metric,
        out_path=out_path,
        sort_by=sort_by,
        normalize=cumulative_normalize,
        show=show,
    )

    # grouped histograms (optional)
    if group_col is not None:
        plot_grouped_histograms(
            df_scalar, df_simd, group_col=group_col, metric=metric, bins=nbins, show=show
        )

    if return_figs:
        figs["hist_seaborn"] = (hist_fig, hist_ax)
        figs["percentile_plotly"] = pct_fig
        figs["boxplot_plotly"] = box_fig
        figs["metric_by_index"] = metric_index_fig
        figs["cumulative_metric"] = cumulative_fig
        return figs
    return None



# ============================================================
# 1) Plotly horizontal grouped boxplots (side-by-side per stage)
# ============================================================
def plot_stage_boxplots_side_by_side(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    kind: str = "ns",  # "ns" or "cycles"
    title: str = "Stage breakdown (boxplots)",
    auto_scale: bool = True,  # only for ns
    box_width: float = 0.45,
    pair_offset: float = 0.18,
    height: int = 560,
    clamp_min: Optional[float] = None,
    show: bool = True,
) -> go.Figure:
    """
    Horizontal grouped boxplots for preprocess/blend/postprocess.
    For each stage, scalar and simd are side-by-side (close together).
    Colors: scalar blue, simd red.
    """
    cols = _stage_cols(kind)
    _require_cols(df_scalar, cols, "df_scalar")
    _require_cols(df_simd, cols, "df_simd")

    # -------------------------
    # Scale X values
    # -------------------------
    if kind == "ns" and auto_scale:
        all_vals = np.concatenate(
            [_as_float_series(df_scalar, c) for c in cols] +
            [_as_float_series(df_simd, c) for c in cols]
        )
        scale = _auto_scale_ns(all_vals)
    else:
        scale = ScaleInfo(1.0, "cycles" if kind == "cycles" else "ns")

    stage_labels = ["preprocess", "blend", "postprocess"]
    y_centers = np.arange(len(stage_labels), dtype=float)  # [0, 1, 2]

    # -------------------------
    # Ensure requested width fits
    # -------------------------
    pad = 0.03  # small safety gap so boxes don't touch visually
    min_pair_offset = (box_width / 2.0) + pad
    pair_offset = max(pair_offset, min_pair_offset)

    y_scalar = y_centers + pair_offset
    y_simd = y_centers - pair_offset

    # -------------------------
    # Build figure
    # -------------------------
    fig = go.Figure()

    for i, (c, stage) in enumerate(zip(cols, stage_labels)):
        xs = _as_float_series(df_scalar, c) * scale.factor
        xv = _as_float_series(df_simd, c) * scale.factor
        if clamp_min is not None:
            xs = xs[xs >= clamp_min]
            xv = xv[xv >= clamp_min]

        # Use numeric y to make width behave as expected
        fig.add_trace(
            go.Box(
                x=xs,
                y=np.full(xs.size, y_scalar[i]),
                name="scalar",
                orientation="h",
                marker_color=SCALAR_COLOR,
                legendgroup="scalar",
                showlegend=(i == 0),
                boxmean=True,
                width=box_width,
            )
        )
        fig.add_trace(
            go.Box(
                x=xv,
                y=np.full(xv.size, y_simd[i]),
                name="simd",
                orientation="h",
                marker_color=SIMD_COLOR,
                legendgroup="simd",
                showlegend=(i == 0),
                boxmean=True,
                width=box_width,
            )
        )

    # -------------------------
    # Layout / axes
    # -------------------------
    fig.update_layout(
        title=title,
        xaxis_title=("Time" if kind == "ns" else "Cycles") + f" ({scale.unit})",
        yaxis_title="Stage",
        height=height,
        margin=dict(l=80, r=30, t=60, b=60),
        legend=dict(x=0.92, y=0.98),
    )
    if clamp_min is not None:
        all_vals = []
        for c in cols:
            all_vals.append(_as_float_series(df_scalar, c) * scale.factor)
            all_vals.append(_as_float_series(df_simd, c) * scale.factor)
        if all_vals:
            vmax = float(np.nanmax(np.concatenate(all_vals)))
            if np.isfinite(vmax):
                fig.update_xaxes(range=[clamp_min, vmax * 1.05])

    # Show stage names at the stage centers (0,1,2), even though traces use shifted y
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_centers,
        ticktext=stage_labels,
        range=[-0.6, len(stage_labels) - 1 + 0.6],
    )

    if show:
        fig.show()
    return fig


# =====================================================================
# 2) Non-scaled stacked mean bars: scalar vs simd, components as hatches
# =====================================================================
def plot_stacked_mean_bars(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    kind: str = "ns",  # "ns" or "cycles"
    title: str = "Mean stage time breakdown (stacked)",
    show: bool = True,
) -> plt.Figure:
    """
    Two bars side by side (scalar, simd). Each bar stacked from mean(preprocess, blend, postprocess).
    - Adds color variation per stage (blues for scalar, reds for simd)
    - Adds % labels inside each section
    - Removes minor ticks; removes vertical grid lines (y-grid only)
    """
    cols = _stage_cols(kind)
    _require_cols(df_scalar, cols, "df_scalar")
    _require_cols(df_simd, cols, "df_simd")

    means_scalar = np.array([np.mean(df_scalar[c]) for c in cols], dtype=np.float64)
    means_simd = np.array([np.mean(df_simd[c]) for c in cols], dtype=np.float64)

    labels = ["preprocess", "blend", "postprocess"]

    # Shades (3 related variations)
    blues = [plt.cm.Blues(v) for v in (0.55, 0.70, 0.85)]
    reds  = [plt.cm.Reds(v)  for v in (0.55, 0.70, 0.85)]

    fig, ax = plt.subplots(figsize=(9, 7))
    x = np.array([0, 1], dtype=np.float64)  # scalar, simd
    width = 0.55

    # Scalar bar
    total_s = float(np.nansum(means_scalar))
    bottom = 0.0
    for val, colr, lab in zip(means_scalar, blues, labels):
        if not np.isfinite(val) or val <= 0:
            continue
        ax.bar(x[0], val, width, bottom=bottom, color=colr, edgecolor="black", linewidth=0.6)
        # % label inside segment
        if total_s > 0:
            pct = 100.0 * val / total_s
            ax.text(
                x[0],
                bottom + val / 2.0,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if pct > 8 else "black",
                fontweight="bold" if pct > 12 else None,
            )
        bottom += val

    # SIMD bar
    total_v = float(np.nansum(means_simd))
    bottom = 0.0
    for val, colr, lab in zip(means_simd, reds, labels):
        if not np.isfinite(val) or val <= 0:
            continue
        ax.bar(x[1], val, width, bottom=bottom, color=colr, edgecolor="black", linewidth=0.6)
        if total_v > 0:
            pct = 100.0 * val / total_v
            ax.text(
                x[1],
                bottom + val / 2.0,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if pct > 8 else "black",
                fontweight="bold" if pct > 12 else None,
            )
        bottom += val

    ax.set_xticks(x, ["scalar", "simd"])
    ax.set_ylabel("Time (ns)" if kind == "ns" else "Cycles")
    ax.set_title(title, fontsize=15)

    # Grid: horizontal only; no minor ticks
    ax.minorticks_off()
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.grid(False, axis="x")

    stage_handles = [
        (mpatches.Patch(facecolor=blues[i], edgecolor="black"),
        mpatches.Patch(facecolor=reds[i],  edgecolor="black"))
        for i in range(3)
    ]

    ax.legend(
        handles=stage_handles,
        labels=labels,  # preprocess, blend, postprocess
        handler_map={tuple: HandlerTuple(ndivide=None)},
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.6,
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig



# ============================================================
# 3) Pie charts: composition of mean(stage) for scalar and simd
# ============================================================
def plot_stage_mean_pies(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    kind: str = "ns",  # "ns" or "cycles"
    title: str = "Mean composition (preprocess/blend/postprocess)",
    show: bool = True,
) -> plt.Figure:
    """
    Two pie charts (scalar, simd), with readable formatting and related color variations.
    """
    cols = _stage_cols(kind)
    _require_cols(df_scalar, cols, "df_scalar")
    _require_cols(df_simd, cols, "df_simd")

    stage_labels = ["preprocess", "blend", "postprocess"]
    means_scalar = np.array([np.mean(df_scalar[c]) for c in cols], dtype=np.float64)
    means_simd = np.array([np.mean(df_simd[c]) for c in cols], dtype=np.float64)

    # Related color sets
    scalar_colors = [plt.cm.Blues(v) for v in (0.55, 0.70, 0.85)]
    simd_colors   = [plt.cm.Reds(v)  for v in (0.55, 0.70, 0.85)]

    def pct_only(pct: float) -> str:
        return f"{pct:.1f}%" if pct > 0 else ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=14)

    wedgeprops = dict(edgecolor="white", linewidth=1.2)
    label_textprops = dict(fontsize=12, color="black")  # labels
    pct_textprops = dict(fontsize=11, color="white", fontweight="semibold")

    for ax, values, colors, name in [
        (axes[0], means_scalar, scalar_colors, "scalar"),
        (axes[1], means_simd,   simd_colors,   "simd"),
    ]:
        wedges, label_texts, pct_texts = ax.pie(
            values,
            labels=stage_labels,
            colors=colors,
            autopct=pct_only,
            startangle=90,
            counterclock=False,
            wedgeprops=wedgeprops,
            textprops=label_textprops,   # applies to labels by default
            pctdistance=0.65,
            labeldistance=1.10,
        )

        # Style percent texts only
        for t in pct_texts:
            t.set(**pct_textprops)

        ax.set_title(name, fontsize=16)
        ax.axis("equal")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

# plot_stage_boxplots_side_by_side(df_scalar, df_simd, kind="ns")
# plot_stacked_mean_bars(df_scalar, df_simd, kind="ns")
# plot_stage_mean_pies(df_scalar, df_simd, kind="ns")

def plot_metric_vs_megapixels(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,
    title: str | None = None,
    agg: str = "median",          # "mean" or "median"
    show_markers: bool = True,
    auto_scale_ns: bool = True,   # scale *_ns to ns/us/ms/s like other plots
    show: bool = True,
) -> go.Figure:
    """
    Line plot: metric vs image size (MP), for scalar and simd.

    Works for both df_hist_* and df_blend_* as long as they contain:
      - image_size or out_size (pixels)
      - metric column (e.g. 'timing_ns', 'cycles', 'total_ns', 'blend_cycles', ...)

    Steps:
      1) Convert image_size (pixels) -> megapixels
      2) Coerce metric to numeric (handles object dtypes)
      3) Aggregate metric per image_size (mean/median)
      4) Sort by image_size and plot two lines (scalar blue, simd red)
    """
    size_col_s = _resolve_size_col(df_scalar)
    size_col_v = _resolve_size_col(df_simd)
    if size_col_s is None or size_col_v is None:
        raise ValueError("Both dataframes must have 'image_size' or 'out_size' column")
    if metric not in df_scalar.columns or metric not in df_simd.columns:
        raise ValueError(f"Both dataframes must have metric column '{metric}'")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    def prep(df: pd.DataFrame, size_col: str) -> pd.DataFrame:
        out = df[[size_col, metric]].copy()
        out = out.rename(columns={size_col: "image_size"})
        out["image_size"] = pd.to_numeric(out["image_size"], errors="coerce")
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
        out = out.dropna(subset=["image_size", metric])
        out["mp"] = out["image_size"] / 1e6
        return out

    s = prep(df_scalar, size_col_s)
    v = prep(df_simd, size_col_v)

    # aggregate per image size
    if agg == "mean":
        s_agg = s.groupby("mp", as_index=False)[metric].mean()
        v_agg = v.groupby("mp", as_index=False)[metric].mean()
    else:
        s_agg = s.groupby("mp", as_index=False)[metric].median()
        v_agg = v.groupby("mp", as_index=False)[metric].median()

    # sort by MP
    s_agg = s_agg.sort_values("mp")
    v_agg = v_agg.sort_values("mp")

    # optional scaling for *_ns columns
    y_label = metric
    scale_factor = 1.0
    unit = ""
    if auto_scale_ns and str(metric).endswith("_ns"):
        all_vals = np.concatenate([s_agg[metric].to_numpy(), v_agg[metric].to_numpy()])
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size:
            p99 = float(np.percentile(all_vals, 99))
            if p99 < 1e3:
                scale_factor, unit = 1.0, "ns"
            elif p99 < 1e6:
                scale_factor, unit = 1e-3, "µs"
            elif p99 < 1e9:
                scale_factor, unit = 1e-6, "ms"
            else:
                scale_factor, unit = 1e-9, "s"
        y_label = f"{metric} ({unit})"

    if title is None:
        title = f"{metric} vs Image Size (MP) [{agg}]"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_agg["mp"],
            y=s_agg[metric] * scale_factor,
            mode="lines+markers" if show_markers else "lines",
            name="scalar",
            line=dict(color=SCALAR_COLOR, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=v_agg["mp"],
            y=v_agg[metric] * scale_factor,
            mode="lines+markers" if show_markers else "lines",
            name="simd",
            line=dict(color=SIMD_COLOR, width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Image size (megapixels)",
        yaxis_title=y_label,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=60, b=60),
    )

    if show:
        fig.show()
    return fig
