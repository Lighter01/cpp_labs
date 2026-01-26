import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STRATEGY_MAP = {
    ("scalar", "seq"): "scalar_seq",
    ("simd", "seq"): "simd_seq",
    ("scalar", "par"): "scalar_par",
    ("simd", "par"): "simd_par",
}

BEST_STRATEGY_ORDER = [
    "scalar_seq",
    "simd_seq",
    "scalar_par_best",
    "simd_par_best",
]


def find_csv_files(csv_root: str | os.PathLike[str]) -> List[str]:
    root = Path(csv_root)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*.csv") if p.is_file())


def load_results(csv_root: str | os.PathLike[str]) -> pd.DataFrame:
    files = find_csv_files(csv_root)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {csv_root}")

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skipping unreadable CSV: {path} ({exc})")
            continue
        df = df.copy()
        df["source_file"] = os.path.basename(path)
        df["source_path"] = path
        frames.append(df)

    if not frames:
        raise ValueError(f"All CSV files failed to load under {csv_root}")

    return pd.concat(frames, ignore_index=True)


def normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    numeric_cols = [
        "out_size",
        "out_channels",
        "num_threads",
        "grain",
        "preprocess_ns",
        "blend_ns",
        "postprocess_ns",
        "total_ns",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "num_threads" not in df.columns:
        df["num_threads"] = 1

    required_cols = ["out_size", "preprocess_ns", "blend_ns", "postprocess_ns", "impl", "exec_mode", "mode"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["out_size", "preprocess_ns", "blend_ns", "postprocess_ns", "impl", "exec_mode", "mode"])

    df["impl"] = df["impl"].astype(str).str.lower()
    df["exec_mode"] = df["exec_mode"].astype(str).str.lower()
    df.loc[df["exec_mode"] == "seq", "num_threads"] = 1

    df["pixels"] = df["out_size"]
    df["mpix"] = (df["pixels"] / 1e6).round(6)

    df["total_ns_calc"] = df["preprocess_ns"] + df["blend_ns"] + df["postprocess_ns"]
    df["total_ns_use"] = df["total_ns_calc"]
    if "total_ns" in df.columns:
        missing = df["total_ns_use"].isna()
        df.loc[missing, "total_ns_use"] = df.loc[missing, "total_ns"]

    df["ns_per_pixel"] = df["total_ns_use"] / df["pixels"]

    df["strategy"] = [
        STRATEGY_MAP.get((impl, exec_mode), "unknown")
        for impl, exec_mode in zip(df["impl"], df["exec_mode"])
    ]

    df = df[df["strategy"] != "unknown"]
    if df.empty:
        raise ValueError("No valid rows after filtering. Check CSV root and schema.")

    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "mode",
        "mpix",
        "strategy",
        "num_threads",
        "preprocess_ns",
        "blend_ns",
        "postprocess_ns",
        "total_ns_use",
        "ns_per_pixel",
    ]
    _require_columns(df, required, "normalized df")

    group_cols = ["mode", "mpix", "strategy", "num_threads"]
    agg_inputs = {
        "preprocess_ns": "preprocess_ns",
        "blend_ns": "blend_ns",
        "postprocess_ns": "postprocess_ns",
        "total_ns_use": "total_ns",
        "ns_per_pixel": "ns_per_pixel",
    }

    def _agg_group(g: pd.DataFrame) -> pd.Series:
        out = {}
        for src, name in agg_inputs.items():
            s = pd.to_numeric(g[src], errors="coerce").dropna()
            if s.empty:
                out[f"median_{name}"] = np.nan
                out[f"q25_{name}"] = np.nan
                out[f"q75_{name}"] = np.nan
            else:
                out[f"median_{name}"] = s.median()
                out[f"q25_{name}"] = s.quantile(0.25)
                out[f"q75_{name}"] = s.quantile(0.75)
        out["count"] = len(g)
        return pd.Series(out)

    df_agg = df.groupby(group_cols, dropna=False).apply(_agg_group).reset_index()

    parts = df_agg["strategy"].astype(str).str.split("_", n=1, expand=True)
    if parts.shape[1] == 2:
        df_agg["impl"] = parts[0]
        df_agg["exec_mode"] = parts[1]
    else:
        df_agg["impl"] = "unknown"
        df_agg["exec_mode"] = "unknown"

    return df_agg


def select_best_threads(df_agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = ["mode", "mpix", "strategy", "num_threads", "median_total_ns"]
    _require_columns(df_agg, required, "df_agg")

    df = df_agg.copy()
    if "impl" not in df.columns:
        parts = df["strategy"].astype(str).str.split("_", n=1, expand=True)
        if parts.shape[1] == 2:
            df["impl"] = parts[0]
        else:
            df["impl"] = "unknown"

    df_par = df[df["strategy"].isin(["scalar_par", "simd_par"])]
    if df_par.empty:
        return df_par.copy(), df_par.copy()

    idx = df_par.groupby(["mode", "mpix", "impl"])["median_total_ns"].idxmin()
    df_best = df_par.loc[idx].copy()
    df_best = df_best.rename(columns={"median_total_ns": "best_total_ns"})
    df_best["best_t"] = df_best["num_threads"]
    df_best["strategy"] = df_best["impl"] + "_par_best"

    seq = df[df["strategy"].isin(["scalar_seq", "simd_seq"])]
    seq = seq[["mode", "mpix", "impl", "median_total_ns"]].rename(
        columns={"median_total_ns": "seq_total_ns"}
    )
    best_table = df_best.merge(seq, on=["mode", "mpix", "impl"], how="left")
    best_table["speedup_vs_seq"] = best_table["seq_total_ns"] / best_table["best_total_ns"]
    best_table = best_table[
        ["mode", "mpix", "impl", "best_t", "best_total_ns", "speedup_vs_seq"]
    ].sort_values(["mode", "mpix", "impl"])

    return df_best, best_table


def plot_best_of(df_agg: pd.DataFrame, df_best: pd.DataFrame, out_dir: str | os.PathLike[str], mode: Optional[str] = None) -> None:
    df_agg_mode, mode_used, multiple = _filter_mode(df_agg, mode)
    df_best_mode, _, _ = _filter_mode(df_best, mode_used)
    if df_agg_mode.empty:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    strategy_styles = {
        "scalar_seq": {"color": "#1f77b4", "label": "scalar seq"},
        "simd_seq": {"color": "#d62728", "label": "simd seq"},
        "scalar_par_best": {"color": "#2ca02c", "label": "scalar par (best)"},
        "simd_par_best": {"color": "#9467bd", "label": "simd par (best)"},
    }

    for strategy in ["scalar_seq", "simd_seq"]:
        df_s = df_agg_mode[df_agg_mode["strategy"] == strategy].sort_values("mpix")
        if df_s.empty:
            continue
        y = df_s["median_total_ns"] * 1e-6
        ax_top.plot(df_s["mpix"], y, marker="o", **strategy_styles[strategy])

    for impl in ["scalar", "simd"]:
        df_b = df_best_mode[df_best_mode["impl"] == impl].sort_values("mpix")
        if df_b.empty:
            continue
        strategy = f"{impl}_par_best"
        y = df_b["best_total_ns"] * 1e-6
        ax_top.plot(df_b["mpix"], y, marker="o", **strategy_styles[strategy])

    mpix_values = sorted(df_agg_mode["mpix"].dropna().unique())
    _set_mp_ticks(ax_bottom, mpix_values)

    for impl, color, label in [
        ("scalar", "#2ca02c", "scalar best t"),
        ("simd", "#9467bd", "simd best t"),
    ]:
        df_b = df_best_mode[df_best_mode["impl"] == impl].sort_values("mpix")
        if df_b.empty:
            continue
        ax_bottom.step(df_b["mpix"], df_b["best_t"], where="mid", color=color, label=label)
        ax_bottom.plot(df_b["mpix"], df_b["best_t"], marker="o", color=color, linestyle="none")

    best_ticks = sorted(df_best_mode["best_t"].dropna().unique())
    if best_ticks:
        ax_bottom.set_yticks(best_ticks)
        ax_bottom.set_yticklabels([str(int(t)) for t in best_ticks])
        _stagger_ytick_labels(ax_bottom, step=0.008)

    ax_top.set_ylabel("Total time (ms)")
    ax_bottom.set_xlabel("MP")
    ax_bottom.set_ylabel("Best threads")
    title = "Best-of total time"
    if mode_used is not None:
        title = f"{title} ({mode_used})"
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    ax_bottom.grid(True, axis="y", alpha=0.3)
    ax_bottom.set_title("Best threads vs MP")

    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        ax_top.legend()

    handles, labels = ax_bottom.get_legend_handles_labels()
    if handles:
        ax_bottom.legend(loc="upper left")

    out_path = Path(out_dir) / "plots" / f"best_of_total_time{_mode_suffix(mode_used, multiple)}.png"
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speedup_grid(
    df_agg: pd.DataFrame,
    out_dir: str | os.PathLike[str],
    mode: Optional[str] = None,
    mpix_list: Optional[Iterable[float]] = None,
) -> None:
    df_mode, mode_used, multiple = _filter_mode(df_agg, mode)
    if df_mode.empty:
        return

    available = sorted(df_mode["mpix"].dropna().unique())
    if not available:
        return

    if mpix_list is None:
        mpix_list = [1, 2, 4, 8, 16, 32]

    mpix_vals = _match_mpix_values(available, list(mpix_list))
    if not mpix_vals:
        mpix_vals = available[: min(len(available), 6)]

    seq = df_mode[df_mode["strategy"].isin(["scalar_seq", "simd_seq"])]
    seq_map = {
        (row["impl"], row["mpix"]): row["median_total_ns"]
        for _, row in seq.iterrows()
    }

    ncols = 3
    nrows = int(np.ceil(len(mpix_vals) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6), squeeze=False)

    for idx, mpix in enumerate(mpix_vals):
        ax = axes[idx // ncols][idx % ncols]
        df_m = df_mode[df_mode["mpix"] == mpix]
        df_par = df_m[df_m["strategy"].isin(["scalar_par", "simd_par"])]
        for impl, color in [("scalar", "#1f77b4"), ("simd", "#d62728")]:
            base = seq_map.get((impl, mpix))
            if base is None or np.isnan(base):
                continue
            df_impl = df_par[df_par["strategy"] == f"{impl}_par"].sort_values("num_threads")
            if df_impl.empty:
                continue
            threads = df_impl["num_threads"].astype(float)
            speedup = base / df_impl["median_total_ns"]
            ax.plot(threads, speedup, marker="o", color=color, label=f"{impl} par")

        scalar_seq = seq_map.get(("scalar", mpix))
        simd_seq = seq_map.get(("simd", mpix))
        if scalar_seq is not None and simd_seq is not None and not np.isnan(scalar_seq) and not np.isnan(simd_seq):
            simd_seq_speedup = scalar_seq / simd_seq
            ax.axhline(
                simd_seq_speedup,
                color="#7f3fbf",
                linestyle=":",
                linewidth=1.2,
                label="simd seq vs scalar seq",
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{_mp_title(mpix)} MP")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup")
        thread_ticks = sorted(df_par["num_threads"].dropna().unique())
        if thread_ticks:
            ax.set_xticks(thread_ticks)
            ax.set_xticklabels([str(int(t)) for t in thread_ticks])
        ax.grid(True, alpha=0.3)

    for j in range(len(mpix_vals), nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    legend_items = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in legend_items:
                legend_items[label] = handle
    if legend_items:
        fig.legend(legend_items.values(), legend_items.keys(), loc="upper right")

    out_path = Path(out_dir) / "plots" / f"speedup_vs_threads_grid{_mode_suffix(mode_used, multiple)}.png"
    _ensure_dir(out_path.parent)
    title = "Speedup vs threads by MP"
    if mode_used is not None:
        title = f"{title} ({mode_used})"
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ns_per_pixel_best_of(
    df_agg: pd.DataFrame,
    df_best: pd.DataFrame,
    out_dir: str | os.PathLike[str],
    mode: Optional[str] = None,
) -> None:
    df_agg_mode, mode_used, multiple = _filter_mode(df_agg, mode)
    df_best_mode, _, _ = _filter_mode(df_best, mode_used)
    if df_agg_mode.empty:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    strategy_styles = {
        "scalar_seq": {"color": "#1f77b4", "label": "scalar seq"},
        "simd_seq": {"color": "#d62728", "label": "simd seq"},
        "scalar_par_best": {"color": "#2ca02c", "label": "scalar par (best)"},
        "simd_par_best": {"color": "#9467bd", "label": "simd par (best)"},
    }

    for strategy in ["scalar_seq", "simd_seq"]:
        df_s = df_agg_mode[df_agg_mode["strategy"] == strategy].sort_values("mpix")
        if df_s.empty:
            continue
        y = df_s["median_ns_per_pixel"]
        ax_top.plot(df_s["mpix"], y, marker="o", **strategy_styles[strategy])

    for impl in ["scalar", "simd"]:
        df_b = df_best_mode[df_best_mode["impl"] == impl].sort_values("mpix")
        if df_b.empty or "median_ns_per_pixel" not in df_b.columns:
            continue
        strategy = f"{impl}_par_best"
        y = df_b["median_ns_per_pixel"]
        ax_top.plot(df_b["mpix"], y, marker="o", **strategy_styles[strategy])

    mpix_values = sorted(df_agg_mode["mpix"].dropna().unique())
    _set_mp_ticks(ax_bottom, mpix_values)
    for impl, color, label in [
        ("scalar", "#2ca02c", "scalar best t"),
        ("simd", "#9467bd", "simd best t"),
    ]:
        df_b = df_best_mode[df_best_mode["impl"] == impl].sort_values("mpix")
        if df_b.empty:
            continue
        ax_bottom.step(df_b["mpix"], df_b["best_t"], where="mid", color=color, label=label)
        ax_bottom.plot(df_b["mpix"], df_b["best_t"], marker="o", color=color, linestyle="none")

    best_ticks = sorted(df_best_mode["best_t"].dropna().unique())
    if best_ticks:
        ax_bottom.set_yticks(best_ticks)
        ax_bottom.set_yticklabels([str(int(t)) for t in best_ticks])
        _stagger_ytick_labels(ax_bottom, step=0.008)

    ax_top.set_ylabel("ns per pixel")
    ax_bottom.set_xlabel("MP")
    ax_bottom.set_ylabel("Best threads")
    title = "Best-of ns per pixel"
    if mode_used is not None:
        title = f"{title} ({mode_used})"
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    ax_bottom.grid(True, axis="y", alpha=0.3)
    ax_bottom.set_title("Best threads vs MP")

    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        ax_top.legend()

    handles, labels = ax_bottom.get_legend_handles_labels()
    if handles:
        ax_bottom.legend(loc="upper left")

    out_path = Path(out_dir) / "plots" / f"ns_per_pixel_best_of{_mode_suffix(mode_used, multiple)}.png"
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stage_boxplots(
    df: pd.DataFrame,
    df_best: pd.DataFrame,
    out_dir: str | os.PathLike[str],
    mode: Optional[str] = None,
    mpix_list: Optional[Iterable[float]] = None,
) -> None:
    df_mode, mode_used, multiple = _filter_mode(df, mode)
    df_best_mode, _, _ = _filter_mode(df_best, mode_used)
    if df_mode.empty:
        return

    available = sorted(df_mode["mpix"].dropna().unique())
    if not available:
        return

    if mpix_list is None:
        if len(available) == 1:
            mpix_list = [available[0]]
        else:
            mpix_list = [available[0], available[-1]]

    best_map = {
        (row["impl"], row["mpix"]): row["best_t"]
        for _, row in df_best_mode.iterrows()
        if not pd.isna(row.get("best_t", np.nan))
    }

    for mpix in mpix_list:
        if mpix not in available:
            continue
        base = df_mode[df_mode["mpix"] == mpix]
        frames = []
        frames.append(
            base[(base["impl"] == "scalar") & (base["exec_mode"] == "seq")].assign(strategy="scalar_seq")
        )
        frames.append(
            base[(base["impl"] == "simd") & (base["exec_mode"] == "seq")].assign(strategy="simd_seq")
        )
        if ("scalar", mpix) in best_map:
            t = best_map[("scalar", mpix)]
            frames.append(
                base[(base["impl"] == "scalar") & (base["exec_mode"] == "par") & (base["num_threads"] == t)]
                .assign(strategy="scalar_par_best")
            )
        if ("simd", mpix) in best_map:
            t = best_map[("simd", mpix)]
            frames.append(
                base[(base["impl"] == "simd") & (base["exec_mode"] == "par") & (base["num_threads"] == t)]
                .assign(strategy="simd_par_best")
            )

        plot_df = pd.concat(frames, ignore_index=True)
        if plot_df.empty:
            continue

        for stage in ["preprocess_ns", "blend_ns", "postprocess_ns"]:
            data = []
            labels = []
            colors = []
            best_scalar = best_map.get(("scalar", mpix))
            best_simd = best_map.get(("simd", mpix))
            label_map = {
                "scalar_seq": "scalar seq",
                "simd_seq": "simd seq",
                "scalar_par_best": f"scalar par (t={int(best_scalar)})" if best_scalar is not None else "scalar par",
                "simd_par_best": f"simd par (t={int(best_simd)})" if best_simd is not None else "simd par",
            }
            color_map = {
                "scalar_seq": "#1f77b4",
                "simd_seq": "#d62728",
                "scalar_par_best": "#2ca02c",
                "simd_par_best": "#9467bd",
            }
            for strategy in BEST_STRATEGY_ORDER:
                vals = plot_df.loc[plot_df["strategy"] == strategy, stage]
                vals = pd.to_numeric(vals, errors="coerce").dropna()
                if vals.empty:
                    continue
                data.append(vals / 1e6)
                labels.append(label_map.get(strategy, strategy))
                colors.append(color_map.get(strategy, "#808080"))

            if not data:
                continue

            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            box = ax.boxplot(
                data,
                labels=labels,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#222222", "linewidth": 1.4},
                whiskerprops={"color": "#555555"},
                capprops={"color": "#555555"},
            )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
                patch.set_edgecolor("#333333")
            ax.set_xlabel("Strategy")
            ax.set_ylabel("Time (ms)")
            best_info = []
            if best_scalar is not None:
                best_info.append(f"scalar t={int(best_scalar)}")
            if best_simd is not None:
                best_info.append(f"simd t={int(best_simd)}")
            best_suffix = f" | best {' / '.join(best_info)}" if best_info else ""
            title = f"{stage.replace('_ns', '')} stage ({_mp_title(mpix)} MP){best_suffix}"
            if mode_used is not None:
                title = f"{title} ({mode_used})"
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

            filename = f"box_{stage.replace('_ns', '')}_{_mpix_label(mpix)}{_mode_suffix(mode_used, multiple)}.png"
            out_path = Path(out_dir) / "plots" / filename
            _ensure_dir(out_path.parent)
            fig.savefig(out_path, dpi=200)
            plt.close(fig)


def plot_best_threads(
    df_best: pd.DataFrame,
    out_dir: str | os.PathLike[str],
    mode: Optional[str] = None,
) -> None:
    df_best_mode, mode_used, multiple = _filter_mode(df_best, mode)
    if df_best_mode.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for impl, color in [("scalar", "#1f77b4"), ("simd", "#d62728")]:
        df_impl = df_best_mode[df_best_mode["impl"] == impl].sort_values("mpix")
        if df_impl.empty:
            continue
        ax.step(df_impl["mpix"], df_impl["best_t"], where="mid", color=color, label=impl)
        ax.plot(df_impl["mpix"], df_impl["best_t"], marker="o", color=color, linestyle="none")

    mpix_values = sorted(df_best_mode["mpix"].dropna().unique())
    _set_mp_ticks(ax, mpix_values)
    ax.set_xlabel("MP")
    ax.set_ylabel("Best threads")
    title = "Best threads vs MP"
    if mode_used is not None:
        title = f"{title} ({mode_used})"
    ax.set_title(title)
    best_ticks = sorted(df_best_mode["best_t"].dropna().unique())
    if best_ticks:
        ax.set_yticks(best_ticks)
        ax.set_yticklabels([str(int(t)) for t in best_ticks])
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = Path(out_dir) / "plots" / f"best_threads_vs_mpix{_mode_suffix(mode_used, multiple)}.png"
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_tables(
    df: pd.DataFrame,
    df_agg: pd.DataFrame,
    df_best: pd.DataFrame,
    out_dir: str | os.PathLike[str],
    mode: Optional[str] = None,
) -> None:
    df_mode, mode_used, multiple = _filter_mode(df, mode)
    df_agg_mode, _, _ = _filter_mode(df_agg, mode_used)
    df_best_mode, _, _ = _filter_mode(df_best, mode_used)
    if df_agg_mode.empty:
        return

    out_dir = Path(out_dir) / "tables"
    _ensure_dir(out_dir)

    seq = df_agg_mode[df_agg_mode["strategy"].isin(["scalar_seq", "simd_seq"])]
    seq = seq[["mode", "mpix", "impl", "median_total_ns"]].rename(
        columns={"median_total_ns": "seq_total_ns"}
    )
    scalar_seq = seq[seq["impl"] == "scalar"][["mode", "mpix", "seq_total_ns"]].rename(
        columns={"seq_total_ns": "scalar_seq_total_ns"}
    )

    table_a = df_best_mode.merge(seq, on=["mode", "mpix", "impl"], how="left")
    table_a = table_a.merge(scalar_seq, on=["mode", "mpix"], how="left")
    table_a["speedup_vs_seq"] = table_a["seq_total_ns"] / table_a["best_total_ns"]
    table_a["speedup_vs_scalar_seq"] = table_a["scalar_seq_total_ns"] / table_a["best_total_ns"]
    table_a = table_a[
        ["mode", "mpix", "impl", "best_t", "best_total_ns", "speedup_vs_seq", "speedup_vs_scalar_seq"]
    ].sort_values(["mode", "mpix", "impl"])
    table_a.to_csv(out_dir / "best_threads_per_mpix.csv", index=False)

    df_par = df_agg_mode[df_agg_mode["strategy"].isin(["scalar_par", "simd_par"])]
    df_par = df_par.merge(seq, on=["mode", "mpix", "impl"], how="left")
    df_par["speedup_vs_seq"] = df_par["seq_total_ns"] / df_par["median_total_ns"]
    idx = df_par.groupby(["mode", "mpix", "impl"])["speedup_vs_seq"].idxmax()
    table_b = df_par.loc[idx, ["mode", "mpix", "impl", "speedup_vs_seq", "num_threads"]].rename(
        columns={"speedup_vs_seq": "max_speedup", "num_threads": "argmax_threads"}
    )
    table_b = table_b.sort_values(["mode", "mpix", "impl"])
    table_b.to_csv(out_dir / "max_speedup_per_mpix.csv", index=False)

    seq_scalar = seq[seq["impl"] == "scalar"].rename(columns={"seq_total_ns": "scalar_seq_total_ns"})
    seq_simd = seq[seq["impl"] == "simd"].rename(columns={"seq_total_ns": "simd_seq_total_ns"})
    table_c = seq_scalar.merge(seq_simd, on=["mode", "mpix"], how="outer")
    table_c["simd_benefit_seq"] = table_c["scalar_seq_total_ns"] / table_c["simd_seq_total_ns"]

    best_scalar = df_best_mode[df_best_mode["impl"] == "scalar"][["mode", "mpix", "best_total_ns"]].rename(
        columns={"best_total_ns": "scalar_best_total_ns"}
    )
    best_simd = df_best_mode[df_best_mode["impl"] == "simd"][["mode", "mpix", "best_total_ns"]].rename(
        columns={"best_total_ns": "simd_best_total_ns"}
    )
    table_c = table_c.merge(best_scalar, on=["mode", "mpix"], how="left")
    table_c = table_c.merge(best_simd, on=["mode", "mpix"], how="left")
    table_c["simd_benefit_best_par"] = table_c["scalar_best_total_ns"] / table_c["simd_best_total_ns"]
    table_c = table_c.sort_values(["mode", "mpix"])
    table_c.to_csv(out_dir / "simd_benefit.csv", index=False)

def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _match_mpix_values(available: List[float], desired: List[float]) -> List[float]:
    if not available:
        return []
    arr = np.asarray(available, dtype=float)
    matched = []
    used_idx = set()
    for target in desired:
        idx = int(np.argmin(np.abs(arr - target)))
        if idx in used_idx:
            continue
        val = float(arr[idx])
        tol = max(0.05, 0.02 * float(target))
        if abs(val - target) <= tol:
            matched.append(val)
            used_idx.add(idx)
    return matched


def _filter_mode(df: pd.DataFrame, mode: Optional[str]) -> Tuple[pd.DataFrame, Optional[str], bool]:
    if df is None or df.empty or "mode" not in df.columns:
        return df, mode, False

    modes_series = df["mode"].dropna()
    if modes_series.empty:
        return df, mode, False

    mode_counts = modes_series.value_counts()
    modes = list(mode_counts.index)
    multiple = len(modes) > 1

    if mode is None:
        mode = modes[0]
        return df[df["mode"] == mode], mode, multiple

    return df[df["mode"] == mode], mode, multiple


def _mode_suffix(mode: Optional[str], multiple: bool) -> str:
    if mode is None or not multiple:
        return ""
    safe = str(mode).replace(" ", "_")
    safe = safe.replace("/", "_")
    return f"_mode-{safe}"


def _mp_tick_label(mpix: float) -> str:
    if pd.isna(mpix):
        return ""
    value = float(mpix)
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return str(int(rounded))
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def _set_mp_ticks(ax: plt.Axes, mpix_values: Iterable[float]) -> None:
    values = sorted({float(v) for v in mpix_values if not pd.isna(v)})
    if not values:
        return
    ax.set_xticks(values)
    ax.set_xticklabels([_mp_tick_label(v) for v in values])


def _mp_title(mpix: float) -> str:
    if pd.isna(mpix):
        return ""
    return str(int(round(float(mpix))))


def _stagger_ytick_labels(ax: plt.Axes, step: float = 0.02) -> None:
    labels = ax.get_yticklabels()
    for idx, label in enumerate(labels):
        label.set_horizontalalignment("right")
        if idx == 1:
            label.set_x(-step)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mpix_label(mpix: float) -> str:
    if float(mpix).is_integer():
        return f"{int(mpix)}mpix"
    text = str(mpix).replace(".", "p")
    return f"{text}mpix"
