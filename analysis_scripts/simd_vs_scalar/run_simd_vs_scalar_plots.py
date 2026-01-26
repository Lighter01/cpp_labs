#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from plots import (
    make_all_plots,
    plot_grouped_histograms,
    plot_metric_vs_megapixels,
    plot_stage_boxplots_side_by_side,
    plot_stage_mean_pies,
    plot_stacked_mean_bars,
)

PLOTLY_IMAGE_SCALE = 2.5
PLOTLY_DEFAULT_WIDTH = 1200
PLOTLY_DEFAULT_HEIGHT = 800


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SIMD vs scalar plots for blend and histogram results.")
    parser.add_argument("out_dir", help="Output directory for plots")
    parser.add_argument(
        "--blend-scalar",
        default="results_release/blend_results_scalar_seq/csv/blend_performance__scalar_over_seq_g0_a0.70.csv",
        help="CSV with alpha-blend scalar results (default: results_release/...scalar_over_seq...csv)",
    )
    parser.add_argument(
        "--blend-simd",
        default="results_release/blend_results_simd_seq/csv/blend_performance__simd_over_seq_g0_a0.70.csv",
        help="CSV with alpha-blend SIMD results (default: results_release/...simd_over_seq...csv)",
    )
    parser.add_argument(
        "--hist-scalar",
        default="results_release/hist_results_scalar/histogram_performance_scalar.csv",
        help="CSV with histogram scalar results (default: results_release/hist_results_scalar/histogram_performance_scalar.csv)",
    )
    parser.add_argument(
        "--hist-simd",
        default="results_release/hist_results_simd/histogram_performance_simd.csv",
        help="CSV with histogram SIMD results (default: results_release/hist_results_simd/histogram_performance_simd.csv)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    blend_out = out_dir / "alpha_blend"
    hist_out = out_dir / "histogram"
    blend_out.mkdir(parents=True, exist_ok=True)
    hist_out.mkdir(parents=True, exist_ok=True)

    _require_file(args.blend_scalar)
    _require_file(args.blend_simd)
    _require_file(args.hist_scalar)
    _require_file(args.hist_simd)

    df_blend_scalar = pd.read_csv(args.blend_scalar)
    df_blend_simd = pd.read_csv(args.blend_simd)
    df_hist_scalar = pd.read_csv(args.hist_scalar)
    df_hist_simd = pd.read_csv(args.hist_simd)

    _run_make_all_plots(
        df_blend_scalar,
        df_blend_simd,
        metric="total_ns",
        out_dir=blend_out,
        prefix="blend_total_ns",
    )

    # Alpha-blend stage breakdown plots
    stage_box = plot_stage_boxplots_side_by_side(
        df_blend_scalar,
        df_blend_simd,
        kind="ns",
        title="Alpha-blend stage breakdown (ns)",
        box_width=0.35,
        height=640,
        clamp_min=None,
        show=False,
    )
    _save_plotly(stage_box, blend_out / "blend_stage_boxplots_ns.png")

    stage_bars = plot_stacked_mean_bars(
        df_blend_scalar,
        df_blend_simd,
        kind="ns",
        title="Alpha-blend mean stage breakdown (stacked)",
        show=False,
    )
    _save_matplotlib(stage_bars, blend_out / "blend_stage_mean_stacked.png")

    stage_pies = plot_stage_mean_pies(
        df_blend_scalar,
        df_blend_simd,
        kind="ns",
        title="Alpha-blend mean composition (ns)",
        show=False,
    )
    _save_matplotlib(stage_pies, blend_out / "blend_stage_mean_pies.png")

    grouped = plot_grouped_histograms(
        df_blend_scalar,
        df_blend_simd,
        group_col="out_size",
        metric="total_ns",
        bins=30,
        show=False,
    )
    if grouped:
        grouped_dir = blend_out / "grouped_histograms"
        grouped_dir.mkdir(parents=True, exist_ok=True)
        for key, (fig, _) in grouped.items():
            label = _sanitize_key(key)
            path = grouped_dir / f"blend_grouped_total_ns_{label}.png"
            _save_matplotlib(fig, path)

    for metric in ["preprocess_ns", "blend_ns", "postprocess_ns", "total_ns"]:
        fig = plot_metric_vs_megapixels(
            df_blend_scalar,
            df_blend_simd,
            metric=metric,
            show=False,
        )
        _save_plotly(fig, blend_out / f"blend_metric_vs_mp_{metric}.png")

    _run_make_all_plots(
        df_hist_scalar,
        df_hist_simd,
        metric="timing_ns",
        out_dir=hist_out,
        prefix="hist_timing_ns",
    )

    print(f"Saved plots to: {out_dir}")
    return 0


def _run_make_all_plots(
    df_scalar: pd.DataFrame,
    df_simd: pd.DataFrame,
    *,
    metric: str,
    out_dir: Path,
    prefix: str,
) -> None:
    figs = make_all_plots(
        df_scalar,
        df_simd,
        metric=metric,
        nbins=50,
        show=False,
        return_figs=True,
    )
    if not figs:
        return

    hist_fig, _ = figs["hist_seaborn"]
    _save_matplotlib(hist_fig, out_dir / f"{prefix}_hist_kde.png")

    _save_plotly(figs["percentile_plotly"], out_dir / f"{prefix}_percentile.png")
    _save_plotly(figs["boxplot_plotly"], out_dir / f"{prefix}_boxplot.png")

    _save_matplotlib(figs["metric_by_index"], out_dir / f"{prefix}_by_index.png")
    _save_matplotlib(figs["cumulative_metric"], out_dir / f"{prefix}_cumulative.png")


def _save_matplotlib(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_plotly(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        print(f"Skipping empty plotly figure: {path}")
        return
    width = int(fig.layout.width) if fig.layout.width else PLOTLY_DEFAULT_WIDTH
    height = int(fig.layout.height) if fig.layout.height else PLOTLY_DEFAULT_HEIGHT
    try:
        fig.write_image(str(path), scale=PLOTLY_IMAGE_SCALE, width=width, height=height)
        return
    except Exception:  # pylint: disable=broad-except
        html_path = path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"Plotly image export failed; wrote HTML: {html_path}")


def _require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")


def _sanitize_key(key) -> str:
    text = "unknown" if key is None else str(key)
    text = text.replace(" ", "_").replace("/", "_")
    text = text.replace("\\", "_")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
