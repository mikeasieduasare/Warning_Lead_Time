# =================================================================
# PUBLICATION UTILITIES
# Clean, consistent styling for tables (CSV + Markdown) and figures
# (matplotlib, error bars, dual-axis fixes, 300dpi) suitable for direct
# inclusion in the paper, rather than default matplotlib/pandas output.
# =================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = {
    "accuracy": "#1f5fa8",
    "threshold": "#111111",
    "ac1": "#1b7a3d",
    "onset": "#c0392b",
    "shell": "#8e44ad",
    "msp": "#e08214",
    "energy": "#2b8cbe",
    "mahalanobis": "#c51b7d",
    "mcdropout": "#6a994e",
    "variance": "#1f5fa8",
    "skewness": "#c0392b",
    "ci_band": "#1f5fa8",
}

# Distinct, colorblind-friendly colors for the multi-architecture summary
# figure -- one per architecture, consistent across every plot that shows
# more than one architecture at once.
ARCH_PALETTE = ["#1f5fa8", "#c0392b", "#1b7a3d", "#8e44ad", "#e08214", "#2b8cbe"]


def set_publication_style():
    """Consistent, readable rcParams for all figures in this pipeline."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.constrained_layout.use": True,
    })


def export_table(df, output_dir, name):
    """Saves a DataFrame as both CSV (for further analysis) and Markdown
    (for direct paste into the paper) under a consistent naming scheme."""
    csv_path = os.path.join(output_dir, f"{name}.csv")
    md_path = os.path.join(output_dir, f"{name}.md")
    df.to_csv(csv_path, index=False)
    try:
        with open(md_path, "w") as f:
            f.write(df.to_markdown(index=False))
    except ImportError:
        # tabulate not installed; CSV is still saved, markdown skipped gracefully
        with open(md_path, "w") as f:
            f.write(df.to_string(index=False))
    return csv_path, md_path


def plot_accuracy_by_seed(ax, alphas, accuracy_by_seed, threshold, title):
    """Accuracy vs alpha with per-seed lines (thin) + mean (bold) + failure threshold."""
    accs = np.array(accuracy_by_seed)  # (n_seeds, n_alphas)
    mean_acc = accs.mean(axis=0)
    for row in accs:
        ax.plot(alphas, row, color=PALETTE["accuracy"], alpha=0.25, linewidth=1)
    ax.plot(alphas, mean_acc, color=PALETTE["accuracy"], linewidth=2.2, label="Mean across seeds")
    ax.axhline(threshold, color=PALETTE["threshold"], linestyle="--", linewidth=1.3, label=f"Failure threshold ({threshold})")
    ax.set_title(title)
    ax.set_xlabel(r"Shift intensity $\alpha$")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best", frameon=True)


def plot_detrended_ac1(ax, alphas, ac1_by_seed, onset_alpha, title):
    """Detrended windowed AC1 vs alpha, per-seed + mean, with onset marker."""
    ac1s = np.array(ac1_by_seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_ac1 = np.nanmean(ac1s, axis=0)
    for row in ac1s:
        ax.plot(alphas, row, color=PALETTE["ac1"], alpha=0.25, linewidth=1)
    ax.plot(alphas, mean_ac1, color=PALETTE["ac1"], linewidth=2.2, label="Mean detrended AC1")
    if onset_alpha is not None and not (isinstance(onset_alpha, float) and np.isnan(onset_alpha)):
        ax.axvline(onset_alpha, color=PALETTE["onset"], linestyle=":", linewidth=1.6,
                    label=f"EWS onset (α={onset_alpha:.2f})")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(r"Shift intensity $\alpha$")
    ax.set_ylabel("Detrended AC1 (residual)")
    ax.legend(loc="best", frameon=True)


def plot_threshold_sensitivity(ax, threshold_rows, title):
    """alpha* and lead_time vs tau, with 95% CI error bars."""
    taus = [r["threshold"] for r in threshold_rows]
    alpha_stars = [r["alpha_star"] for r in threshold_rows]
    lead_times = [r["lead_time"] for r in threshold_rows]
    a_lo = [r["alpha_star_ci95"][0] for r in threshold_rows]
    a_hi = [r["alpha_star_ci95"][1] for r in threshold_rows]
    l_lo = [r["lead_time_ci95"][0] for r in threshold_rows]
    l_hi = [r["lead_time_ci95"][1] for r in threshold_rows]

    a_err = [np.array(alpha_stars) - np.array(a_lo), np.array(a_hi) - np.array(alpha_stars)]
    l_err = [np.array(lead_times) - np.array(l_lo), np.array(l_hi) - np.array(lead_times)]

    ax.errorbar(taus, alpha_stars, yerr=a_err, marker="o", capsize=4,
                color=PALETTE["accuracy"], label=r"$\alpha^*$ (95% CI)")
    ax.errorbar(taus, lead_times, yerr=l_err, marker="s", capsize=4,
                color=PALETTE["onset"], label=r"$\Delta\alpha$ (95% CI)")
    ax.set_title(title)
    ax.set_xlabel(r"Clinical acceptability threshold $\tau$")
    ax.set_ylabel(r"$\alpha$")
    ax.legend(loc="best", frameon=True)


def plot_detector_comparison(ax, detector_names, blocked_fractions, ci_lowers, ci_uppers, title):
    """Bar chart of blocked-failure fraction per detector, with 95% CI error bars."""
    x = np.arange(len(detector_names))
    errs = [np.array(blocked_fractions) - np.array(ci_lowers),
            np.array(ci_uppers) - np.array(blocked_fractions)]
    colors = [PALETTE.get(n.lower(), "#666666") for n in detector_names]
    ax.bar(x, blocked_fractions, yerr=errs, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(detector_names, rotation=20, ha="right")
    ax.set_ylabel("Fraction of failures blocked")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)


def plot_pairwise_significance_heatmap(ax, names, p_matrix, title):
    """Heatmap of pairwise permutation-test p-values across architectures."""
    im = ax.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(p_matrix[i, j]):
                ax.text(j, i, f"{p_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)


def plot_summary_figure_1_reliability(ax, alphas, curves_by_arch, alpha_star_by_arch,
                                       alpha_ews_by_arch, sustained_by_arch, threshold,
                                       metric_name="Accuracy", show_warning_band=False):
    """
    SUMMARY FIGURE 1 of 2: every architecture's headline-metric trajectory
    (mean across seeds) on one plot. Deliberately MINIMAL -- axis labels,
    a legend (architecture -> color), and the failure threshold line are
    the only labeling; there is NO in-plot annotation text (no callouts,
    no value labels) cluttering the data area. Distinct in form (line
    plot) and color scheme (ARCH_PALETTE) from Figure 2, so the two are
    immediately distinguishable at a glance.

    show_warning_band (default False): a shaded (alpha_ews -> alpha_star)
    band is DISABLED by default and should stay off whenever the plotted
    curve is a MEAN ACROSS SEEDS with seed-dependent collapse. Reason: a
    warning-window band only has coherent meaning for a SINGLE trajectory
    with a genuine (alpha_ews, alpha_star) pair. When alpha_star_by_arch
    and alpha_ews_by_arch are averaged across seeds -- especially when
    some seeds genuinely collapse and others are censored at alpha_star=1
    -- the averaged band edges correspond to no real trajectory and do
    not line up with where the mean curve actually crosses the threshold,
    which is misleading in exactly the way a careful reviewer would catch.
    Only enable this for a genuinely single-trajectory plot where a true
    per-seed (alpha_ews, alpha_star) pair exists.
    """
    for i, (arch_name, curve) in enumerate(curves_by_arch.items()):
        color = ARCH_PALETTE[i % len(ARCH_PALETTE)]
        ax.plot(alphas, curve, color=color, linewidth=2.6, label=arch_name, zorder=3)

        if show_warning_band:
            sustained = sustained_by_arch.get(arch_name, False)
            a_star = alpha_star_by_arch.get(arch_name)
            a_ews = alpha_ews_by_arch.get(arch_name)
            if sustained and a_star is not None and a_ews is not None and a_star > a_ews:
                ax.axvspan(a_ews, a_star, color=color, alpha=0.14, zorder=1)

    ax.axhline(threshold, color=PALETTE["threshold"], linestyle="--", linewidth=1.6,
               label="Clinical acceptability threshold", zorder=2)
    ax.set_xlabel(r"Shift intensity $\alpha$", fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.set_xlim(alphas.min(), alphas.max())


def plot_summary_figure_2_detector_effectiveness(ax, arch_names, detector_names,
                                                  blocked_fraction_matrix):
    """
    SUMMARY FIGURE 2 of 2: grouped bar chart of blocked-failure fraction
    per detector, per architecture -- the Safety Shell vs. every OOD
    baseline, in one glance. Same minimal-labeling philosophy as Figure 1:
    axis labels + legend only, no per-bar value text. Distinct in form
    (bars, not lines) and color scheme (per-detector palette, not
    per-architecture) from Figure 1, so the two summary figures are
    immediately distinguishable and cannot be confused with one another.

    `blocked_fraction_matrix`: (n_arch, n_detectors) array, mean blocked
    fraction per architecture x detector.
    """
    n_arch = len(arch_names)
    n_det = len(detector_names)
    x = np.arange(n_arch)
    width = 0.8 / n_det

    for j, det_name in enumerate(detector_names):
        color = PALETTE.get(det_name.lower().replace(" ", ""), "#666666")
        offset = (j - (n_det - 1) / 2) * width
        ax.bar(x + offset, blocked_fraction_matrix[:, j], width=width * 0.92,
               color=color, label=det_name)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, fontsize=11)
    ax.set_ylabel("Fraction of failures blocked", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5, ncol=1)


# =================================================================
# RAW-SIGNAL PERSISTENCE  (survives a lost/restarted kernel session)
# =================================================================
def save_raw_signals_npz(output_dir, arch_name, seed, signals_dict, prefix=""):
    """
    Persists one (architecture, seed) run's signals to disk immediately
    after computation. As of spec v3 this includes BOTH the compact
    alpha-indexed mean signals (for onset detection and plotting) AND the
    full per-image matrices of shape (n_alpha, n_panel) -- entropy,
    correctness, and each detector's per-image scores -- which are
    required for the hierarchical bootstrap, potentially-intercepted-
    failure counts, and the real-data critical-slowing-down audit.

    np.savez stores 2D arrays natively, so per-image matrices are saved
    losslessly. The size cost is modest (20 alpha x 400 panel x a handful
    of matrices per run) and is what makes every downstream re-analysis
    recomputable without re-running training/inference.

    `prefix`: optional filename prefix (e.g. "linshift_") so multiple
    parallel experiments' raw signals can coexist in the same output
    directory without overwriting each other.
    """
    path = os.path.join(output_dir, f"{prefix}raw_signals_{arch_name}_seed{seed}.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in signals_dict.items() if v is not None})
    return path


def load_raw_signals(output_dir, arch_names, seeds, prefix=""):
    """
    Reconstructs the raw_signals dict from disk (written by
    save_raw_signals_npz during a prior run), for use when the original
    Python session/kernel is no longer available. Returns the SAME
    structure as results["raw_signals"] in the main pipeline.

    `prefix`: must match whatever prefix was used when saving (e.g.
    "linshift_" for the linear-blend experiment's files).
    """
    raw_signals = {arch: [] for arch in arch_names}
    for arch_name in arch_names:
        for seed in seeds:
            path = os.path.join(output_dir, f"{prefix}raw_signals_{arch_name}_seed{seed}.npz")
            if os.path.isfile(path):
                data = np.load(path)
                raw_signals[arch_name].append({k: data[k] for k in data.files})
            else:
                print(f"[load_raw_signals] Warning: missing file for {arch_name} seed {seed}: {path}")
    return raw_signals
    return im
