# ============================================================
# STRENGTHENING PLAN — EXPERIMENT A (hyperparameter sensitivity grid) AND
# EXPERIMENT B (Page-Hinkley adjacent-baseline comparison)
#
# Runs entirely against SAVED npz files. No GPU, no retraining, no new
# images or seeds. Reuses the manuscript's own established, verified
# functions (calibrate_population_threshold, apply_onset_threshold,
# detect_persistent_failure, _auc_from_scores, windowed_ews_indicators,
# detrend_series) imported directly from enhanced_ews_pipeline.py, rather
# than reimplementing this logic — the exact same code path that produced
# every other calibrated result in the manuscript.
#
# HARD REQUIREMENT: this script verifies every array it needs is actually
# present, with a sane shape, in every required npz file BEFORE running
# any analysis. If anything required is missing or malformed, the script
# prints exactly what is missing from which file and EXITS without
# computing or printing any result. A partial or silently-wrong result is
# worse than no result; this script is designed to fail loudly, not
# gracefully degrade.
#
# HONEST LIMITATION, stated up front rather than glossed over: the
# false-alarm-rate measurement below reconstructs the "joint
# dependence-preserving surrogate" procedure described in Section 4.12 —
# the same resampled nominal-region time points applied across every
# signal, so cross-detector correlation present in the real data is not
# destroyed by independent per-detector resampling. This is a faithful,
# clearly-documented reconstruction of that DESCRIBED procedure, built
# from the same calibration primitives used everywhere else in the
# pipeline. It is not a byte-for-byte replay of whatever specific script
# originally produced Table 2's exact false-alarm figures, which was not
# available when this script was written. If that original script turns
# up, its numbers should be treated as authoritative over this script's,
# and any discrepancy is worth investigating rather than silently
# preferring one over the other.
# ============================================================

import sys, os, glob, json, shutil
import numpy as np

# ------------------------------------------------------------------
# 0. Locate and import the manuscript's own established pipeline code.
#    We do not reimplement calibration/onset-detection logic: we import
#    the exact functions that produced every other calibrated result in
#    the manuscript, so this analysis inherits their correctness (and
#    any bugs) rather than introducing a second, possibly-inconsistent
#    implementation.
# ------------------------------------------------------------------
PIPELINE_CODE = None
for p in glob.glob("/kaggle/input/**/enhanced_ews_pipeline.py", recursive=True):
    if "_auc_from_scores" in open(p).read():
        PIPELINE_CODE = p
        break

if PIPELINE_CODE is None:
    print("FATAL: enhanced_ews_pipeline.py (with _auc_from_scores) was not found")
    print("       under /kaggle/input/. This script imports its calibration and")
    print("       onset-detection functions directly rather than reimplementing")
    print("       them, so it cannot proceed without this file.")
    print("       Upload enhanced_ews_pipeline.py as a dataset/input and rerun.")
    sys.exit(1)

shutil.copy(PIPELINE_CODE, "/kaggle/working/enhanced_ews_pipeline.py")
sys.path.insert(0, "/kaggle/working")
from enhanced_ews_pipeline import (
    calibrate_population_threshold,
    apply_onset_threshold,
    detect_persistent_failure,
    _auc_from_scores,
    windowed_ews_indicators,
    detrend_series,
)
print(f"Using pipeline code: {PIPELINE_CODE}\n")

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


# ------------------------------------------------------------------
# 1. Experiment scope: the two linear-blend experiments only, per the
#    strengthening plan's own reasoning — these are the only cells with
#    enough confirmed failures (9 each) for coverage to move measurably
#    under a hyperparameter sweep. FDA experiments are excluded from
#    Experiment A/B for this reason, not by oversight.
# ------------------------------------------------------------------
EXPERIMENTS = [
    ("Linear-radiography",    "linshift_raw_signals_", False),  # is_auc=False -> accuracy
    ("Linear-histopathology", "exp2lin_raw_signals_",  True),   # is_auc=True  -> AUROC
]
ARCHS = ["ResNet18", "DenseNet121", "ViT_B16"]
REP_DETECTORS = [("Mahalanobis", "mahalanobis_matrix", False),
                  ("Energy",      "energy_matrix",      False),
                  ("MCDropout",   "mc_dropout_var_matrix", False)]
TAU = 0.70          # acceptability threshold, fixed throughout -- does not vary with
                    # the hyperparameters being tested; it is a property of R(alpha).
FAILURE_PERSIST = 2  # persistence used for alpha* itself -- always 2, per Section 4.2.
                    # Only the WARNING-side persistence (onset detection) varies below.
NOMINAL_FPR = 0.05
N_SURROGATES = 400
CALIBRATION_SEED = 0


# ------------------------------------------------------------------
# 2. STRICT VERIFICATION GATE.
#    Every key below must be present, with the expected shape, in every
#    file belonging to the two linear-blend experiments. If anything is
#    missing, we print a precise, per-file diagnostic and exit(1) before
#    touching any analysis code. This is deliberately unforgiving: a
#    missing key should stop the whole script, not just skip one run
#    silently.
# ------------------------------------------------------------------
REQUIRED_KEYS_COMMON = ["alphas", "correct_matrix", "entropy_matrix",
                         "mahalanobis_matrix", "energy_matrix", "mc_dropout_var_matrix"]
REQUIRED_KEYS_AUC_ONLY = ["prob_matrix", "panel_labels"]  # additionally required when is_auc=True


def discover_files():
    all_npz = glob.glob("/kaggle/input/**/*.npz", recursive=True)
    if not all_npz:
        print("FATAL: no .npz files found anywhere under /kaggle/input/.")
        print("       Attach the dataset containing the saved raw-signal files")
        print("       and rerun.")
        sys.exit(1)
    return all_npz


def verify_and_load(all_npz):
    """
    Returns {experiment_name: {"is_auc": bool, "runs": {arch: [run_dict, ...]}}}
    ONLY if every required key, with a sane shape, is present in every file
    for every experiment. Otherwise prints a precise report of what is
    missing from where, and exits without returning anything.
    """
    failures = []      # list of human-readable problem descriptions
    loaded = {}

    for name, prefix, is_auc in EXPERIMENTS:
        files = [f for f in all_npz if os.path.basename(f).startswith(prefix)]
        if len(files) != 9:
            failures.append(
                f"[{name}] expected 9 run files (3 architectures x 3 seeds) "
                f"with prefix '{prefix}', found {len(files)}."
            )
            continue

        runs = {a: [] for a in ARCHS}
        required = REQUIRED_KEYS_COMMON + (REQUIRED_KEYS_AUC_ONLY if is_auc else [])

        for f in sorted(files):
            arch = next((a for a in ARCHS if a in os.path.basename(f)), None)
            if arch is None:
                failures.append(
                    f"[{name}] file '{os.path.basename(f)}' does not contain any "
                    f"recognised architecture name ({ARCHS}) in its filename."
                )
                continue

            try:
                z = np.load(f, allow_pickle=True)
            except Exception as e:
                failures.append(f"[{name}] '{os.path.basename(f)}' failed to load: {e}")
                continue

            missing = [k for k in required if k not in z.files]
            if missing:
                failures.append(
                    f"[{name}] '{os.path.basename(f)}' is missing required key(s): "
                    f"{missing}. Present keys: {list(z.files)}"
                )
                continue

            # Shape sanity: alphas 1-D length >= 6; matrices 2-D with matching
            # n_alpha rows and a shared, positive panel size in columns.
            try:
                alphas = np.asarray(z["alphas"])
                n_alpha = alphas.shape[0]
                if alphas.ndim != 1 or n_alpha < 6:
                    failures.append(
                        f"[{name}] '{os.path.basename(f)}': 'alphas' has unexpected "
                        f"shape {alphas.shape} (expected 1-D, length >= 6)."
                    )
                    continue

                panel_size = None
                shape_ok = True
                for key in ["correct_matrix", "entropy_matrix", "mahalanobis_matrix",
                            "energy_matrix", "mc_dropout_var_matrix"] + (
                            ["prob_matrix"] if is_auc else []):
                    arr = np.asarray(z[key])
                    if arr.ndim != 2 or arr.shape[0] != n_alpha:
                        failures.append(
                            f"[{name}] '{os.path.basename(f)}': '{key}' has shape "
                            f"{arr.shape}, expected 2-D with {n_alpha} rows "
                            f"(matching 'alphas')."
                        )
                        shape_ok = False
                        continue
                    if panel_size is None:
                        panel_size = arr.shape[1]
                    elif arr.shape[1] != panel_size:
                        failures.append(
                            f"[{name}] '{os.path.basename(f)}': '{key}' has "
                            f"{arr.shape[1]} columns, expected {panel_size} "
                            f"(mismatch against other arrays in the same file)."
                        )
                        shape_ok = False
                if is_auc:
                    labels = np.asarray(z["panel_labels"]).ravel()
                    if panel_size is not None and labels.shape[0] not in (panel_size, n_alpha * panel_size):
                        failures.append(
                            f"[{name}] '{os.path.basename(f)}': 'panel_labels' has "
                            f"{labels.shape[0]} entries, incompatible with panel "
                            f"size {panel_size}."
                        )
                        shape_ok = False
                if not shape_ok:
                    continue
            except Exception as e:
                failures.append(f"[{name}] '{os.path.basename(f)}': shape check raised {e}")
                continue

            runs[arch].append({k: z[k] for k in z.files})

        n_loaded = sum(len(v) for v in runs.values())
        if n_loaded == 9 and name not in [f.split(']')[0][1:] for f in failures]:
            loaded[name] = {"is_auc": is_auc, "runs": runs}

    if failures:
        print("=" * 70)
        print("VERIFICATION FAILED. The following problem(s) were found:")
        print("=" * 70)
        for f in failures:
            print(f"  - {f}")
        print()
        print("No analysis will be run. Fix the data availability issue(s) above")
        print("and rerun this script. Nothing below this point has executed.")
        sys.exit(1)

    print("Verification passed: all required arrays present with sane shapes")
    print(f"for both linear-blend experiments (9 runs each).\n")
    return loaded


# ------------------------------------------------------------------
# 3. Shared helpers
# ------------------------------------------------------------------
def perf_curve(run, is_auc):
    """R(alpha) for one run: AUROC for histopathology, accuracy for radiography."""
    if is_auc:
        pm = np.asarray(run["prob_matrix"], dtype=float)
        y = np.asarray(run["panel_labels"]).ravel()
        return np.asarray([_auc_from_scores(y, pm[i, :]) for i in range(pm.shape[0])])
    return np.asarray(run["correct_matrix"], dtype=float).mean(axis=1)


def loro_calibrate_and_apply(all_run_signals, alphas, persistence, serial_dependence=False,
                               nominal_fpr=NOMINAL_FPR, seed=CALIBRATION_SEED):
    """
    Leave-one-run-out calibration and onset detection for ONE detector across
    all runs of one experiment. Each run's threshold is calibrated from the
    OTHER 8 runs' pooled nominal-region data only, matching the manuscript's
    primary calibration convention (never a run's own data).
    """
    n_runs = len(all_run_signals)
    onsets = []
    for held_out in range(n_runs):
        other = [all_run_signals[j] for j in range(n_runs) if j != held_out]
        thr = calibrate_population_threshold(
            other, persistence=persistence, nominal_fpr=nominal_fpr,
            n_surrogates=N_SURROGATES, seed=seed, serial_dependence=serial_dependence)
        onset = apply_onset_threshold(alphas, all_run_signals[held_out], thr,
                                       persistence=persistence, nominal_fpr=nominal_fpr)
        onset["threshold"] = thr
        onsets.append(onset)
    return onsets


def joint_representation_surrogate_far(pooled_reps, thresholds, full_len, persistence,
                                        n_draws=1000, seed=1):
    """
    Empirical false-alarm rate for the representation family's fusion rule
    (earliest sustained crossing among Mahalanobis/Energy/MCDropout), using
    ONE shared resampled index sequence per draw across all three detectors
    -- a genuine, complete reconstruction of "the same resampled
    nominal-region time points applied to every signal" for this family,
    since all three are i.i.d.-resampled by the manuscript's own convention.
    pooled_reps: {"Mahalanobis": 1-D array, "Energy": 1-D array, "MCDropout": 1-D array},
    all built from concatenating the SAME set of runs in the SAME order.
    """
    rng = np.random.RandomState(seed)
    n_pool = len(next(iter(pooled_reps.values())))
    fires = 0
    for _ in range(n_draws):
        idx = rng.randint(0, n_pool, size=full_len)
        onset_alphas = []
        for name, pooled in pooled_reps.items():
            surrogate = pooled[idx]
            above = surrogate > thresholds[name]
            run = 0
            fired_here = False
            for a in above:
                run = run + 1 if a else 0
                if run >= persistence:
                    fired_here = True
                    break
            if fired_here:
                onset_alphas.append(True)
        if any(onset_alphas):
            fires += 1
    return fires / n_draws


def ac1_block_surrogate_fires(nominal_segments, full_len, persistence, threshold,
                               window, sigma, rng):
    """
    ONE block-resampled AC1 null draw (mirrors calibrate_population_threshold's
    own block-resampling logic exactly, for consistency), re-detrended and
    re-windowed at the SAME (window, sigma) being tested, then checked against
    the given threshold under the given persistence. Returns True/False.
    """
    block = max(2, full_len // 4)
    out = []
    while len(out) < full_len:
        seg = nominal_segments[rng.randint(len(nominal_segments))]
        if len(seg) <= block:
            out.extend(seg)
        else:
            st = rng.randint(0, len(seg) - block)
            out.extend(seg[st:st + block])
    surrogate_entropy = np.array(out[:full_len])
    ac1 = windowed_ews_indicators(surrogate_entropy, window=window, detrend=True,
                                   detrend_sigma=sigma)["ac1"]
    ac1 = np.nan_to_num(ac1, nan=-np.inf)
    above = ac1 > threshold
    run = 0
    for a in above:
        run = run + 1 if a else 0
        if run >= persistence:
            return True
    return False


def run_experiment_a(loaded):
    """
    Hyperparameter sensitivity grid: window length in {3,5,7}, detrending
    sigma in {1.0,2.0,3.0}, onset-persistence in {1,2,3}, one factor at a
    time holding the other two at the manuscript's reported values
    (window=5, sigma=2.0, persistence=2). Reports coverage and mean
    false-alarm rate for both the dual-family and single-family fusion
    rules at every grid point, on both linear-blend experiments.
    """
    REPORTED = {"window": 5, "sigma": 2.0, "persistence": 2}
    GRID = {
        "window": [3, 5, 7],
        "sigma": [1.0, 2.0, 3.0],
        "persistence": [1, 2, 3],
    }

    results = {}
    for exp_name, exp_data in loaded.items():
        is_auc = exp_data["is_auc"]
        runs = exp_data["runs"]
        flat = [r for a in ARCHS for r in runs[a]]
        alphas = np.asarray(flat[0]["alphas"])
        full_len = len(alphas)
        n_base = max(4, int(full_len * 0.4))  # baseline_frac=0.4, matching calibrate_population_threshold's default

        # alpha* is fixed throughout -- computed once, not swept.
        confirmed_failures = []  # (run_index, alpha_star_idx) for genuinely confirmed failures
        for i, r in enumerate(flat):
            perf = perf_curve(r, is_auc)
            fail = detect_persistent_failure(alphas, perf, TAU, FAILURE_PERSIST)
            if fail["sustained"]:
                confirmed_failures.append((i, fail["idx"]))
        n_confirmed = len(confirmed_failures)

        exp_results = []
        for factor, values in GRID.items():
            for val in values:
                window = val if factor == "window" else REPORTED["window"]
                sigma = val if factor == "sigma" else REPORTED["sigma"]
                persistence = val if factor == "persistence" else REPORTED["persistence"]

                # --- dynamical family (AC1) at this (window, sigma) ---
                ac1_signals = []
                nominal_segments_ac1 = []
                for r in flat:
                    entropy = np.asarray(r["entropy_matrix"], dtype=float).mean(axis=1)
                    ac1 = windowed_ews_indicators(entropy, window=window, detrend=True,
                                                   detrend_sigma=sigma)["ac1"]
                    ac1_signals.append(ac1)
                    nominal_segments_ac1.append(entropy[:n_base])

                ac1_onsets = loro_calibrate_and_apply(ac1_signals, alphas, persistence,
                                                       serial_dependence=True)

                # --- representation family at this persistence ---
                rep_onsets_by_detector = {}
                pooled_rep_for_far = {}
                thresholds_rep = {}
                for dname, key, invert in REP_DETECTORS:
                    sigs = []
                    for r in flat:
                        s = np.asarray(r[key], dtype=float).mean(axis=1)
                        sigs.append(-s if invert else s)
                    onsets = loro_calibrate_and_apply(sigs, alphas, persistence,
                                                       serial_dependence=False)
                    rep_onsets_by_detector[dname] = onsets
                    # for FAR measurement: pool ALL 9 runs' nominal region (using
                    # each run's own LORO-excluded calibration is not meaningful
                    # for a single shared FAR estimate, so we use full-pool here,
                    # consistent with how Table 2's FAR is a per-experiment summary)
                    pooled_rep_for_far[dname] = np.concatenate([sg[:n_base] for sg in sigs])
                    thresholds_rep[dname] = np.median([o["threshold"] for o in onsets
                                                        if o.get("threshold") is not None])

                # combine representation onsets: earliest sustained crossing among the three
                rep_onset_alpha = []
                for run_idx in range(len(flat)):
                    candidates = [rep_onsets_by_detector[d][run_idx]["alpha_ews"]
                                  for d in rep_onsets_by_detector
                                  if rep_onsets_by_detector[d][run_idx]["alpha_ews"] is not None]
                    rep_onset_alpha.append(min(candidates) if candidates else None)

                dual_onset_alpha = []
                for run_idx in range(len(flat)):
                    a_dyn = ac1_onsets[run_idx]["alpha_ews"]
                    a_rep = rep_onset_alpha[run_idx]
                    if a_dyn is not None and a_rep is not None:
                        dual_onset_alpha.append(max(a_dyn, a_rep))
                    else:
                        dual_onset_alpha.append(None)

                # --- coverage: fraction of confirmed failures preceded by onset ---
                def coverage(onset_list):
                    warned = 0
                    for run_idx, astar_idx in confirmed_failures:
                        onset = onset_list[run_idx]
                        if onset is not None:
                            onset_idx = int(np.argmin(np.abs(alphas - onset)))
                            if onset_idx < astar_idx:
                                warned += 1
                    return warned / n_confirmed if n_confirmed else float("nan")

                cov_single = coverage(rep_onset_alpha)
                cov_dual = coverage(dual_onset_alpha)

                # --- false-alarm rate: joint surrogate reconstruction (see docstrings) ---
                far_single = joint_representation_surrogate_far(
                    pooled_rep_for_far, thresholds_rep, full_len, persistence)

                ac1_threshold_median = np.median([o["threshold"] for o in ac1_onsets
                                                   if o.get("threshold") is not None])
                rng_far = np.random.RandomState(2)
                dual_fires = 0
                n_draws_dual = 500
                for _ in range(n_draws_dual):
                    idx = rng_far.randint(0, len(pooled_rep_for_far["Mahalanobis"]), size=full_len)
                    rep_fired = False
                    for dname, pooled in pooled_rep_for_far.items():
                        surrogate = pooled[idx]
                        above = surrogate > thresholds_rep[dname]
                        run_ct = 0
                        for a in above:
                            run_ct = run_ct + 1 if a else 0
                            if run_ct >= persistence:
                                rep_fired = True
                                break
                        if rep_fired:
                            break
                    ac1_fired = ac1_block_surrogate_fires(
                        nominal_segments_ac1, full_len, persistence,
                        ac1_threshold_median, window, sigma, rng_far)
                    if rep_fired and ac1_fired:
                        dual_fires += 1
                far_dual = dual_fires / n_draws_dual

                exp_results.append({
                    "factor_varied": factor, "value": val,
                    "window": window, "sigma": sigma, "persistence": persistence,
                    "coverage_single": round(cov_single, 3),
                    "coverage_dual": round(cov_dual, 3),
                    "far_single": round(far_single, 3),
                    "far_dual": round(far_dual, 3),
                    "n_confirmed_failures": n_confirmed,
                })

        results[exp_name] = exp_results
    return results


# ------------------------------------------------------------------
# 4. Experiment B: Page-Hinkley adjacent-baseline comparison
# ------------------------------------------------------------------
def page_hinkley_statistic(series, delta):
    """
    Standard Page-Hinkley cumulative statistic: PH_t = m_t - min_{s<=t} m_s,
    where m_t = sum_{i=1}^{t} (x_i - xbar_t - delta). Detects an upward drift
    in the mean of `series` relative to its own running mean, allowing a
    `delta` magnitude of tolerated drift before accumulating.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    ph = np.full(n, np.nan)
    running_mean = series[0]
    m = 0.0
    m_min = 0.0
    for t in range(n):
        running_mean = running_mean + (series[t] - running_mean) / (t + 1)
        m = m + (series[t] - running_mean - delta)
        m_min = min(m_min, m)
        ph[t] = m - m_min
    return ph


def calibrate_and_apply_page_hinkley(all_run_entropy_means, nominal_segments, alphas,
                                       full_len, delta=0.05, nominal_fpr=NOMINAL_FPR,
                                       n_surrogates=N_SURROGATES, seed=CALIBRATION_SEED):
    """
    Leave-one-run-out calibration of the Page-Hinkley alarm threshold lambda,
    using the SAME block-resampled null construction as AC1 (Page-Hinkley
    operates on the same serially-dependent mean-entropy stream, so the
    same justification for block resampling over i.i.d. resampling applies
    here). No separate persistence rule is layered on top: Page-Hinkley's
    cumulative statistic already provides its own built-in memory, and
    adding a persistence-N requirement on top of an already-cumulative
    statistic would not be a fair, like-for-like comparison to how
    Page-Hinkley is used in the drift-detection literature this baseline
    is drawn from.
    """
    n_runs = len(all_run_entropy_means)
    rng = np.random.RandomState(seed)

    def make_block_surrogate(segments):
        block = max(2, full_len // 4)
        out = []
        while len(out) < full_len:
            seg = segments[rng.randint(len(segments))]
            if len(seg) <= block:
                out.extend(seg)
            else:
                st = rng.randint(0, len(seg) - block)
                out.extend(seg[st:st + block])
        return np.array(out[:full_len])

    onsets = []
    for held_out in range(n_runs):
        other_segments = [nominal_segments[j] for j in range(n_runs) if j != held_out]
        surrogates = [make_block_surrogate(other_segments) for _ in range(n_surrogates)]
        surrogate_ph_max = [np.nanmax(page_hinkley_statistic(sg, delta)) for sg in surrogates]
        qs = np.percentile(surrogate_ph_max, np.arange(50, 100.01, 0.5))
        candidates = np.append(qs, max(surrogate_ph_max) + 1e-6)
        lam = candidates[-1]
        for thr in candidates:
            rate = np.mean([1.0 if m > thr else 0.0 for m in surrogate_ph_max])
            if rate <= nominal_fpr:
                lam = thr
                break

        ph_series = page_hinkley_statistic(all_run_entropy_means[held_out], delta)
        above_idx = np.where(ph_series > lam)[0]
        alpha_ews = alphas[above_idx[0]] if len(above_idx) > 0 else None
        onsets.append({"alpha_ews": alpha_ews, "threshold": lam})
    return onsets


def run_experiment_b(loaded):
    """
    Page-Hinkley on the already-saved mean-entropy sequence, calibrated to
    the same 5% nominal LORO operating point as every other detector, added
    as a third candidate row alongside the existing dual-family/single-family
    comparison. delta=0.05 is a single fixed choice (not swept), matching
    the plan's own "cheapest 1-day version" scope; delta-sensitivity is
    explicitly out of scope here.
    """
    DELTA = 0.05
    results = {}
    for exp_name, exp_data in loaded.items():
        is_auc = exp_data["is_auc"]
        runs = exp_data["runs"]
        flat = [r for a in ARCHS for r in runs[a]]
        alphas = np.asarray(flat[0]["alphas"])
        full_len = len(alphas)
        n_base = max(4, int(full_len * 0.4))

        confirmed_failures = []
        for i, r in enumerate(flat):
            perf = perf_curve(r, is_auc)
            fail = detect_persistent_failure(alphas, perf, TAU, FAILURE_PERSIST)
            if fail["sustained"]:
                confirmed_failures.append((i, fail["idx"]))
        n_confirmed = len(confirmed_failures)

        entropy_means = [np.asarray(r["entropy_matrix"], dtype=float).mean(axis=1) for r in flat]
        nominal_segments = [em[:n_base] for em in entropy_means]

        ph_onsets = calibrate_and_apply_page_hinkley(entropy_means, nominal_segments,
                                                      alphas, full_len, delta=DELTA)

        warned = 0
        for run_idx, astar_idx in confirmed_failures:
            onset = ph_onsets[run_idx]["alpha_ews"]
            if onset is not None:
                onset_idx = int(np.argmin(np.abs(alphas - onset)))
                if onset_idx < astar_idx:
                    warned += 1
        coverage = warned / n_confirmed if n_confirmed else float("nan")

        # FAR: fresh block-resampled surrogates, fully independent of calibration draws
        rng = np.random.RandomState(3)
        n_draws = 1000
        fires = 0
        lam_median = np.median([o["threshold"] for o in ph_onsets])
        for _ in range(n_draws):
            block = max(2, full_len // 4)
            out = []
            while len(out) < full_len:
                seg = nominal_segments[rng.randint(len(nominal_segments))]
                if len(seg) <= block:
                    out.extend(seg)
                else:
                    st = rng.randint(0, len(seg) - block)
                    out.extend(seg[st:st + block])
            surrogate = np.array(out[:full_len])
            ph = page_hinkley_statistic(surrogate, DELTA)
            if np.nanmax(ph) > lam_median:
                fires += 1
        far = fires / n_draws

        results[exp_name] = {
            "detector": "Page-Hinkley (delta=0.05)",
            "coverage": round(coverage, 3),
            "false_alarm_rate": round(far, 3),
            "n_confirmed_failures": n_confirmed,
            "calibrated_threshold_median": round(float(lam_median), 4),
        }
    return results


# ------------------------------------------------------------------
# 5. Main orchestration
# ------------------------------------------------------------------
def print_experiment_a_table(results_a):
    print("=" * 78)
    print("EXPERIMENT A -- Hyperparameter Sensitivity Grid")
    print("=" * 78)
    print("Reported configuration: window=5, sigma=2.0, persistence=2")
    print()
    for exp_name, rows in results_a.items():
        print(f"--- {exp_name} ---")
        header = f"{'factor':<12}{'value':<8}{'cov_single':<12}{'cov_dual':<10}{'far_single':<12}{'far_dual':<10}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['factor_varied']:<12}{row['value']:<8}{row['coverage_single']:<12}"
                  f"{row['coverage_dual']:<10}{row['far_single']:<12}{row['far_dual']:<10}")
        print()


def print_experiment_b_table(results_b):
    print("=" * 78)
    print("EXPERIMENT B -- Page-Hinkley Adjacent-Baseline Comparison")
    print("=" * 78)
    header = f"{'experiment':<26}{'coverage':<12}{'false_alarm_rate':<18}{'n_failures':<10}"
    print(header)
    print("-" * len(header))
    for exp_name, r in results_b.items():
        print(f"{exp_name:<26}{r['coverage']:<12}{r['false_alarm_rate']:<18}{r['n_confirmed_failures']:<10}")
    print()


def main():
    print("Discovering npz files under /kaggle/input/ ...")
    all_npz = discover_files()
    print(f"Found {len(all_npz)} .npz files total.\n")

    print("Running strict verification gate before any analysis...")
    loaded = verify_and_load(all_npz)  # exits with diagnostics if anything required is missing

    print("Running Experiment A (hyperparameter sensitivity grid)...")
    results_a = run_experiment_a(loaded)
    print("Done.\n")

    print("Running Experiment B (Page-Hinkley baseline)...")
    results_b = run_experiment_b(loaded)
    print("Done.\n")

    print_experiment_a_table(results_a)
    print_experiment_b_table(results_b)

    out = {"experiment_a": results_a, "experiment_b": results_b}
    with open("/kaggle/working/strengthening_experiments_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(">>> saved /kaggle/working/strengthening_experiments_results.json <<<")


if __name__ == "__main__":
    main()



