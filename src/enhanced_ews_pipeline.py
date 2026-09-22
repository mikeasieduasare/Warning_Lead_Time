# =================================================================
# ENHANCED EWS / WARNING LEAD TIME PIPELINE
# Fixes applied vs. original implementation:
#   1. alpha_EWS: sustained-onset detection instead of raw argmax
#   2. alpha*:    persistence-checked collapse instead of first-dip
#   3. Boltzmann fit: bounded parameters + R^2 / convergence diagnostics
#   4. Temporal autocorrelation: FIXED PANEL design so "lag" is real
#      (alpha becomes the genuine time/driving axis, per CSD theory)
#   5. Composite vs. single-indicator metric: Kendall-tau trend test
#      with surrogate significance, AC1 primary + convergent-evidence check
#   6. Bootstrap 95% CIs for alpha*, sharpness, lead time
#   7. Baseline OOD comparison: MSP + Energy + Mahalanobis distance, same
#      onset-detection rule applied fairly, plus AUROC separation check
#   8. Hypothesis testing: permutation test on bootstrap lead-time samples,
#      wired into main() for BOTH cross-architecture and proposed-vs-baseline
#      comparisons (not just available as an unused library function)
# =================================================================

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, skew as skew_fn

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # The statistical functions (Sections 1-6, 9) work without torch.
    # Only the model-inference helpers (Sections 5's data-collection half,
    # 7, 8, 10) require it and will raise a clear error if called.


# =================================================================
# 1. BOUNDED BOLTZMANN FIT WITH DIAGNOSTICS
# =================================================================
def boltzmann_function(x, L, x0, k, b):
    """Sigmoid function to fit accuracy collapse."""
    return L / (1 + np.exp(k * (x - x0))) + b


def fit_boltzmann_with_diagnostics(alphas, accuracy):
    """
    Bounded Levenberg-Marquardt fit with explicit convergence + goodness-
    of-fit reporting. Unbounded fits on a curve that doesn't contain a full
    sigmoid (e.g. collapse only at the very last step) can return extreme,
    meaningless k values (as seen with ResNet18's k = -154.06 in an
    earlier run).

    PERFORMANCE/CORRECTNESS FIX: on a FLAT curve (near-zero variance --
    e.g. a bootstrap resample of a robust model that never degrades, or
    any accuracy trajectory with no real transition), the bounded
    trust-region optimizer has no informative gradient to follow and can
    burn through thousands of iterations against the bounds before
    reporting non-convergence -- this was observed to take 15+ seconds
    PER CALL on trivial 20-point data, which is fatal inside a bootstrap
    loop calling this thousands of times. Since a flat curve has no
    sigmoid to fit in the first place, we skip curve_fit entirely in that
    case and return an explicitly-flagged degenerate result -- correct
    AND fast.
    """
    alphas = np.asarray(alphas, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)

    rng = accuracy.max() - accuracy.min()
    # Two independent no-signal checks, EITHER of which skips curve_fit
    # entirely: (1) near-flat range, or (2) no real monotonic trend (weak
    # correlation with alpha) even if the range happens to be wide due to
    # noise. Both cases have no genuine sigmoid to find, and both were
    # observed to make the bounded optimizer burn through its evaluation
    # budget for nothing -- catching them here is what keeps the bootstrap
    # loop fast, since a single per-call retry budget large enough to
    # reliably converge on GENUINE shallow curves (see below) is still far
    # too expensive to pay on every no-signal resample.
    if rng < 0.03:
        return {
            "L": 0.0, "x0": float(np.median(alphas)), "k": 0.0, "b": float(accuracy.mean()),
            "r_squared": 0.0, "param_se": [float("nan")] * 4,
            "converged": False, "flagged_low_quality": True,
            "flat_curve_skip": True,
        }
    if np.std(accuracy) > 0:
        corr = np.corrcoef(alphas, accuracy)[0, 1]
        if abs(corr) < 0.5:
            return {
                "L": 0.0, "x0": float(np.median(alphas)), "k": 0.0, "b": float(accuracy.mean()),
                "r_squared": 0.0, "param_se": [float("nan")] * 4,
                "converged": False, "flagged_low_quality": True,
                "flat_curve_skip": True,
            }

    p0 = [max(rng, 1e-3), np.median(alphas), 10.0, accuracy.min()]
    bounds = ([0.0, alphas.min(), -200.0, 0.0],
              [1.5, alphas.max(), 200.0, 1.0])

    def _try_fit(maxfev):
        popt, pcov = curve_fit(boltzmann_function, alphas, accuracy,
                                p0=p0, bounds=bounds, maxfev=maxfev)
        preds = boltzmann_function(alphas, *popt)
        ss_res = np.sum((accuracy - preds) ** 2)
        ss_tot = np.sum((accuracy - accuracy.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [float("nan")] * 4
        return {
            "L": popt[0], "x0": popt[1], "k": popt[2], "b": popt[3],
            "r_squared": r2, "param_se": perr.tolist(),
            "converged": True, "flagged_low_quality": r2 < 0.8 if not np.isnan(r2) else True,
            "flat_curve_skip": False,
        }

    # Two-stage retry, now safe to use a slightly larger fallback budget
    # since the correlation gate above already filters out the common
    # "retry is pointless" cases (flat OR noisy-no-trend). Genuine but
    # shallow monotonic declines -- the case that motivated this retry in
    # the first place -- pass the correlation gate and get a real second
    # attempt.
    try:
        return _try_fit(maxfev=20)
    except Exception:
        pass
    try:
        return _try_fit(maxfev=60)
    except Exception as e:
        return {
            "L": float("nan"), "x0": float("nan"), "k": float("nan"), "b": float("nan"),
            "r_squared": float("nan"), "param_se": [float("nan")] * 4,
            "converged": False, "flagged_low_quality": True, "flat_curve_skip": False, "error": str(e),
        }


# =================================================================
# 2. SUSTAINED-ONSET DETECTION FOR alpha_EWS  (replaces raw argmax)
# =================================================================
def calibrate_population_threshold(signal_pool, persistence=2, nominal_fpr=0.05,
                                    baseline_frac=0.4, n_surrogates=400, seed=0,
                                    serial_dependence=False):
    """
    Estimate ONE detection threshold for a detector, pooled across all runs of
    an experiment (spec v3, Section 5; implements the PI's concern #5
    recommendation for "a larger nominal reference sample"), with the
    calibration null LENGTH-MATCHED to the trajectory the threshold is applied to.

    The critical correctness point (learned from validated failures): a
    persistence-k crossing rule fires more often on a longer series, so the null
    used to set the operating point MUST be the same length as the real
    trajectory. Calibrating on short nominal SEGMENTS and applying to the full
    trajectory inflates the realised false-alarm rate. We therefore build
    FULL-LENGTH no-onset surrogates and measure the firing rate on those.

    Null construction:
      - Pool each run's nominal-region values (first `baseline_frac` of valid
        points; the pre-transition regime, which contains no onset).
      - Generate `n_surrogates` full-length surrogate trajectories by resampling
        the pooled nominal values to the trajectory length.
      - `serial_dependence=False` (default): iid resampling. VALID for
        representation- and uncertainty-based detectors (Mahalanobis, Energy,
        MC-Dropout), whose scores are serially near-independent across alpha
        (verified). This is the family that carries the primary Warning Lead Time.
      - `serial_dependence=True`: block resampling that preserves short-range
        autocorrelation, for statistics like AC1 that are themselves
        autocorrelations. NOTE: AC1 is a weak onset detector at this resolution
        even when calibrated (verified: informedness ~0 even on textbook CSD),
        so its achieved false-alarm rate is reported honestly rather than
        assumed to meet the nominal target.

    Returns the smallest threshold whose full-length surrogate firing rate is
    <= nominal_fpr, or the max observed value if unreachable.
    """
    nominal_segments = []
    full_len = 0
    for sig in signal_pool:
        v = np.asarray(sig, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 6:
            continue
        full_len = max(full_len, len(v))
        n_base = max(4, int(len(v) * baseline_frac))
        nominal_segments.append(v[:n_base])
    if not nominal_segments or full_len < 6:
        return None

    pooled = np.concatenate(nominal_segments)
    rng = np.random.RandomState(seed)

    def make_surrogate():
        if not serial_dependence:
            return rng.choice(pooled, size=full_len, replace=True)
        # block resample: draw contiguous blocks from nominal segments to keep
        # short-range autocorrelation, tile to full length
        block = max(2, full_len // 4)
        out = []
        while len(out) < full_len:
            seg = nominal_segments[rng.randint(len(nominal_segments))]
            if len(seg) <= block:
                out.extend(seg)
            else:
                st = rng.randint(0, len(seg) - block)
                out.extend(seg[st:st + block])
        return np.array(out[:full_len])

    surrogates = [make_surrogate() for _ in range(n_surrogates)]

    def rule_fires(series, thr):
        run = 0
        for a in (series > thr):
            if a:
                run += 1
                if run >= persistence:
                    return True
            else:
                run = 0
        return False

    qs = np.percentile(pooled, np.arange(50, 100.01, 0.5))
    candidates = np.append(qs, pooled.max() + 1e-6)
    for thr in candidates:
        rate = np.mean([rule_fires(sg, thr) for sg in surrogates])
        if rate <= nominal_fpr:
            return float(thr)
    return float(candidates[-1])


def apply_onset_threshold(alphas, signal, threshold, persistence=2, nominal_fpr=0.05):
    """
    Apply a precomputed (population-calibrated) threshold to one run's signal.
    Reports the first alpha at which the crossing-plus-persistence rule triggers.
    No fallback path: an undetected onset is a genuine, reportable negative,
    never a peak chosen using the full trajectory.

    detection_method in:
      'calibrated_threshold_crossing' | 'no_onset_at_nominal_fpr'
      'threshold_non_estimable' (threshold is None) | 'insufficient_data'
    """
    alphas = np.asarray(alphas)
    signal = np.asarray(signal, dtype=float)
    valid_mask = ~np.isnan(signal)
    if valid_mask.sum() < 4:
        return {"alpha_ews": None, "onset_idx": None,
                "detection_method": "insufficient_data"}
    if threshold is None:
        return {"alpha_ews": None, "onset_idx": None,
                "detection_method": "threshold_non_estimable"}

    above = np.where(valid_mask, signal > threshold, False)
    onset_idx = None
    for i in range(len(signal) - persistence + 1):
        if np.all(above[i:i + persistence]):
            onset_idx = i
            break
    if onset_idx is None:
        return {"alpha_ews": None, "onset_idx": None, "threshold_used": threshold,
                "nominal_fpr": nominal_fpr, "detection_method": "no_onset_at_nominal_fpr"}
    return {"alpha_ews": alphas[onset_idx], "onset_idx": onset_idx,
            "threshold_used": threshold, "nominal_fpr": nominal_fpr,
            "detection_method": "calibrated_threshold_crossing"}


def _circular_shift_threshold(signal, persistence, nominal_fpr, baseline_frac=0.4,
                              n_surrogates=300, seed=0):
    """
    Calibrate a detection threshold so the FULL rule (a crossing that persists
    for `persistence` points) fires on at most `nominal_fpr` of FULL-LENGTH
    no-onset surrogates.

    Null construction (the key design point). Two naive choices both fail, as
    verified against ground truth:
      * circular-shifting the whole signal keeps a genuine onset's elevated
        block in the surrogate, inflating the threshold so real onsets are
        never detected;
      * calibrating only on the short baseline segment ignores the natural
        late-trajectory variance of the statistic, so pure-null noise past the
        baseline trips the rule (false positives).
    The correct null is a series that is (a) the SAME LENGTH as the real
    trajectory, so late variance is represented, and (b) free of any onset by
    construction. We build it by bootstrapping the baseline residuals to full
    length: sample, with replacement, from the mean-centred baseline values and
    add back the baseline mean. Each such surrogate is a no-transition version
    of the signal at the signal's own noise level. The threshold is the smallest
    value whose crossing-plus-persistence firing rate over these surrogates is
    <= nominal_fpr.

    Returns None if the target rate is unachievable (caller records the onset as
    non-estimable rather than fabricating one).
    """
    s = np.asarray(signal, dtype=float)
    valid = s[~np.isnan(s)]
    n_full = len(valid)
    if n_full < 6:
        return None

    n_base = max(4, int(n_full * baseline_frac))
    baseline = valid[:n_base]
    base_mean = np.mean(baseline)
    base_resid = baseline - base_mean
    if np.allclose(base_resid, 0):
        return None  # degenerate flat baseline: no dispersion to calibrate against

    rng = np.random.RandomState(seed)
    surrogates = [base_mean + rng.choice(base_resid, size=n_full, replace=True)
                  for _ in range(n_surrogates)]

    def rule_fires(series, thr):
        run = 0
        for a in (series > thr):
            if a:
                run += 1
                if run >= persistence:
                    return True
            else:
                run = 0
        return False

    lo, hi = np.percentile(baseline, 50), np.nanmax(valid)
    candidates = np.linspace(lo, hi, 80)
    for thr in candidates:
        rate = np.mean([rule_fires(sg, thr) for sg in surrogates])
        if rate <= nominal_fpr:
            return float(thr)
    return None


def detect_onset_calibrated(alphas, signal, nominal_fpr=0.05, persistence=2,
                            n_surrogates=200, seed=0):
    """
    Procedure-calibrated onset detector (spec v3, Section 5).

    Sets the detection threshold so the full crossing-plus-persistence rule
    fires on <= nominal_fpr of circular-shift surrogates of THIS signal, then
    reports the first alpha at which the real signal triggers that rule.

    Advantages over the mean+k*sigma rule this replaces:
      - Attainable by construction: the threshold is a surrogate percentile,
        never a value outside the statistic's own observed range (the old rule
        produced thresholds above AC1's [-1, 1] bound in ~18% of cases).
      - Every detector gets the SAME stated operating point (a common nominal
        false-alarm rate), removing the confound that the Safety Shell only
        wins because its threshold happens to be stricter.

    Returns detection_method in:
      'calibrated_threshold_crossing' - genuine onset found
      'no_onset_at_nominal_fpr'       - signal never triggers the calibrated
                                        rule (a real, reportable negative:
                                        NOT a fallback, NOT look-ahead)
      'threshold_non_estimable'       - surrogate calibration could not reach
                                        the target FPR (too few points, etc.)
      'insufficient_data'
    NO fallback path exists: an undetected onset is reported as a genuine
    negative, never as a peak selected using the full trajectory.
    """
    alphas = np.asarray(alphas)
    signal = np.asarray(signal, dtype=float)
    valid_mask = ~np.isnan(signal)
    if valid_mask.sum() < 4:
        return {"alpha_ews": None, "onset_idx": None,
                "detection_method": "insufficient_data"}

    thr = _circular_shift_threshold(signal, persistence, nominal_fpr,
                                    n_surrogates=n_surrogates, seed=seed)
    if thr is None:
        return {"alpha_ews": None, "onset_idx": None,
                "detection_method": "threshold_non_estimable"}

    above = np.where(valid_mask, signal > thr, False)
    onset_idx = None
    for i in range(len(signal) - persistence + 1):
        if np.all(above[i:i + persistence]):
            onset_idx = i
            break

    if onset_idx is None:
        return {"alpha_ews": None, "onset_idx": None,
                "threshold_used": thr, "nominal_fpr": nominal_fpr,
                "detection_method": "no_onset_at_nominal_fpr"}

    return {"alpha_ews": alphas[onset_idx], "onset_idx": onset_idx,
            "threshold_used": thr, "nominal_fpr": nominal_fpr,
            "detection_method": "calibrated_threshold_crossing"}


def detect_sustained_onset(alphas, signal, baseline_frac=0.2, k_sigma=2.0, persistence=2):
    """
    First alpha at which `signal` exceeds baseline_mean + k_sigma*baseline_std
    and STAYS above that line for >= `persistence` consecutive points.
    If no sustained crossing exists, falls back to argmax but LABELS it as
    such, rather than silently returning a possibly-spurious peak.
    """
    alphas = np.asarray(alphas)
    signal = np.asarray(signal, dtype=float)
    valid = ~np.isnan(signal)
    if valid.sum() < 4:
        return {"alpha_ews": alphas[np.nanargmax(signal)] if valid.any() else alphas[-1],
                "onset_idx": None, "detection_method": "insufficient_data"}

    n_baseline = max(3, int(valid.sum() * baseline_frac))
    baseline_vals = signal[valid][:n_baseline]
    mu, sigma = np.mean(baseline_vals), np.std(baseline_vals)
    thresh = mu + k_sigma * sigma

    above = np.where(valid, signal > thresh, False)
    onset_idx = None
    for i in range(len(signal) - persistence + 1):
        if np.all(above[i:i + persistence]):
            onset_idx = i
            break

    if onset_idx is None:
        onset_idx = int(np.nanargmax(signal))
        method = "fallback_argmax_no_sustained_crossing"
    else:
        method = "sustained_threshold_crossing"

    return {
        "alpha_ews": alphas[onset_idx],
        "onset_idx": onset_idx,
        "threshold_used": thresh,
        "baseline_mean": mu,
        "baseline_std": sigma,
        "detection_method": method,
    }


# =================================================================
# 3. PERSISTENCE-CHECKED CRITICAL FAILURE POINT (alpha*)
# =================================================================
def detect_persistent_failure(alphas, accuracy, threshold=0.70, persistence=2):
    """
    alpha* = first index after which accuracy stays below `threshold` for
    the REST of the trajectory (true collapse), not just a transient dip
    caused by sampling noise in a freshly-drawn test subset.

    The persistence window must genuinely exist: i is only considered if at
    least `persistence` grid points remain from i to the end of the
    trajectory (n - i >= persistence). A single subthreshold point at the
    final grid level, with no subsequent level available to confirm it,
    does NOT satisfy a persistence-2 rule and must remain right-censored.

    An earlier version of this function sliced `below[i:min(i+persistence, n)]`,
    which silently truncates at the array boundary rather than requiring the
    full window to exist -- at the last index this collapses a persistence-2
    requirement to persistence-1, so a lone terminal subthreshold value was
    wrongly accepted as a confirmed failure. Verified against real data: a
    trajectory below threshold ONLY at its final grid point, with every
    preceding point above threshold, was returned as sustained=True under the
    old code and is correctly right-censored under this version.
    """
    alphas = np.asarray(alphas)
    accuracy = np.asarray(accuracy, dtype=float)
    below = accuracy < threshold
    n = len(accuracy)

    for i in range(n - persistence + 1):  # require the full window to exist
        if below[i] and np.all(below[i:i + persistence]) and np.all(below[i:]):
            return {"alpha_star": alphas[i], "idx": i, "sustained": True}

    return {"alpha_star": alphas[-1], "idx": n - 1, "sustained": False}


# =================================================================
# 4. CORRECTED calculate_transition_metrics
# =================================================================
def calculate_transition_metrics(alphas, accuracy, ews_signal, threshold=0.70,
                                  baseline_frac=0.2, k_sigma=2.0, persistence=2):
    fail = detect_persistent_failure(alphas, accuracy, threshold, persistence)
    onset = detect_sustained_onset(alphas, ews_signal, baseline_frac, k_sigma, persistence)
    fit = fit_boltzmann_with_diagnostics(alphas, accuracy)

    alpha_star = fail["alpha_star"]
    alpha_ews = onset["alpha_ews"]
    lead_time = (alpha_star - alpha_ews) if alpha_star > alpha_ews else 0.0

    return {
        "alpha_star": alpha_star,
        "alpha_star_sustained": fail["sustained"],
        "alpha_ews": alpha_ews,
        "onset_method": onset["detection_method"],
        "sharpness": fit["k"],
        "fit_r_squared": fit["r_squared"],
        "fit_converged": fit["converged"],
        "fit_flagged_low_quality": fit["flagged_low_quality"],
        "lead_time": lead_time,
    }


# =================================================================
# 5. FIXING THE TEMPORAL AUTOCORRELATION PROBLEM
# =================================================================
# CONCEPTUAL FIX
# --------------
# Critical Slowing Down requires a genuine TIME axis: the system must be
# observed repeatedly AS IT IS PUSHED toward the transition, so lag-1
# correlation reflects how slowly it relaxes between consecutive
# observations of the driving parameter.
#
# The original code drew a FRESH random subsample of test images at every
# alpha (`np.random.choice(..., replace=False)`), and computed AC1 across
# entropies of DIFFERENT, randomly-ordered images within one batch. There
# is no meaningful "lag" between unrelated images and no temporal link
# between alpha steps.
#
# FIX: track a FIXED panel of images (same indices, same order) across ALL
# alpha levels. Alpha becomes the genuine driving/time axis. The scalar
# system-state series is the panel-mean entropy at each alpha step:
#     E(alpha_t) = mean_i [ H_i(alpha_t) ]
# Sliding-window AC1 / variance / skewness are then computed over THIS
# single alpha-indexed series -- the standard "generic early-warning-signal"
# method (Dakos et al., 2012) -- instead of cross-sectional stats within
# one shuffled batch.

def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "This function requires torch (model inference). "
            "Install torch, or use the statistical functions in Sections 1-6/9 "
            "directly on pre-exported entropy/accuracy arrays."
        )


def build_fixed_panel_indices(test_ds, panel_size=400, seed=42):
    """Select ONE fixed, seeded panel of test indices reused at every alpha."""
    rng = np.random.RandomState(seed)
    return rng.choice(len(test_ds), panel_size, replace=False)


def compute_panel_state_series(model, test_ds, panel_indices, alphas,
                                apply_shift_fn, device, batch_size=32):
    """
    Runs the SAME fixed panel through the model at every alpha, returning:
      - a single alpha-indexed time series of mean entropy (system state)
      - the full per-image entropy matrix (n_alphas x panel_size) for reuse
        in the bootstrap (no re-inference needed for resampling)
      - the matching per-image correctness matrix, resampled jointly with
        entropy so bootstrap accuracy stays consistent with bootstrap entropy
    """
    _require_torch()
    panel_subset = Subset(test_ds, panel_indices)
    loader = DataLoader(panel_subset, batch_size=batch_size, shuffle=False)

    entropy_matrix, correct_matrix = [], []
    model.eval()
    for alpha in alphas:
        step_entropies, step_correct = [], []
        for inputs, labels in loader:
            inputs = apply_shift_fn(inputs.to(device), alpha)
            labels = labels.to(device).long().view(-1)
            with torch.no_grad():
                probs = torch.softmax(model(inputs), dim=1)
                preds = probs.argmax(dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            step_entropies.extend(ent.cpu().numpy())
            step_correct.extend((preds == labels).float().cpu().numpy())
        entropy_matrix.append(step_entropies)
        correct_matrix.append(step_correct)

    entropy_matrix = np.array(entropy_matrix)   # (n_alphas, panel_size)
    correct_matrix = np.array(correct_matrix)   # (n_alphas, panel_size)
    return {
        "alphas": np.asarray(alphas),
        "state_series": entropy_matrix.mean(axis=1),
        "accuracy": correct_matrix.mean(axis=1),
        "entropy_matrix": entropy_matrix,
        "correct_matrix": correct_matrix,
    }


def detrend_series(series, sigma=2.0):
    """
    Gaussian-kernel detrending: subtracts a smoothed trend from the raw
    series before computing EWS indicators, following standard practice in
    the EWS literature (e.g. epidemiological incidence-data EWS studies
    explicitly detrend before computing variance/AC1/skewness; the climate
    phase-transition EWS literature computes temporal statistics on
    residuals after subtracting a Gaussian-filter moving average).

    Without this, autocorrelation on a smoothly trending raw series is
    inflated by the trend itself -- neighboring points on any smooth curve
    look correlated whether or not the system is actually losing resilience.
    This is very likely why raw (undetrended) AC1 saturated near 1.0 across
    almost the entire alpha range in the uncorrected run, rather than
    showing the classic "low, then climbing near collapse" CSD signature.
    """
    from scipy.ndimage import gaussian_filter1d
    series = np.asarray(series, dtype=float)
    trend = gaussian_filter1d(series, sigma=sigma, mode="nearest")
    residual = series - trend
    return residual, trend


def windowed_ews_indicators(state_series, window=5, detrend=True, detrend_sigma=2.0):
    """
    Sliding-window AC1, variance, skewness over the alpha-indexed state
    series (the genuine time axis). First `window-1` points are NaN
    (insufficient history) rather than silently dropped/zeroed.

    `detrend=True` (new default): indicators are computed on the RESIDUAL
    after subtracting a Gaussian-filter trend, not the raw series -- see
    detrend_series() docstring for why this matters. Set detrend=False to
    reproduce the old (raw-series) behavior for comparison.
    """
    state_series = np.asarray(state_series, dtype=float)
    if detrend:
        residual, trend = detrend_series(state_series, sigma=detrend_sigma)
        series_to_use = residual
    else:
        residual, trend = None, None
        series_to_use = state_series

    n = len(series_to_use)
    ac1 = np.full(n, np.nan)
    var = np.full(n, np.nan)
    skw = np.full(n, np.nan)

    for i in range(window - 1, n):
        w = series_to_use[i - window + 1: i + 1]
        if len(w) >= 3 and np.std(w) > 0:
            ac1[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
            var[i] = np.var(w)
            skw[i] = skew_fn(w)
    return {"ac1": ac1, "variance": var, "skewness": skw, "trend": trend, "residual": residual}


# =================================================================
# 6. COMPOSITE VS. SINGLE-INDICATOR HEADLINE METRIC
# =================================================================
# RECOMMENDATION: report BOTH, with AC1 as the pre-registered PRIMARY
# indicator (it is the theoretically canonical CSD statistic, directly
# tied to the return-rate/relaxation-time argument in Scheffer et al. 2009
# and Dakos et al. 2012) and the composite as a ROBUSTNESS / SENSITIVITY
# check, not the reverse.
#
# Why not composite-only: variance and skewness correlate with AC1 but
# their grounding as universal CSD precursors is weaker and more
# system-dependent. A black-box composite as the headline risks reading
# as metric-shopping to a reviewer who knows this literature.
#
# Why not AC1-only, unqualified: a single statistic is fragile to noise
# (this is exactly what produced the boundary-snapping in Table 1). The
# fix is (a) trend-significance rather than a raw peak [Section 2 above],
# and (b) reporting whether independent indicators AGREE, as an explicit
# robustness statement, without collapsing them into one opaque number.

def kendall_trend_test(alphas, signal, n_surrogates=1000, seed=0):
    """
    Kendall's tau trend test with SURROGATE significance (random shuffles
    of the signal destroy temporal order while preserving its distribution,
    avoiding inflated significance from Savitzky-Golay-smoothed series).
    """
    rng = np.random.RandomState(seed)
    valid = ~np.isnan(signal)
    a, s = np.asarray(alphas)[valid], np.asarray(signal)[valid]
    if len(s) < 4:
        return {"tau": float("nan"), "p_surrogate": float("nan"), "significant": False}

    tau_obs, _ = kendalltau(a, s)
    surrogate_taus = np.empty(n_surrogates)
    for b in range(n_surrogates):
        surrogate_taus[b], _ = kendalltau(a, rng.permutation(s))

    p_surrogate = float(np.mean(np.abs(surrogate_taus) >= np.abs(tau_obs)))
    return {"tau": float(tau_obs), "p_surrogate": p_surrogate, "significant": p_surrogate < 0.05}


def composite_ews_agreement(alphas, indicators_dict, onset_idx):
    """
    Evaluates trend significance for each indicator over the PRE-ONSET
    window (alphas[:onset_idx+1]) and reports how many show a significant
    positive trend ("convergent evidence of CSD" if >= 2 of 3 agree;
    otherwise explicitly flagged as weak/inconclusive rather than hidden).
    """
    if onset_idx is None:
        onset_idx = len(alphas) - 1
    pre_alphas = alphas[:onset_idx + 1]

    results, n_sig = {}, 0
    for name, arr in indicators_dict.items():
        test = kendall_trend_test(pre_alphas, arr[:onset_idx + 1])
        results[name] = test
        if test["significant"] and test["tau"] > 0:
            n_sig += 1
    results["n_agree"] = n_sig
    results["convergent_csd_evidence"] = n_sig >= 2
    return results


# =================================================================
# 7. BOOTSTRAP CONFIDENCE INTERVALS
# =================================================================
def bootstrap_transition_metrics(model, test_ds, apply_shift_fn, device, alphas,
                                  panel_size=400, n_boot=500, threshold=0.70,
                                  seed=0, batch_size=32):
    """
    Resamples the FIXED PANEL with replacement (image-level block bootstrap)
    B times. Entropy + correctness are cached from ONE forward pass over the
    full panel, so bootstrap replicates are recomputed from the cached
    matrices without re-running inference. Returns percentile 95% CIs for
    alpha*, sharpness, and lead time, plus the full-panel point estimate.
    """
    rng = np.random.RandomState(seed)
    panel = build_fixed_panel_indices(test_ds, panel_size, seed=seed)
    point = compute_panel_state_series(model, test_ds, panel, alphas, apply_shift_fn, device, batch_size)

    ews = windowed_ews_indicators(point["state_series"])
    point_metrics = calculate_transition_metrics(point["alphas"], point["accuracy"], ews["ac1"], threshold)
    composite = composite_ews_agreement(
        point["alphas"], ews,
        detect_sustained_onset(point["alphas"], ews["ac1"])["onset_idx"]
    )

    entropy_matrix, correct_matrix = point["entropy_matrix"], point["correct_matrix"]
    boot_alpha_star, boot_sharpness, boot_lead = [], [], []

    for _ in range(n_boot):
        cols = rng.choice(panel_size, panel_size, replace=True)
        boot_state = entropy_matrix[:, cols].mean(axis=1)
        boot_acc = correct_matrix[:, cols].mean(axis=1)
        boot_ews = windowed_ews_indicators(boot_state)
        m = calculate_transition_metrics(point["alphas"], boot_acc, boot_ews["ac1"], threshold)
        boot_alpha_star.append(m["alpha_star"])
        boot_sharpness.append(m["sharpness"])
        boot_lead.append(m["lead_time"])

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2

    return {
        "point_estimate": point_metrics,
        "composite_agreement": composite,
        "alpha_star_ci": ci(boot_alpha_star),
        "sharpness_ci": ci(boot_sharpness),
        "lead_time_ci": ci(boot_lead),
        "lead_time_samples": boot_lead,   # kept for cross-model significance testing (Section 9)
        "n_boot": n_boot,
    }


# =================================================================
# 8. BASELINE OOD / DRIFT DETECTORS FOR COMPARISON
# =================================================================
def compute_baseline_scores(model, inputs):
    """Per-sample MSP and Energy scores for one batch (no grad)."""
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        msp = probs.max(dim=1)[0]                  # higher = more in-distribution
        energy = -torch.logsumexp(logits, dim=1)   # higher = more OOD
    return msp.cpu().numpy(), energy.cpu().numpy()


def mc_dropout_predict(model, inputs, n_samples=10):
    """
    Monte Carlo Dropout uncertainty estimate (Gal & Ghahramani, 2016), cited
    in this paper's own Related Work but never previously implemented here.

    Runs `n_samples` stochastic forward passes with ONLY Dropout submodules
    switched to train() mode (BatchNorm and everything else stays in eval()
    mode, so running statistics aren't corrupted -- the standard MC-Dropout
    trick). Returns per-sample mean softmax probability and predictive
    variance (mean of per-class variance across the MC samples) as the
    uncertainty score; higher variance = more uncertain = more likely OOD.

    ARCHITECTURE NOTE: torchvision's ResNet18 and DenseNet121 do NOT include
    dropout layers by default (unlike ViT-B16, which does). For MC Dropout
    to be usable and COMPARABLE across all three architectures, a dropout
    layer must be explicitly inserted before each architecture's final
    classification layer at model-construction time (see
    initialize_single_model in the main pipeline) -- this function assumes
    that has already been done and simply activates whatever nn.Dropout
    modules exist in the given model.

    COST NOTE: this requires n_samples forward passes per batch instead of
    1, specifically for this baseline's data collection -- budget for a
    meaningful slowdown of the trajectory-collection step where this is
    called, proportional to n_samples.
    """
    def _enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    was_training = model.training
    model.eval()          # BatchNorm etc. stay in eval mode
    model.apply(_enable_dropout)  # ...but Dropout submodules are flipped back to train mode

    probs_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            probs = torch.softmax(model(inputs), dim=1)
            probs_samples.append(probs.unsqueeze(0))

    if was_training:
        model.train()
    else:
        model.eval()

    probs_stack = torch.cat(probs_samples, dim=0)          # (n_samples, B, C)
    mean_probs = probs_stack.mean(dim=0)                    # (B, C)
    var_probs = probs_stack.var(dim=0).mean(dim=1)           # (B,) -- mean variance across classes
    return mean_probs.cpu().numpy(), var_probs.cpu().numpy()


def calibrate_baseline_thresholds(model, cal_loader, device, q=0.95):
    """Calibrate MSP / Energy thresholds on clean (alpha=0) calibration data."""
    msp_scores, energy_scores = [], []
    for inputs, _ in cal_loader:
        inputs = inputs.to(device)
        msp, energy = compute_baseline_scores(model, inputs)
        msp_scores.extend(msp)
        energy_scores.extend(energy)
    return {
        "msp_threshold": float(np.quantile(msp_scores, 1 - q)),
        "energy_threshold": float(np.quantile(energy_scores, q)),
    }


def baseline_onset_detection(alphas, msp_means, energy_means, mahalanobis_means=None,
                              extra_signals=None, persistence=2):
    """
    Applies the SAME sustained-crossing rule used for the proposed EWS
    signal (Section 2) to each baseline, so the comparison uses identical
    onset logic and differs only in the input signal. `mahalanobis_means`
    is optional so existing MSP/Energy-only call sites keep working.

    `extra_signals`: optional dict of {name: alpha-indexed array} for any
    additional detector (e.g. MC Dropout predictive variance) without
    hardcoding it into this function's signature. Each entry produces a
    "{name}_onset" key in the output, exactly like the built-in baselines,
    and therefore flows automatically through compare_lead_times (which
    already generically strips "_onset" -> "_lead_time" for any key).
    Higher values in `extra_signals` arrays are assumed to mean "more
    anomalous" -- invert beforehand if a given signal's convention differs
    (as MSP is inverted internally below, since higher MSP = more confident
    = LESS anomalous under nominal conditions).
    """
    msp_signal = -np.asarray(msp_means)  # invert: higher = more anomalous
    onset_msp = detect_sustained_onset(alphas, msp_signal, persistence=persistence)
    onset_energy = detect_sustained_onset(alphas, np.asarray(energy_means), persistence=persistence)
    out = {"msp_onset": onset_msp, "energy_onset": onset_energy}
    if mahalanobis_means is not None:
        out["mahalanobis_onset"] = detect_sustained_onset(
            alphas, np.asarray(mahalanobis_means), persistence=persistence
        )
    if extra_signals:
        for name, arr in extra_signals.items():
            out[f"{name}_onset"] = detect_sustained_onset(alphas, np.asarray(arr), persistence=persistence)
    return out


def compare_lead_times(alpha_star, onset_dict):
    """Delta-alpha for each baseline detector using the SAME alpha* as the proposed method."""
    out = {}
    for name, onset in onset_dict.items():
        a_ews = onset["alpha_ews"]
        out[name.replace("_onset", "_lead_time")] = max(0.0, alpha_star - a_ews) if alpha_star > a_ews else 0.0
    return out


def _auc_from_scores(labels, scores):
    """
    Panel AUROC at one shift level, from per-image predicted probabilities and
    binary labels. Used to locate the Critical Failure Point in experiments
    reported in AUROC (histopathology) rather than accuracy (radiography).
    Returns NaN when AUROC is undefined (a bootstrap resample can draw only one
    class); NaN levels are simply not counted as threshold crossings.
    """
    y = np.asarray(labels).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if roc_auc_score is None or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def detection_auroc_vs_alpha(id_scores, shifted_scores):
    """AUROC separating in-distribution (alpha=0) scores from a shifted-alpha level's scores."""
    if roc_auc_score is None:
        return float("nan")
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(shifted_scores))])
    s = np.concatenate([id_scores, shifted_scores])
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


# =================================================================
# 8b. MAHALANOBIS DISTANCE BASELINE  (Lee et al., 2018)
# =================================================================
# Named explicitly in this paper's own Related Work section but never
# implemented until now. Requires penultimate-layer features, extracted
# via a forward hook on each architecture's final classifier layer.
def register_feature_hook(model, arch_name):
    """
    Registers a forward hook that captures the INPUT to the final
    classifier layer (i.e. the penultimate feature vector) for the three
    architectures used in this study. Returns (handle, holder); read
    holder["features"] after each forward pass, and call handle.remove()
    when done.
    """
    _require_torch()
    holder = {}

    def hook(module, inp, out):
        holder["features"] = inp[0].detach()

    if arch_name == "ResNet18":
        handle = model.fc.register_forward_hook(hook)
    elif arch_name == "DenseNet121":
        handle = model.classifier.register_forward_hook(hook)
    elif arch_name == "ViT_B16":
        handle = model.heads.head.register_forward_hook(hook)
    else:
        raise ValueError(
            f"No feature-hook rule defined for architecture '{arch_name}'. "
            "Add a case here naming the final classifier layer to hook."
        )
    return handle, holder


def calibrate_mahalanobis(features_cal, labels_cal, num_classes):
    """
    Fits class-conditional means and a shared (tied) covariance matrix from
    calibration-set penultimate features, following Lee et al. (2018).
    """
    features_cal = np.asarray(features_cal, dtype=float)
    labels_cal = np.asarray(labels_cal).astype(int)
    d = features_cal.shape[1]

    class_means, centered = [], []
    for c in range(num_classes):
        mask = labels_cal == c
        if mask.sum() == 0:
            class_means.append(np.zeros(d))
            continue
        mu_c = features_cal[mask].mean(axis=0)
        class_means.append(mu_c)
        centered.append(features_cal[mask] - mu_c)

    centered_all = np.concatenate(centered, axis=0) if centered else features_cal
    cov = np.cov(centered_all, rowvar=False) + 1e-6 * np.eye(d)
    cov_inv = np.linalg.pinv(cov)
    return {"class_means": np.stack(class_means), "cov_inv": cov_inv}


def mahalanobis_scores(features, mahalanobis_params):
    """
    Per-sample Mahalanobis OOD score = distance to the NEAREST class mean
    (Lee et al., 2018). Higher = farther from every class centroid = more
    anomalous -- same sign convention as the Energy score, so it plugs
    directly into baseline_onset_detection without inversion.
    """
    means = mahalanobis_params["class_means"]
    cov_inv = mahalanobis_params["cov_inv"]
    features = np.asarray(features, dtype=float)

    dists = np.stack([
        np.einsum("ij,jk,ik->i", features - mu_c, cov_inv, features - mu_c)
        for mu_c in means
    ], axis=1)
    return dists.min(axis=1)


def detect_fixed_threshold_onset(alphas, signal, threshold, persistence=2):
    """
    Like detect_sustained_onset, but against an EXTERNALLY SPECIFIED
    threshold rather than a data-driven baseline+k_sigma. Used for
    conformal prediction-set size, which has a theoretically meaningful
    nominal value (1.0, single-label sets) rather than an empirical
    baseline that would need to be estimated from the data itself.
    """
    alphas = np.asarray(alphas)
    signal = np.asarray(signal, dtype=float)
    valid = ~np.isnan(signal)
    if valid.sum() < persistence:
        return {"onset_idx": None, "detection_method": "insufficient_data"}

    above = np.where(valid, signal > threshold, False)
    onset_idx = None
    for i in range(len(signal) - persistence + 1):
        if np.all(above[i:i + persistence]):
            onset_idx = i
            break

    method = "sustained_threshold_crossing" if onset_idx is not None else "never_crossed"
    return {"onset_idx": onset_idx, "threshold_used": threshold, "detection_method": method}


# =================================================================
# 10b. SAFETY SHELL: FUSION OF CONFORMAL EXPANSION + EWS AGREEMENT
# =================================================================
# Per Section 4.6.2 of the paper: the Shell alerts at "the earliest
# sustained AGREEMENT between these complementary indicators" -- i.e. an
# AND-fusion of (a) conformal prediction-set size sustainedly exceeding
# its nominal single-label value, and (b) the EWS (detrended AC1) onset.
# This was described in the Methodology but never actually computed or
# evaluated anywhere in the codebase until now.
def compute_safety_shell_onset(alphas, mean_setsize, ac1_onset_idx,
                                setsize_threshold=1.0, persistence=2):
    """
    DEPRECATED (kept for reference / ablation comparison against the
    voting redesign below). Returns the Safety Shell's alert index: the
    LATER of (a) conformal set-size sustained expansion and (b) the EWS
    onset -- i.e. the point at which BOTH independent evidence streams
    concur the system is unstable.

    WHY THIS WAS SUPERSEDED: conformal prediction-set size, as constructed
    here, is essentially a monotonic transform of max softmax probability
    -- a sample's set only grows once its top-class probability drops
    below the calibrated cutoff. Under a shift that makes the model
    CONFIDENTLY WRONG rather than appropriately uncertain (observed
    empirically via MSP AUROC below chance under the FDA-based RSNA
    shift), set size never expands, so this AND-fusion's conformal half
    goes permanently blind -- and since AND requires BOTH signals, the
    whole Shell never fires, even while EWS alone would have. This isn't
    a fusion-rule strictness problem (OR-fusion would "fix" this specific
    run but leaves the same single-point-of-failure for any future
    confidence-distorting shift); it's that one of the two ingredients
    isn't actually independent evidence for this failure mode. See
    compute_safety_shell_onset_voting for the redesigned version.
    """
    conformal = detect_fixed_threshold_onset(alphas, mean_setsize, threshold=setsize_threshold,
                                              persistence=persistence)
    if conformal["onset_idx"] is None or ac1_onset_idx is None:
        return {
            "shell_onset_idx": None,
            "shell_alpha_ews": alphas[-1],
            "agreement_reached": False,
            "conformal_onset_idx": conformal["onset_idx"],
            "ac1_onset_idx": ac1_onset_idx,
        }
    shell_idx = max(conformal["onset_idx"], ac1_onset_idx)
    return {
        "shell_onset_idx": shell_idx,
        "shell_alpha_ews": alphas[shell_idx],
        "agreement_reached": True,
        "conformal_onset_idx": conformal["onset_idx"],
        "ac1_onset_idx": ac1_onset_idx,
    }


def compute_safety_shell_onset_voting(alphas, voter_onsets, m=2):
    """
    REDESIGNED Safety Shell: an m-out-of-n voting ensemble across
    conceptually DISTINCT signal families (by default: detrended-AC1 EWS
    [dynamical-systems-based], conformal set-size [softmax-calibration-
    based], and Mahalanobis distance [feature-space-based]), rather than
    a strict 2-signal AND. This is deliberately more than a simple
    AND->OR swap:

    1. AND-fusion (the original design) fails catastrophically if EITHER
       signal has a blind spot for a given shift type (as conformal
       set-size does under confidence-distorting shift) -- one blind
       signal silences the whole Shell.
    2. Plain OR-fusion avoids that failure but over-triggers: any single
       noisy indicator can fire the Shell alone, and it provides no way
       to know retrospectively whether the alert reflects real convergent
       evidence or one indicator's false positive.
    3. m-out-of-n voting (m=2 of 3 here) requires genuine agreement from
       AT LEAST TWO conceptually independent detection paradigms, so it
       is robust to any ONE family going blind (unlike AND) while still
       requiring real convergence, not a single trigger (unlike OR).
       This mirrors established fault-detection/sensor-fusion practice
       and generalizes the same "how many indicators agree" logic already
       used in composite_ews_agreement, lifted from individual EWS
       statistics up to independent detector FAMILIES.

    `voter_onsets`: dict of {name: onset_dict}, where each onset_dict has
    at minimum 'onset_idx' and 'detection_method' (as returned by
    detect_sustained_onset / detect_fixed_threshold_onset). CRITICALLY, a
    voter only counts toward the vote if detection_method indicates a
    GENUINE sustained crossing ("sustained_threshold_crossing") -- a
    fallback argmax guess ("fallback_argmax_no_sustained_crossing") does
    NOT count as a vote. Without this distinction, voting would be
    meaningless: detect_sustained_onset's fallback behavior means AC1 and
    Mahalanobis onsets are ALMOST ALWAYS non-None (they always produce
    SOME index via argmax fallback if nothing genuine is found), so
    naively counting "onset_idx is not None" would make the Shell almost
    always fire regardless of whether real evidence existed -- the
    opposite failure mode from the AND-fusion bug this replaces.

    Returns the alpha at which the count of GENUINELY-fired voters first
    reaches `m` (the m-th earliest genuine onset among those that fired).
    """
    genuine = [(name, d["onset_idx"]) for name, d in voter_onsets.items()
               if d.get("detection_method") == "sustained_threshold_crossing" and d.get("onset_idx") is not None]
    n_total = len(voter_onsets)

    if len(genuine) < m:
        return {
            "shell_onset_idx": None,
            "shell_alpha_ews": alphas[-1],
            "agreement_reached": False,
            "n_voted": len(genuine), "m_required": m, "n_total": n_total,
            "voter_details": {name: d.get("detection_method") for name, d in voter_onsets.items()},
        }

    sorted_onsets = sorted(idx for _, idx in genuine)
    shell_idx = sorted_onsets[m - 1]  # m-th smallest -> the alpha at which the vote count first reaches m
    return {
        "shell_onset_idx": shell_idx,
        "shell_alpha_ews": alphas[shell_idx],
        "agreement_reached": True,
        "n_voted": len(genuine), "m_required": m, "n_total": n_total,
        "voter_details": {name: d.get("detection_method") for name, d in voter_onsets.items()},
    }


def compute_safety_shell_onset_stratified(alphas, ac1_onset, calibration_independent_onsets):
    """
    FAMILY-STRATIFIED Safety Shell. Requires a genuine vote from EACH of two
    conceptually distinct families:

    DYNAMICAL FAMILY (must contribute a genuine vote): AC1's own sustained
    calibrated-threshold crossing. An earlier version of this function also
    accepted a composite-evidence fallback (variance/skewness corroborating a
    fallback argmax pick), which was appropriate when the underlying onset
    detector could still return a look-ahead argmax pick. The current
    calibrated onset detector (apply_onset_threshold) never returns a
    fallback pick -- it reports a genuine crossing or an honest absence of
    one -- so that fallback path no longer has anything to rescue, and
    reinstating it would let the dynamical family satisfy its vote through a
    second, unvalidated statistical test precisely when its own primary test
    (Section 5.6: ~53%, chance-level) has already been reported as unreliable.
    Requiring AC1's own genuine crossing, and nothing else, keeps this test
    at least as strict as the finding it is meant to act on.

    REPRESENTATION- AND UNCERTAINTY-BASED FAMILY (must contribute a genuine
    vote): whichever of {Mahalanobis, Energy, MC Dropout} fires genuinely
    EARLIEST.

    A genuine vote is one with detection_method == "calibrated_threshold_crossing"
    (the current calibrated onset detector's label for a real, sustained
    crossing). An earlier version of this function checked for
    "sustained_threshold_crossing", the label produced by a deprecated onset
    detector; apply_onset_threshold never returns that string, so that check
    could never match and the dynamical family could never register a vote
    regardless of the data -- silently forcing agreement_reached to False in
    every case. Fixed here to check the correct, current label.

    `calibration_independent_onsets`: dict of {name: onset_dict}, e.g.
    {"mahalanobis": ..., "energy": ..., "mc_dropout": ...}.

    Returns the alpha at which BOTH families have independently contributed
    a genuine vote (the LATER of the two families' earliest genuine
    confirmation), plus which specific member satisfied each family slot,
    for full auditability.
    """
    dynamical_idx, dynamical_source = None, None
    if ac1_onset.get("detection_method") == "calibrated_threshold_crossing":
        dynamical_idx = ac1_onset["onset_idx"]
        dynamical_source = "ac1_calibrated_crossing"

    genuine_calib = [(name, d["onset_idx"]) for name, d in calibration_independent_onsets.items()
                     if d.get("detection_method") == "calibrated_threshold_crossing" and d.get("onset_idx") is not None]
    calib_idx, calib_source = None, None
    if genuine_calib:
        calib_source, calib_idx = min(genuine_calib, key=lambda x: x[1])

    base_result = {
        "dynamical_vote_source": dynamical_source, "dynamical_vote_idx": dynamical_idx,
        "calibration_independent_vote_source": calib_source, "calibration_independent_vote_idx": calib_idx,
    }

    if dynamical_idx is None or calib_idx is None:
        return {
            "shell_onset_idx": None,
            "shell_alpha_ews": alphas[-1],
            "agreement_reached": False,
            **base_result,
        }

    shell_idx = max(dynamical_idx, calib_idx)
    return {
        "shell_onset_idx": shell_idx,
        "shell_alpha_ews": alphas[shell_idx],
        "agreement_reached": True,
        **base_result,
    }


def compute_blocked_silent_counts(correct_matrix, alert_idx):
    """
    For every (alpha, sample) pair where the model was WRONG, classifies
    the failure as:
      - 'blocked': alert_idx is not None and this alpha is at/after the
        alert point -- the monitor had already raised an alarm by then,
        so this failure would trigger deferral/clinician review.
      - 'silent': the alert had not yet fired -- the failure passes
        through completely undetected.
    This is the direct computation behind the paper's originally-reported
    "Failures Blocked / Silent Failures" columns, which were asserted in
    the first version of Table 1 but never actually computed from data.
    """
    correct_matrix = np.asarray(correct_matrix)
    wrong = (correct_matrix == 0)

    if alert_idx is None:
        blocked, silent = 0, int(wrong.sum())
    else:
        blocked = int(wrong[alert_idx:].sum())
        silent = int(wrong[:alert_idx].sum())

    total = blocked + silent
    blocked_fraction = (blocked / total) if total > 0 else float("nan")
    return {"blocked": blocked, "silent": silent, "total_failures": total,
            "blocked_fraction": blocked_fraction}


def paired_bootstrap_blocked_fraction_test(correct_matrix, alert_idx_a, alert_idx_b,
                                            n_boot=500, seed=0, return_samples=False):
    """
    Paired bootstrap comparison of blocked-failure fraction between two
    detectors (e.g. Safety Shell vs. an OOD baseline). Resamples the SAME
    panel columns for both detectors in each replicate (paired design,
    controlling for shared sampling variability) rather than comparing two
    independent bootstrap distributions. `return_samples=True` includes
    the raw per-replicate arrays so multiple seeds' results can be pooled.
    """
    rng = np.random.RandomState(seed)
    n_alphas, panel_size = correct_matrix.shape
    frac_a = np.empty(n_boot)
    frac_b = np.empty(n_boot)

    for b in range(n_boot):
        cols = rng.choice(panel_size, panel_size, replace=True)
        cm = correct_matrix[:, cols]
        frac_a[b] = compute_blocked_silent_counts(cm, alert_idx_a)["blocked_fraction"]
        frac_b[b] = compute_blocked_silent_counts(cm, alert_idx_b)["blocked_fraction"]

    diffs = frac_a - frac_b
    valid = ~np.isnan(diffs)
    diffs = diffs[valid]

    # NaN here means the model made ZERO misclassifications across every
    # bootstrap replicate at this alpha range (0/0 blocked_fraction is
    # genuinely undefined, not a bug) -- e.g. an architecture that stayed
    # essentially perfect throughout the tested shift range. Suppress the
    # expected RuntimeWarning and label this explicitly rather than
    # letting a bare NaN or a console warning leak into "clean" output.
    all_nan_a = np.all(np.isnan(frac_a))
    all_nan_b = np.all(np.isnan(frac_b))
    with np.errstate(invalid="ignore"):
        obs_diff = float(np.nanmean(frac_a) - np.nanmean(frac_b)) if not (all_nan_a or all_nan_b) else float("nan")

    if len(diffs) == 0:
        result = {"observed_diff": obs_diff, "ci95": (float("nan"),) * 2,
                  "p_value": float("nan"), "significant": False,
                  "no_failures_observed": True,
                  "blocked_fraction_a_mean": float("nan"), "blocked_fraction_b_mean": float("nan")}
    else:
        p_value = float(min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0))
        ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        with np.errstate(invalid="ignore"):
            frac_a_mean = float(np.nanmean(frac_a)) if not all_nan_a else float("nan")
            frac_b_mean = float(np.nanmean(frac_b)) if not all_nan_b else float("nan")
        result = {
            "observed_diff": obs_diff, "ci95": ci, "p_value": p_value, "significant": p_value < 0.05,
            "blocked_fraction_a_mean": frac_a_mean, "blocked_fraction_b_mean": frac_b_mean,
            "no_failures_observed": False,
        }
    if return_samples:
        result["frac_a_samples"] = frac_a
        result["frac_b_samples"] = frac_b
    return result


# =================================================================
# 10c. FALSE-ALARM-RATE ANALYSIS
# (does the Safety Shell's conservatism actually buy fewer false alarms,
#  or is it strictly worse than the best individual detector on every axis?)
# =================================================================
def false_alarm_rate_analysis(alphas, signals_dict, n_surrogates=400, seed=0,
                               persistence=2, nominal_fpr=0.05, conformal_threshold=1.0,
                               calibrated_thresholds=None, signal_pools=None):
    """
    Estimates each detector's FALSE-ALARM RATE using a JOINT CIRCULAR-SHIFT
    surrogate null (spec v3, Sections 5-6). This replaces the earlier
    independent-permutation null, which was correctly criticised for
    destroying cross-signal dependence and thereby understating the chance
    of a spurious coincident Safety Shell alert (PI concern #4).

    METHOD. In each surrogate replicate a SINGLE common circular shift is
    applied to ALL signals together. A joint shift:
      * preserves each signal's own within-signal autocorrelation (unlike
        permutation, which destroys it), and
      * preserves the cross-signal dependence BETWEEN detectors (unlike
        independent permutation, which destroys it) -- so if two detector
        families genuinely tend to move together, the surrogate keeps that,
        and the Shell's two-family agreement requirement is tested against a
        FAIR null rather than one rigged in its favour.
    Circular shifting removes only the phase alignment between the signals
    and the shift-intensity axis, i.e. it destroys any genuine alpha-trend
    (the thing an onset detector is supposed to key on) while leaving the
    signals' joint second-order structure intact.

    Onset detection inside the surrogate uses the SAME population-calibrated
    thresholds as the real analysis when `calibrated_thresholds` is supplied
    (a dict {detector: threshold}); this gives every detector the identical
    stated operating point, so false-alarm rates are directly comparable and
    the Shell cannot appear better merely because its components use a
    stricter threshold. If thresholds are not supplied the routine falls back
    to per-surrogate calibration.

    `signals_dict`: keys 'ac1','setsize','msp','energy','mahalanobis',
    'mc_dropout', each an alpha-indexed mean signal.

    The Safety Shell's dynamical family is evaluated via AC1's own genuine
    calibrated-threshold crossing only (see compute_safety_shell_onset_stratified
    for why a composite-evidence fallback is not used here).

    Returns {detector_name: false_alarm_rate} for the six detectors plus the
    fused Safety Shell.
    """
    rng = np.random.RandomState(seed)
    alphas = np.asarray(alphas)
    n = len(alphas)

    fires = {"AC1": 0, "Conformal": 0, "MSP": 0, "Energy": 0,
             "Mahalanobis": 0, "MCDropout": 0, "SafetyShell": 0,
             "SingleFamilyRepresentation": 0}

    ct = calibrated_thresholds or {}
    keys = ["ac1", "setsize", "msp", "energy", "mahalanobis", "mc_dropout"]

    # The no-degradation null must be built from the SAME pooled nominal data the
    # thresholds were calibrated on, or the realised operating point will not match
    # the nominal target (verified: single-run nominal inflates the rate ~15x).
    # `signal_pools` (optional): {key: list-of-per-run-arrays}. When supplied, each
    # surrogate JOINTLY resamples nominal-region time points from the pooled runs --
    # the SAME drawn (run, index) pairs across every detector -- preserving
    # cross-detector dependence (PI concern #4) while representing no degradation.
    # When absent, falls back to this single run's nominal region.
    def _nominal_pool(key):
        if signal_pools and key in signal_pools:
            segs = []
            for arr in signal_pools[key]:
                v = np.asarray(arr, dtype=float)
                v = v[~np.isnan(v)]
                if len(v) >= 6:
                    segs.append(v[:max(4, int(len(v) * 0.4))])
            if segs:
                return segs
        v = np.asarray(signals_dict[key], dtype=float)
        v = v[~np.isnan(v)]
        return [v[:max(4, int(len(v) * 0.4))]]

    pools = {k: _nominal_pool(k) for k in keys}
    n_runs_pool = max(len(pools[k]) for k in keys)

    def _onset(sig, thr_key, invert=False):
        s = -np.asarray(sig, dtype=float) if invert else np.asarray(sig, dtype=float)
        thr = ct.get(thr_key)
        if thr is None:
            thr = calibrate_population_threshold(pools[thr_key.lower()] if thr_key.lower() in pools else [s],
                                                 persistence=persistence, nominal_fpr=nominal_fpr)
        return apply_onset_threshold(alphas, s, thr, persistence=persistence)

    for _ in range(n_surrogates):
        # JOINT draw: pick a run then an index within its nominal region, same
        # (run, idx) pairs reused across all detectors to preserve dependence.
        run_pick = rng.randint(n_runs_pool, size=n)
        surr = {}
        for k in keys:
            pk = pools[k]
            idx_pick = [rng.randint(len(pk[min(rp, len(pk) - 1)])) for rp in run_pick]
            surr[k] = np.array([pk[min(rp, len(pk) - 1)][ix] for rp, ix in zip(run_pick, idx_pick)])

        ac1_onset = _onset(surr["ac1"], "AC1")
        conformal_onset = detect_fixed_threshold_onset(alphas, surr["setsize"],
                                                       threshold=conformal_threshold,
                                                       persistence=persistence)
        msp_onset = _onset(surr["msp"], "MSP", invert=True)
        energy_onset = _onset(surr["energy"], "Energy")
        maha_onset = _onset(surr["mahalanobis"], "Mahalanobis")
        mc_onset = _onset(surr["mc_dropout"], "MCDropout")

        for name, onset in [("AC1", ac1_onset), ("MSP", msp_onset),
                             ("Energy", energy_onset), ("Mahalanobis", maha_onset),
                             ("MCDropout", mc_onset)]:
            if onset["detection_method"] == "calibrated_threshold_crossing":
                fires[name] += 1
        if conformal_onset["detection_method"] == "sustained_threshold_crossing":
            fires["Conformal"] += 1

        calib_onsets = {"mahalanobis": maha_onset, "energy": energy_onset, "mc_dropout": mc_onset}
        shell = compute_safety_shell_onset_stratified(alphas, ac1_onset, calib_onsets)
        if shell["agreement_reached"]:
            fires["SafetyShell"] += 1

        # Single-family design: an alert requires only ONE of the three
        # representation- and uncertainty-based detectors to fire (no
        # dynamical-family corroboration required). This is the compound
        # (union) event, not any individual detector's own rate -- P(A or B
        # or C) is generally higher than max(P(A),P(B),P(C)), so it must be
        # tracked as its own quantity, not inferred from the per-detector
        # rows above.
        if any(o["detection_method"] == "calibrated_threshold_crossing"
              for o in (maha_onset, energy_onset, mc_onset)):
            fires["SingleFamilyRepresentation"] += 1

    return {name: count / n_surrogates for name, count in fires.items()}




# =================================================================
# 11. THRESHOLD SENSITIVITY ANALYSIS  (addresses "why tau=0.70?")
# =================================================================
def threshold_sensitivity_analysis(alphas, entropy_matrix, correct_matrix,
                                    thresholds=(0.65, 0.70, 0.75, 0.80),
                                    n_boot=200, seed=0, window=5, detrend=True):
    """
    Recomputes alpha*/lead_time across a range of clinical-acceptability
    thresholds tau, reusing the SAME cached entropy/correctness matrices
    (no re-inference needed). Directly addresses the "why was tau=0.70
    chosen, and how sensitive are the conclusions to that choice"
    critique -- cheap to run given the bootstrap infrastructure already
    exists for a single threshold.
    """
    rng = np.random.RandomState(seed)
    panel_size = entropy_matrix.shape[1]
    state_series = entropy_matrix.mean(axis=1)
    accuracy = correct_matrix.mean(axis=1)
    ews = windowed_ews_indicators(state_series, window=window, detrend=detrend)

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2

    rows = []
    for tau in thresholds:
        m = calculate_transition_metrics(alphas, accuracy, ews["ac1"], threshold=tau)

        boot_alpha_star, boot_lead = [], []
        for _ in range(n_boot):
            cols = rng.choice(panel_size, panel_size, replace=True)
            b_state = entropy_matrix[:, cols].mean(axis=1)
            b_acc = correct_matrix[:, cols].mean(axis=1)
            b_ews = windowed_ews_indicators(b_state, window=window, detrend=detrend)
            bm = calculate_transition_metrics(alphas, b_acc, b_ews["ac1"], threshold=tau)
            boot_alpha_star.append(bm["alpha_star"])
            boot_lead.append(bm["lead_time"])

        rows.append({
            "threshold": tau,
            "alpha_star": m["alpha_star"],
            "alpha_star_sustained": m["alpha_star_sustained"],
            "alpha_star_ci95": ci(boot_alpha_star),
            "lead_time": m["lead_time"],
            "lead_time_ci95": ci(boot_lead),
            "boot_alpha_star_samples": boot_alpha_star,  # raw, for cross-seed pooling
            "boot_lead_samples": boot_lead,               # raw, for cross-seed pooling
        })
    return rows


# =================================================================
# 10c. AUC-BASED METRIC  (Experiment 2: histopathology, AUC replaces accuracy)
# =================================================================
# Everything in calculate_transition_metrics/detect_persistent_failure is
# generic over "the performance metric array being thresholded" -- it was
# never hardcoded to accuracy specifically. These two functions let the
# SAME downstream machinery (Safety Shell, bootstrap, significance
# testing, threshold sensitivity) operate on AUC instead, by supplying an
# AUC-per-alpha array in place of the accuracy-per-alpha array, with zero
# changes needed to the core statistical functions themselves.
def compute_alpha_indexed_auc(prob_matrix, labels):
    """
    prob_matrix: (n_alphas, panel_size) predicted probability of the
    positive (tumor) class, from the FIXED panel at each alpha.
    labels: (panel_size,) fixed ground-truth binary labels (same panel,
    same labels at every alpha, since it's the same tracked images).
    Returns an (n_alphas,) array of ROC-AUC, one per shift level.
    """
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for AUC computation.")
    labels = np.asarray(labels)
    n_alphas = prob_matrix.shape[0]
    auc_per_alpha = np.full(n_alphas, np.nan)
    for a in range(n_alphas):
        try:
            auc_per_alpha[a] = roc_auc_score(labels, prob_matrix[a])
        except ValueError:
            # only one class present in this resample -- AUC undefined
            auc_per_alpha[a] = np.nan
    return auc_per_alpha


def threshold_sensitivity_analysis_auc(alphas, entropy_matrix, prob_matrix, labels,
                                        thresholds=(0.65, 0.70, 0.75, 0.80),
                                        n_boot=200, seed=0, window=5, detrend=True):
    """
    AUC-metric counterpart to threshold_sensitivity_analysis. Identical in
    spirit (recomputes alpha*/lead_time across a tau grid, reusing cached
    matrices, no re-inference) but thresholds AUC instead of accuracy, and
    resamples (prob_matrix, labels) jointly per bootstrap replicate so the
    AUC recomputed on each replicate remains a valid ROC-AUC.
    """
    rng = np.random.RandomState(seed)
    panel_size = entropy_matrix.shape[1]
    state_series = entropy_matrix.mean(axis=1)
    labels = np.asarray(labels)
    auc_curve = compute_alpha_indexed_auc(prob_matrix, labels)
    ews = windowed_ews_indicators(state_series, window=window, detrend=detrend)

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2

    rows = []
    for tau in thresholds:
        m = calculate_transition_metrics(alphas, auc_curve, ews["ac1"], threshold=tau)

        boot_alpha_star, boot_lead = [], []
        for _ in range(n_boot):
            cols = rng.choice(panel_size, panel_size, replace=True)
            b_state = entropy_matrix[:, cols].mean(axis=1)
            b_auc = compute_alpha_indexed_auc(prob_matrix[:, cols], labels[cols])
            b_ews = windowed_ews_indicators(b_state, window=window, detrend=detrend)
            bm = calculate_transition_metrics(alphas, b_auc, b_ews["ac1"], threshold=tau)
            boot_alpha_star.append(bm["alpha_star"])
            boot_lead.append(bm["lead_time"])

        rows.append({
            "threshold": tau,
            "alpha_star": m["alpha_star"],
            "alpha_star_sustained": m["alpha_star_sustained"],
            "alpha_star_ci95": ci(boot_alpha_star),
            "lead_time": m["lead_time"],
            "lead_time_ci95": ci(boot_lead),
            "boot_alpha_star_samples": boot_alpha_star,
            "boot_lead_samples": boot_lead,
        })
    return rows



# =================================================================
def pool_across_seeds(list_of_arrays):
    """
    Pools bootstrap replicate arrays from multiple independently-trained
    seeds into ONE combined array, so the resulting CI reflects BOTH
    sources of uncertainty: panel-resampling variance (within a seed) and
    training-stochasticity variance (across seeds) -- a hierarchical/
    nested bootstrap, rather than reporting only one source and ignoring
    the other.
    """
    pooled = np.concatenate([np.asarray(a, dtype=float) for a in list_of_arrays])
    pooled = pooled[~np.isnan(pooled)]
    if len(pooled) == 0:
        return {"pooled_mean": float("nan"), "ci95": (float("nan"),) * 2, "n_pooled": 0}
    return {
        "pooled_mean": float(np.mean(pooled)),
        "ci95": (float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))),
        "n_pooled": len(pooled),
    }


def seed_variance_summary(values_across_seeds):
    """Simple mean +/- std across independently-trained seeds (raw, unpooled)."""
    v = np.asarray(values_across_seeds, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "values": v.tolist()}


# =================================================================
# 9. SIGNIFICANCE TESTING ACROSS ARCHITECTURES / METHODS
# =================================================================
def permutation_test_lead_time_diff(boot_samples_a, boot_samples_b, n_perm=10000, seed=0):
    """
    Two-sided permutation test on the difference in mean lead time between
    two architectures (or the proposed method vs. a baseline), reusing the
    bootstrap replicates already computed -- no extra inference required.
    """
    rng = np.random.RandomState(seed)
    a = np.asarray(boot_samples_a, dtype=float)
    b = np.asarray(boot_samples_b, dtype=float)
    obs_diff = a.mean() - b.mean()

    pooled = np.concatenate([a, b])
    n_a = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()

    p_value = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))
    return {"observed_diff": float(obs_diff), "p_value": p_value, "significant": p_value < 0.05}


# =================================================================
# 10. ORCHESTRATOR: TIES EVERYTHING TOGETHER PER ARCHITECTURE
# =================================================================
def run_full_analysis(model_suite, test_ds, cal_loader, apply_shift_fn, device,
                       alphas, panel_size=400, n_boot=500, threshold=0.70, seed=42):
    """
    Runs the corrected pipeline for every model in `model_suite`, producing:
      - point estimates + bootstrap CIs for alpha*, sharpness, lead time
      - composite EWS agreement (AC1 / variance / skewness convergence)
      - baseline OOD (MSP, Energy) lead times + detection AUROC for comparison
      - pairwise permutation-test significance between architectures'
        lead-time distributions

    Re-run this against existing checkpoints to see how much Table 1 shifts
    once (a) onset detection uses sustained crossing instead of argmax,
    (b) alpha* requires persistence, and (c) autocorrelation is computed
    over a genuine alpha-indexed time series instead of shuffled batches.
    """
    summary, bootstrap_cache, baseline_cache = {}, {}, {}

    for name, model in model_suite.items():
        boot = bootstrap_transition_metrics(
            model, test_ds, apply_shift_fn, device, alphas,
            panel_size=panel_size, n_boot=n_boot, threshold=threshold, seed=seed
        )
        bootstrap_cache[name] = boot
        summary[name] = boot["point_estimate"]
        summary[name]["alpha_star_ci95"] = boot["alpha_star_ci"]
        summary[name]["sharpness_ci95"] = boot["sharpness_ci"]
        summary[name]["lead_time_ci95"] = boot["lead_time_ci"]
        summary[name]["composite_n_agree"] = boot["composite_agreement"]["n_agree"]
        summary[name]["convergent_csd_evidence"] = boot["composite_agreement"]["convergent_csd_evidence"]

        # --- baseline OOD comparison ---
        thresholds = calibrate_baseline_thresholds(model, cal_loader, device)
        panel = build_fixed_panel_indices(test_ds, panel_size, seed=seed)
        loader = DataLoader(Subset(test_ds, panel), batch_size=32, shuffle=False)

        msp_means, energy_means, id_msp, id_energy = [], [], None, None
        for a_idx, alpha in enumerate(alphas):
            step_msp, step_energy = [], []
            for inputs, _ in loader:
                inputs = apply_shift_fn(inputs.to(device), alpha)
                msp, energy = compute_baseline_scores(model, inputs)
                step_msp.extend(msp)
                step_energy.extend(energy)
            msp_means.append(np.mean(step_msp))
            energy_means.append(np.mean(step_energy))
            if alpha == alphas[0]:
                id_msp, id_energy = np.array(step_msp), np.array(step_energy)

        onset_dict = baseline_onset_detection(alphas, msp_means, energy_means)
        baseline_lead = compare_lead_times(summary[name]["alpha_star"], onset_dict)
        baseline_auroc = {
            "msp_auroc_final_alpha": detection_auroc_vs_alpha(id_msp, np.array(step_msp)),
            "energy_auroc_final_alpha": detection_auroc_vs_alpha(id_energy, np.array(step_energy)),
        }
        baseline_cache[name] = {**baseline_lead, **baseline_auroc}
        summary[name].update(baseline_cache[name])

    # --- pairwise significance testing across architectures ---
    pairwise = {}
    names = list(model_suite.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            pairwise[key] = permutation_test_lead_time_diff(
                bootstrap_cache[names[i]]["lead_time_samples"],
                bootstrap_cache[names[j]]["lead_time_samples"],
            )

    return {"summary": summary, "pairwise_significance": pairwise, "bootstrap_cache": bootstrap_cache}


# =====================================================================
# CSD PRECURSOR AUDIT (spec v3, Section 12)
# Real-data validation of whether the classical critical-slowing-down
# precursor (rising lag-1 autocorrelation of entropy) is actually present
# BEFORE clinical failure. Runs on the per-image entropy_matrix that the
# persistence fix now saves. Prespecified interpretation: a positive-slope
# fraction near 0.5 confirms the precursor is ABSENT (chance-level);
# a fraction above ~0.75 would overturn the "CSD largely absent" finding.
# =====================================================================
def csd_precollapse_slope_audit(entropy_matrix, alphas, alpha_star,
                                 window=5, detrend_sigma=2.0):
    """
    For a single run, test whether detrended lag-1 autocorrelation of the
    panel-mean entropy trends UPWARD in the pre-failure window (alpha < alpha_star).

    A genuine critical-slowing-down precursor implies a positive pre-failure
    slope. Returns the slope, a boolean rises_before_failure, and the number
    of valid pre-failure points the slope was computed from.

    entropy_matrix: (n_alpha, n_panel) per-image entropy. If a 1D mean series
                    is passed it is used directly (backward compatible).
    alpha_star:     Critical Failure Point for this run. If None or censored
                    (>= max alpha), the whole trajectory is used.
    """
    ent = np.asarray(entropy_matrix)
    state = ent.mean(axis=1) if ent.ndim == 2 else ent
    ac1 = windowed_ews_indicators(state, window=window,
                                  detrend=True, detrend_sigma=detrend_sigma)["ac1"]
    alphas = np.asarray(alphas)

    # restrict to the pre-failure window
    if alpha_star is not None and np.isfinite(alpha_star) and alpha_star < alphas.max():
        pre_mask = alphas < alpha_star
    else:
        pre_mask = np.ones_like(alphas, dtype=bool)

    pre_ac1 = ac1[pre_mask]
    valid = pre_ac1[~np.isnan(pre_ac1)]
    if len(valid) < 4:
        return {"slope": np.nan, "rises_before_failure": False,
                "n_prefailure_points": int(len(valid)), "evaluable": False}

    slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0])
    return {"slope": slope, "rises_before_failure": bool(slope > 0),
            "n_prefailure_points": int(len(valid)), "evaluable": True}


def csd_audit_across_runs(runs, window=5):
    """
    Aggregate csd_precollapse_slope_audit over many runs.
    `runs`: list of dicts each with keys 'entropy_matrix' (or 'entropy_means'),
            'alphas', 'alpha_star'. Returns the fraction of evaluable runs whose
            AC1 rises before failure, plus the per-run detail.
    Prespecified reading (spec v3 S12): fraction ~0.5 => precursor absent;
    > ~0.75 => precursor present, reopen dynamical-route framing.
    """
    details = []
    for r in runs:
        em = r.get("entropy_matrix", r.get("entropy_means"))
        res = csd_precollapse_slope_audit(em, r["alphas"], r.get("alpha_star"), window=window)
        details.append(res)
    evaluable = [d for d in details if d["evaluable"]]
    frac = (np.mean([d["rises_before_failure"] for d in evaluable])
            if evaluable else np.nan)
    return {"fraction_rising_before_failure": float(frac) if evaluable else np.nan,
            "n_evaluable": len(evaluable), "n_total": len(details),
            "per_run": details}


# =====================================================================
# HIERARCHICAL BOOTSTRAP + CENSORING STATUS (spec v3, Sections 4, 6, 7)
# Operates POST-HOC on the per-image matrices saved by the persistence fix,
# so it needs no retraining and can be re-run for any analysis choice.
# =====================================================================
def classify_run_status(alpha_star_observed, shell_alpha_observed, alpha_max):
    """
    Assign each run exactly one status (spec v3 Section 4):
      'observed'         : a genuine collapse AND a Shell alert that PRECEDED it
      'lower_bound'      : Shell alerted but collapse right-censored (alpha* > max)
      'unwarned_failure' : genuine collapse with no alert, or an alert that arrived
                           at or after the failure point (Delta <= 0). A PRINCIPAL
                           SAFETY OUTCOME, counted not discarded.
      'non_estimable'    : neither observed

    The precedence requirement matters and was absent from an earlier revision of
    this function, which tested only whether an alert existed. Under that version a
    run whose alert arrived AFTER the model had already collapsed was scored as
    'observed' and inflated warning coverage. A warning that follows the failure it
    is meant to warn about provides no operational lead time, so it is now counted
    as an unwarned failure. This also removes the source of negative Delta-alpha
    values, which were the same defect showing up in the confidence intervals.

    `*_observed` args are the alpha value if observed, else None.
    """
    # alpha_star_observed is None whenever the caller found no sustained crossing
    # (see hierarchical_bootstrap_delta_alpha's delta_for). Any non-None value is
    # therefore already a genuine observed failure, including one that occurs at
    # the last tested shift level. Comparing it against alpha_max was redundant by
    # construction (alpha_star can never exceed alpha_max) and actively wrong at
    # the boundary: a failure at exactly alpha_max was being discarded as censored
    # instead of counted, silently dropping a real collapse from the results.
    collapse = alpha_star_observed is not None
    alerted = shell_alpha_observed is not None
    precedes = alerted and collapse and (shell_alpha_observed < alpha_star_observed)

    if collapse and precedes:
        return "observed"
    if alerted and not collapse:
        return "lower_bound"
    if collapse:                      # failed with no alert, or a late alert
        return "unwarned_failure"
    return "non_estimable"


def calibrate_leave_one_out_thresholds(all_runs, detector_specs, nominal_fpr=0.05,
                                        persistence=2):
    """
    For each run in `all_runs`, calibrate detector thresholds using ONLY the
    other runs' nominal-region data -- never that run's own.

    Addresses a calibration-leakage concern: the original design pooled
    nominal-region data ACROSS ALL runs of an experiment and then used the
    resulting threshold to evaluate onset detection on those SAME runs,
    including their own nominal data. That means each run's threshold was
    partly informed by its own baseline before being used to judge whether
    that run's own trajectory crossed it -- a non-independence between
    calibration and evaluation. Leave-one-run-out (LORO) calibration removes
    this by construction: the threshold applied to run i is always calibrated
    from the pool of every OTHER run in the experiment, so run i's own data
    can never influence the threshold used to judge it. This is what makes
    the onset detection a defensible model of a genuine runtime alert: a
    future run's threshold would likewise come from prior reference runs, not
    from the run being monitored.

    `all_runs`: list of run dicts (each containing the detector matrix keys
        named in `detector_specs`), one per (architecture, seed) combination
        in the experiment.
    `detector_specs`: list of tuples, each either
        (name, key, invert) -- for per-image matrices, averaged over images, or
        (name, key, invert, serial_dependence) -- as above, plus an explicit
        serial-dependence flag passed to calibrate_population_threshold.
        `serial_dependence` defaults to False when omitted. AC1's own
        detrended series (key "ac1_detrended") is already a 1-D alpha-indexed
        series (there is no per-image axis to average over, since it is
        computed from the panel-mean entropy trajectory) and is detected
        automatically by ndim; it should be passed with serial_dependence=True,
        since AC1 is itself an autocorrelation statistic and therefore
        serially dependent, unlike the near-independent representation- and
        uncertainty-based detectors.

    Returns a list of {name: threshold} dicts, one per run, in the same order
    as `all_runs`. The false-alarm surrogate analysis (Section 4.12) is a
    separate validity question -- it tests whether a GIVEN threshold fires on
    manufactured no-degradation data at the intended rate, not whether a
    specific run's own data influenced its own evaluation -- and continues to
    use the pooled threshold, since surrogates are not any single real run.
    """
    n = len(all_runs)
    specs = [(s + (False,))[:4] for s in detector_specs]  # normalize to 4-tuples
    means = {name: [] for name, _, _, _ in specs}
    for r in all_runs:
        for name, key, invert, _ in specs:
            arr = np.asarray(r[key], dtype=float)
            m = arr.mean(axis=1) if arr.ndim == 2 else arr  # AC1's series is already 1-D
            means[name].append(-m if invert else m)

    per_run_thresholds = []
    for i in range(n):
        thr_i = {}
        for name, key, invert, serial_dep in specs:
            pool = [means[name][j] for j in range(n) if j != i]
            thr_i[name] = calibrate_population_threshold(pool, nominal_fpr=nominal_fpr,
                                                          persistence=persistence, seed=i,
                                                          serial_dependence=serial_dep)
        per_run_thresholds.append(thr_i)
    return per_run_thresholds


def hierarchical_bootstrap_delta_alpha(runs_by_seed, alpha_max, n_boot=2000,
                                        threshold=0.70, nominal_fpr=0.05,
                                        persistence=2, seed=0,
                                        metric_is_auc=False,
                                        calibrated_thresholds=None):
    """
    Two-level bootstrap for Warning Lead Time (spec v3 Section 6): resample
    SEEDS with replacement, then resample IMAGES within each chosen seed's
    per-image matrices, recompute Delta-alpha per replicate, and return a
    percentile 95% CI that reflects BOTH training-stochasticity (seed) and
    panel-sampling variance. This replaces the invalid pooled-replicate
    permutation approach (PI concern #3): seeds are the top-level independent
    unit, so they are the level that must be resampled for a valid CI.

    `runs_by_seed`: list (one entry per seed) of dicts each containing at least
        'entropy_matrix' (n_alpha, n_panel), 'correct_matrix' (n_alpha, n_panel),
        'mahalanobis_matrix','energy_matrix','mc_dropout_var_matrix', 'alphas'.
    Delta-alpha is computed from the Safety Shell onset, which requires a
    genuine onset from BOTH the dynamical family (AC1's own calibrated
    crossing, recomputed from entropy_matrix under any image resampling) and
    the representation- and uncertainty-based family (earliest genuine
    crossing among Mahalanobis, Energy, MC-Dropout); the shell onset is the
    LATER of the two families' confirmations. An earlier revision of this
    function used only the representation-based family's earliest onset,
    silently bypassing the dynamical-family requirement described in
    Methods 4.9 -- found when a synthetic run with a genuine representation
    onset but no dynamical signal at all was still reported as "observed"
    with a real lead time.

    Returns point estimate, 95% CI, per-status counts, and the replicate samples.
    Censored / unwarned / non-estimable runs are handled per classify_run_status
    and NEVER imputed at 1.0.
    """
    rng = np.random.RandomState(seed)
    n_seeds = len(runs_by_seed)
    alphas = np.asarray(runs_by_seed[0]["alphas"])

    # Detection thresholds. `calibrated_thresholds` may be:
    #   - a LIST of one {name: threshold} dict per run in runs_by_seed, in the
    #     same order -- the leave-one-run-out (LORO) case, where run i's
    #     threshold was calibrated excluding run i's own data (see
    #     calibrate_leave_one_out_thresholds). This is what the reported
    #     analysis uses.
    #   - a single shared dict (or None) -- retained for standalone use only,
    #     applying one experiment-wide pooled threshold to every run. This
    #     fallback carries the calibration-leakage caveat described above and
    #     is not what analyze_experiment_posthoc uses for the reported results.
    ct = calibrated_thresholds
    if isinstance(ct, list):
        assert len(ct) == n_seeds, (
            "calibrated_thresholds list must have exactly one entry per run "
            f"in runs_by_seed (got {len(ct)} for {n_seeds} runs)")
        per_run_thr = ct
    else:
        def pooled(key, invert=False):
            out = []
            for r in runs_by_seed:
                m = np.asarray(r[key], dtype=float)
                s = (-m if invert else m).mean(axis=1)
                out.append(s)
            return out
        ctd = ct or {}
        thr_maha = ctd.get("Mahalanobis")
        thr_energy = ctd.get("Energy")
        thr_mc = ctd.get("MCDropout")
        thr_ac1 = ctd.get("AC1")
        if thr_maha is None:
            thr_maha = calibrate_population_threshold(pooled("mahalanobis_matrix"), nominal_fpr=nominal_fpr, persistence=persistence)
        if thr_energy is None:
            thr_energy = calibrate_population_threshold(pooled("energy_matrix"), nominal_fpr=nominal_fpr, persistence=persistence)
        if thr_mc is None:
            thr_mc = calibrate_population_threshold(pooled("mc_dropout_var_matrix"), nominal_fpr=nominal_fpr, persistence=persistence)
        if thr_ac1 is None:
            ac1_pool = [windowed_ews_indicators(np.asarray(r["entropy_matrix"], dtype=float).mean(axis=1))["ac1"]
                       for r in runs_by_seed]
            thr_ac1 = calibrate_population_threshold(ac1_pool, nominal_fpr=nominal_fpr,
                                                      persistence=persistence, serial_dependence=True)
        shared = {"Mahalanobis": thr_maha, "Energy": thr_energy, "MCDropout": thr_mc, "AC1": thr_ac1}
        per_run_thr = [shared] * n_seeds

    def _performance_curve(run, col_idx=None):
        """
        Reliability curve used to locate the Critical Failure Point.
        Radiography: panel accuracy. Histopathology: panel AUROC computed from
        the saved per-image probabilities and labels, matching the metric the
        experiment is reported in. Passing metric_is_auc=True without saved
        prob_matrix/panel_labels raises rather than silently falling back to
        accuracy, since a silent fallback would define collapse by the wrong
        metric (a defect found in an earlier revision of this function).
        """
        if not metric_is_auc:
            cm = np.asarray(run["correct_matrix"], dtype=float)
            if col_idx is not None:
                cm = cm[:, col_idx]
            return cm.mean(axis=1)

        if "prob_matrix" not in run or "panel_labels" not in run:
            raise KeyError("metric_is_auc=True requires 'prob_matrix' and "
                           "'panel_labels' in each run (saved by the histopathology "
                           "pipelines); refusing to fall back to accuracy.")
        pm = np.asarray(run["prob_matrix"], dtype=float)
        y = np.asarray(run["panel_labels"]).ravel()
        if col_idx is not None:
            pm = pm[:, col_idx]
            y = y[col_idx]
        curve = []
        for a in range(pm.shape[0]):
            curve.append(_auc_from_scores(y, pm[a, :]))
        return np.asarray(curve, dtype=float)

    def delta_for(run, run_idx, col_idx=None):
        perf = _performance_curve(run, col_idx)
        fail = detect_persistent_failure(alphas, perf, threshold, persistence)
        thr_i = per_run_thr[run_idx]

        # Representation- and uncertainty-based family: earliest genuine
        # crossing among Mahalanobis, Energy, MC-Dropout.
        rep_onsets = []
        for key, thr in [("mahalanobis_matrix", thr_i["Mahalanobis"]), ("energy_matrix", thr_i["Energy"]),
                          ("mc_dropout_var_matrix", thr_i["MCDropout"])]:
            mm = np.asarray(run[key], dtype=float)
            if col_idx is not None:
                mm = mm[:, col_idx]
            o = apply_onset_threshold(alphas, mm.mean(axis=1), thr, persistence=persistence)
            if o["onset_idx"] is not None:
                rep_onsets.append(o["alpha_ews"])
        rep_alpha = min(rep_onsets) if rep_onsets else None

        # Dynamical family: AC1's own genuine calibrated-threshold crossing.
        # AC1 is computed from the panel-mean entropy trajectory, so under
        # image resampling (col_idx) it must be recomputed from the raw
        # per-image entropy_matrix for the resampled subset, not read from a
        # precomputed series -- there is no per-image AC1 value to resample
        # directly, since detrending and the windowed statistic are applied
        # after the images have already been averaged within each alpha
        # level (Methods 4.7.1-4.7.3).
        em = np.asarray(run["entropy_matrix"], dtype=float)
        if col_idx is not None:
            em = em[:, col_idx]
        ac1_series = windowed_ews_indicators(em.mean(axis=1))["ac1"]
        ac1_o = apply_onset_threshold(alphas, ac1_series, thr_i["AC1"], persistence=persistence)
        dyn_alpha = ac1_o["alpha_ews"] if ac1_o["onset_idx"] is not None else None

        # Safety Shell: a genuine vote from BOTH families is required
        # (compute_safety_shell_onset_stratified implements the identical
        # rule; inlined here since both onsets are already alpha values
        # rather than indices at this point). The Shell's onset is the LATER
        # of the two families' earliest genuine confirmation -- both must
        # have independently corroborated before the alert counts.
        shell_alpha = max(rep_alpha, dyn_alpha) if (rep_alpha is not None and dyn_alpha is not None) else None

        astar = fail["alpha_star"] if fail["sustained"] else None
        status = classify_run_status(astar, shell_alpha, alpha_max)
        if status == "observed":
            return astar - shell_alpha, status
        if status == "unwarned_failure":
            return 0.0, status
        return np.nan, status  # lower_bound / non_estimable: excluded from Delta mean

    # point estimate + status counts on the real runs
    point_deltas, statuses = [], []
    for i, r in enumerate(runs_by_seed):
        d, st = delta_for(r, i)
        point_deltas.append(d); statuses.append(st)
    status_counts = {s: statuses.count(s) for s in
                     ["observed", "lower_bound", "unwarned_failure", "non_estimable"]}
    point = float(np.nanmean(point_deltas)) if np.any(~np.isnan(point_deltas)) else np.nan

    # hierarchical resampling
    # The evaluation panel is the SAME 400 physical images across every seed
    # (a fixed, shared panel by design -- Section 4.3). Each bootstrap replicate
    # therefore draws ONE set of resampled image indices, shared across every
    # seed selected in that replicate, rather than an independent resample per
    # seed. An earlier version drew `cols` inside the per-seed loop, which
    # treated the same physical images as if they varied independently across
    # seeds -- a genuine mismatch with the fixed-panel design that understates
    # the correlation in image-level difficulty across seeds and architectures.
    panel_size = np.asarray(runs_by_seed[0]["correct_matrix"]).shape[1]
    boot = []
    for _ in range(n_boot):
        seed_pick = rng.randint(n_seeds, size=n_seeds)
        cols = rng.choice(panel_size, panel_size, replace=True)  # shared across this replicate
        reps = []
        for si in seed_pick:
            run = runs_by_seed[si]
            d, _ = delta_for(run, si, col_idx=cols)
            reps.append(d)
        if np.any(~np.isnan(reps)):
            boot.append(np.nanmean(reps))
    boot = np.asarray(boot, dtype=float)
    ci = ((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
          if len(boot) else (np.nan, np.nan))

    # warning coverage: observed alerts / observed failures
    n_fail = status_counts["observed"] + status_counts["unwarned_failure"]
    coverage = (status_counts["observed"] / n_fail) if n_fail else np.nan

    return {"delta_alpha_point": point, "delta_alpha_ci95": ci,
            "status_counts": status_counts, "warning_coverage": coverage,
            "n_boot_effective": len(boot),
            "thresholds": per_run_thr}


# =====================================================================
# POST-HOC ANALYSIS ENTRY POINT (spec v3)
# The AUTHORITATIVE analysis. Runs AFTER all (arch, seed) runs have saved
# their per-image matrices. Calibrates thresholds on POOLED nominal data,
# then computes onsets, Warning Lead Time, false-alarm rates, the CSD audit,
# and status/coverage -- all from saved matrices, no retraining. This is the
# single tested path that produces the numbers reported in the paper, so the
# four live pipelines only need to run and save matrices (they do not each
# re-implement the calibrated analysis).
# =====================================================================
def analyze_experiment_posthoc(raw_signals_by_arch, threshold=0.70,
                                nominal_fpr=0.05, persistence=2,
                                alpha_max=1.0, n_boot=2000, seed=0,
                                metric_is_auc=False):
    """
    `raw_signals_by_arch`: {arch_name: [per-seed dict, ...]} exactly as returned
    by publication_utils.load_raw_signals, where each per-seed dict contains the
    per-image matrices saved by the persistence fix ('entropy_matrix',
    'correct_matrix','mahalanobis_matrix','energy_matrix','mc_dropout_var_matrix',
    'msp_matrix','setsize_matrix' or their *_means, and 'alphas').

    Returns, per architecture and pooled: population-calibrated thresholds, the
    Warning Lead Time point estimate with hierarchical-bootstrap CI, status
    counts (observed / lower_bound / unwarned_failure / non_estimable), warning
    coverage, per-detector false-alarm rates under the joint no-degradation null,
    and the CSD pre-collapse-slope audit fraction.

    All choices follow spec v3: Delta-alpha from the Shell onset (representation-
    based route), no 1.0 imputation, joint surrogate false alarms, hierarchical
    CIs, CSD absence reported as a finding.
    """
    out = {"by_arch": {}, "pooled": {}}

    def _mat(run, key):
        if key in run:
            return np.asarray(run[key], dtype=float)
        mkey = key.replace("_matrix", "_means").replace("_var", "")
        return None  # matrices required for full analysis

    all_runs = []
    for arch, runs in raw_signals_by_arch.items():
        for r in runs:
            all_runs.append(r)

    # Compute AC1's detrended series fresh from entropy_matrix for every run,
    # rather than relying on a saved "ac1_detrended" key -- that key was
    # never guaranteed to be present (its only other consumer used a NaN
    # fallback for exactly this reason). entropy_matrix is reliably saved by
    # every pipeline, so deriving AC1 from it here gives one consistent
    # source of truth for both LORO calibration and onset detection, and
    # matches exactly what must be recomputed per bootstrap replicate in
    # hierarchical_bootstrap_delta_alpha (Section 4.7.1-4.7.3).
    for r in all_runs:
        em = _mat(r, "entropy_matrix")
        if em is not None:
            r["ac1_detrended"] = windowed_ews_indicators(em.mean(axis=1))["ac1"]
        else:
            r["ac1_detrended"] = np.full(len(r["alphas"]), np.nan)

    # --- pooled nominal calibration across the WHOLE experiment ---
    def pooled_means(key, invert=False):
        segs = []
        for r in all_runs:
            m = _mat(r, key)
            if m is None:
                continue
            s = m.mean(axis=1)
            segs.append(-s if invert else s)
        return segs

    thr = {
        "Mahalanobis": calibrate_population_threshold(pooled_means("mahalanobis_matrix"),
                        nominal_fpr=nominal_fpr, persistence=persistence),
        "Energy": calibrate_population_threshold(pooled_means("energy_matrix"),
                        nominal_fpr=nominal_fpr, persistence=persistence),
        "MCDropout": calibrate_population_threshold(pooled_means("mc_dropout_var_matrix"),
                        nominal_fpr=nominal_fpr, persistence=persistence),
        "MSP": calibrate_population_threshold(pooled_means("msp_matrix", invert=True),
                        nominal_fpr=nominal_fpr, persistence=persistence),
        "AC1": None,  # reported as-is; weak detector, not forced to nominal
    }

    # --- leave-one-run-out calibration for the PRIMARY Warning Lead Time result ---
    # (see calibrate_leave_one_out_thresholds). `thr` above (pooled across all runs,
    # including each run's own data) remains the threshold used for the false-alarm
    # surrogate analysis below, which tests a different property -- whether a given
    # threshold fires on manufactured no-degradation data at the intended rate --
    # not whether a specific real run's own data influenced its own evaluation.
    detector_specs = [("Mahalanobis", "mahalanobis_matrix", False),
                       ("Energy", "energy_matrix", False),
                       ("MCDropout", "mc_dropout_var_matrix", False),
                       ("AC1", "ac1_detrended", False, True)]  # serial_dependence=True: AC1
                       # is itself an autocorrelation statistic and therefore serially
                       # dependent, unlike the near-independent representation- and
                       # uncertainty-based detectors above.
    loro_thresholds = calibrate_leave_one_out_thresholds(all_runs, detector_specs,
                                                          nominal_fpr=nominal_fpr,
                                                          persistence=persistence)
    # map each architecture's runs to their position in all_runs / loro_thresholds,
    # since hierarchical_bootstrap_delta_alpha needs the per-run LORO thresholds in
    # the same order as that architecture's own run list
    arch_loro_slices = {}
    _pos = 0
    for arch, runs in raw_signals_by_arch.items():
        arch_loro_slices[arch] = loro_thresholds[_pos:_pos + len(runs)]
        _pos += len(runs)

    # signal pools for the joint false-alarm null (same pooled nominal)
    sig_pools = {k: [ _mat(r, mk).mean(axis=1) for r in all_runs if _mat(r, mk) is not None ]
                 for k, mk in [("mahalanobis","mahalanobis_matrix"),("energy","energy_matrix"),
                               ("mc_dropout","mc_dropout_var_matrix"),("msp","msp_matrix"),
                               ("setsize","setsize_matrix")]}

    # --- per-architecture hierarchical bootstrap + CSD audit ---
    for arch, runs in raw_signals_by_arch.items():
        hb = hierarchical_bootstrap_delta_alpha(
            [{"alphas": r["alphas"],
              "correct_matrix": _mat(r, "correct_matrix"),
              "prob_matrix": _mat(r, "prob_matrix"),
              "panel_labels": r.get("panel_labels"),
              "entropy_matrix": _mat(r, "entropy_matrix"),
              "mahalanobis_matrix": _mat(r, "mahalanobis_matrix"),
              "energy_matrix": _mat(r, "energy_matrix"),
              "mc_dropout_var_matrix": _mat(r, "mc_dropout_var_matrix")} for r in runs],
            alpha_max=alpha_max, n_boot=n_boot, threshold=threshold,
            nominal_fpr=nominal_fpr, persistence=persistence, seed=seed,
            metric_is_auc=metric_is_auc,
            calibrated_thresholds=arch_loro_slices[arch])

        # alpha* for the CSD audit must use the SAME reliability metric as the
        # bootstrap, or the pre-failure window would be defined by a different
        # criterion than the failure itself.
        def _alpha_star_for(r):
            if metric_is_auc:
                pm = _mat(r, "prob_matrix")
                y = np.asarray(r["panel_labels"]).ravel()
                perf = np.asarray([_auc_from_scores(y, pm[a, :]) for a in range(pm.shape[0])])
            else:
                perf = _mat(r, "correct_matrix").mean(axis=1)
            return detect_persistent_failure(np.asarray(r["alphas"]), perf,
                                             threshold, persistence)["alpha_star"]

        csd = csd_audit_across_runs(
            [{"entropy_matrix": _mat(r, "entropy_matrix"), "alphas": r["alphas"],
              "alpha_star": _alpha_star_for(r)} for r in runs])
        out["by_arch"][arch] = {"delta_alpha": hb, "csd_audit": csd}

    # --- pooled false-alarm across all runs (representative signals + pools) ---
    rep = all_runs[0]
    far = false_alarm_rate_analysis(
        rep["alphas"],
        {"ac1": rep.get("ac1_detrended", np.full(len(rep["alphas"]), np.nan)),
         "setsize": _mat(rep, "setsize_matrix").mean(axis=1) if _mat(rep,"setsize_matrix") is not None else np.zeros(len(rep["alphas"])),
         "msp": _mat(rep,"msp_matrix").mean(axis=1),
         "energy": _mat(rep,"energy_matrix").mean(axis=1),
         "mahalanobis": _mat(rep,"mahalanobis_matrix").mean(axis=1),
         "mc_dropout": _mat(rep,"mc_dropout_var_matrix").mean(axis=1)},
        n_surrogates=400, seed=seed, persistence=persistence, nominal_fpr=nominal_fpr,
        calibrated_thresholds=thr, signal_pools=sig_pools)

    out["pooled"] = {"calibrated_thresholds": thr, "false_alarm_rates": far}
    return out
