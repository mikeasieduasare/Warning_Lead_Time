"""Matched-specificity audit (Sections 4.12, 5.5.2).

Tests whether a corroboration-based fusion rule contributes information
beyond its primary detector, by recalibrating that detector to the fusion
rule's own false-alarm rate and comparing coverage and Warning Lead Time.

Two modes:

  --mode far-search   estimate the overall false-alarm rate of the
                      representation family at candidate per-detector
                      nominal rates, to locate the matched operating point.
                      Slow: surrogate resampling over 9 LORO folds per
                      experiment.

  --mode coverage     at a chosen per-detector rate, compute coverage and
                      Warning Lead Time over confirmed failures, and the
                      paired table against the dual-family Safety Shell.

The threshold is selected in far-search using nominal-region data only.
Failure outcomes are not consulted until coverage is run, which preserves
the information constraint described in Section 4.12.
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from enhanced_ews_pipeline import (          # noqa: E402
    calibrate_population_threshold,
    apply_onset_threshold,
    detect_persistent_failure,
    false_alarm_rate_analysis,
    windowed_ews_indicators,
    _auc_from_scores,
)

TAU = 0.70
PERSISTENCE = 2
ARCHS = ["ResNet18", "DenseNet121", "ViT_B16"]

# representation- and uncertainty-based family
DET_KEYS = ["mahalanobis_matrix", "energy_matrix", "mc_dropout_var_matrix"]
POOL_KEY = {"mahalanobis_matrix": "mahalanobis",
            "energy_matrix": "energy",
            "mc_dropout_var_matrix": "mc_dropout"}
THR_KEY = {"mahalanobis": "Mahalanobis", "energy": "Energy", "mc_dropout": "MCDropout"}

EXPERIMENTS = [
    ("FDA-radiography",       "raw_signals_",          False),
    ("FDA-histopathology",    "exp2_raw_signals_",     True),
    ("Linear-radiography",    "linshift_raw_signals_", False),
    ("Linear-histopathology", "exp2lin_raw_signals_",  True),
]

# Dual-family (Safety Shell) warned failures, from the primary analysis.
# Used only to build the paired comparison table.
SHELL_WARNED = {
    ("FDA-radiography", "ViT_B16", 0),
    ("Linear-radiography", "ViT_B16", 2),
    ("Linear-histopathology", "DenseNet121", 2),
}


def perf_curve(run, uses_auroc):
    if uses_auroc:
        pm = np.asarray(run["prob_matrix"], float)
        y = np.asarray(run["panel_labels"]).ravel()
        return np.asarray([_auc_from_scores(y, pm[i, :]) for i in range(pm.shape[0])])
    return np.asarray(run["correct_matrix"], float).mean(axis=1)


def ac1_series(run):
    em = np.asarray(run["entropy_matrix"], float).mean(axis=1)
    return windowed_ews_indicators(em, window=5, detrend=True, detrend_sigma=2.0)["ac1"]


def load(data_dir):
    out = {}
    for name, prefix, uses_auroc in EXPERIMENTS:
        files = sorted(f for f in glob.glob(os.path.join(data_dir, "*.npz"))
                       if os.path.basename(f).startswith(prefix))
        runs, meta = [], []
        for arch in ARCHS:
            for f in files:
                b = os.path.basename(f)
                if arch in b:
                    with np.load(f, allow_pickle=True) as z:
                        runs.append({k: z[k] for k in z.files})
                    seed = int(b.split("seed")[-1].split(".")[0])
                    meta.append((arch, seed))
        out[name] = (runs, np.asarray(runs[0]["alphas"], float), uses_auroc, meta)
    return out


def pools_excluding(runs, held_out):
    other = [runs[j] for j in range(len(runs)) if j != held_out]
    p = {pk: [np.asarray(r[k], float).mean(axis=1) for r in other]
         for k, pk in POOL_KEY.items()}
    p["ac1"] = [ac1_series(r) for r in other]
    p["msp"] = [np.asarray(r["msp_matrix"], float).mean(axis=1) for r in other]
    p["setsize"] = [np.asarray(r["setsize_matrix"], float).mean(axis=1) for r in other]
    return p


def family_far(runs, alphas, per_detector_fpr, n_surrogates, seed=0):
    """Mean per-run false-alarm rate of the OR-combined representation family,
    using the study's own joint dependence-preserving surrogate procedure."""
    rates = []
    for held_out in range(len(runs)):
        pools = pools_excluding(runs, held_out)
        thr = {THR_KEY[pk]: calibrate_population_threshold(
                   pools[pk], persistence=PERSISTENCE,
                   nominal_fpr=per_detector_fpr, serial_dependence=False)
               for pk in POOL_KEY.values()}
        r = runs[held_out]
        signals = {"ac1": ac1_series(r),
                   "setsize": np.asarray(r["setsize_matrix"], float).mean(axis=1),
                   "msp": np.asarray(r["msp_matrix"], float).mean(axis=1),
                   "energy": np.asarray(r["energy_matrix"], float).mean(axis=1),
                   "mahalanobis": np.asarray(r["mahalanobis_matrix"], float).mean(axis=1),
                   "mc_dropout": np.asarray(r["mc_dropout_var_matrix"], float).mean(axis=1)}
        res = false_alarm_rate_analysis(
            alphas, signals, n_surrogates=n_surrogates, seed=seed,
            persistence=PERSISTENCE, nominal_fpr=per_detector_fpr,
            calibrated_thresholds=thr, signal_pools=pools)
        rates.append(res["SingleFamilyRepresentation"])
    return float(np.mean(rates))


def family_onset(runs, alphas, per_detector_fpr):
    """Earliest genuine onset across the family, LORO-calibrated, per run."""
    onsets = []
    for held_out in range(len(runs)):
        pools = pools_excluding(runs, held_out)
        first = None
        for k, pk in POOL_KEY.items():
            t = calibrate_population_threshold(pools[pk], persistence=PERSISTENCE,
                                               nominal_fpr=per_detector_fpr,
                                               serial_dependence=False)
            v = np.asarray(runs[held_out][k], float).mean(axis=1)
            res = apply_onset_threshold(alphas, v, t, persistence=PERSISTENCE,
                                        nominal_fpr=per_detector_fpr)
            a = res["alpha_ews"]
            if a is not None and (first is None or a < first):
                first = a
        onsets.append(first)
    return onsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="directory of the 36 .npz files")
    ap.add_argument("--mode", choices=["far-search", "coverage"], default="coverage")
    ap.add_argument("--fpr", type=float, default=0.01,
                    help="per-detector nominal rate (0.01 matches the Shell's 0.02)")
    ap.add_argument("--candidates", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--surrogates", type=int, default=400)
    args = ap.parse_args()

    data = load(args.data)

    if args.mode == "far-search":
        for c in [float(x) for x in args.candidates.split(",")]:
            per_exp = [family_far(r, a, c, args.surrogates)
                       for (r, a, _, _) in data.values()]
            print(f"per-detector fpr={c:<6} overall FAR={np.mean(per_exp):.4f}  "
                  f"per-experiment={[round(x, 3) for x in per_exp]}")
        return

    both = dual_only = single_only = neither = 0
    leads, confirmed_total = [], 0
    for name, (runs, alphas, uses_auroc, meta) in data.items():
        onsets = family_onset(runs, alphas, args.fpr)
        n_conf = n_cov = 0
        for i, run in enumerate(runs):
            fail = detect_persistent_failure(alphas, perf_curve(run, uses_auroc),
                                             TAU, PERSISTENCE)
            if not fail["sustained"]:
                continue
            n_conf += 1
            confirmed_total += 1
            a_star = alphas[fail["idx"]]
            o = onsets[i]
            single = o is not None and o < a_star
            dual = (name, meta[i][0], meta[i][1]) in SHELL_WARNED
            if single:
                n_cov += 1
                leads.append(a_star - o)
            both += single and dual
            dual_only += dual and not single
            single_only += single and not dual
            neither += (not single) and (not dual)
        print(f"{name:24s} confirmed={n_conf}  covered={n_cov}")

    lead = np.asarray(leads)
    print(f"\ncoverage {both + single_only}/{confirmed_total} = "
          f"{(both + single_only) / confirmed_total:.3f}   "
          f"median lead {np.median(lead):.3f}  mean {lead.mean():.3f}")
    print(f"paired table: both={both} dual-only={dual_only} "
          f"single-only={single_only} neither={neither}")
    if dual_only == 0:
        print("dual-family detections are a strict subset of single-family detections")
    try:
        from scipy.stats import binomtest
        n = dual_only + single_only
        if n:
            print(f"exact McNemar p={binomtest(min(dual_only, single_only), n, 0.5).pvalue:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
