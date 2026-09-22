# ============================================================
# Computes the two quantities still missing from Results:
#   (A) Detection AUROC per detector  -> Section 5.4
#   (B) Potentially intercepted failures -> Section 5.5
# Both from the SAVED npz files. No GPU, no retraining.
# Uses the CALIBRATED Shell onset (not the deprecated live-pipeline onset,
# whose blocked_fraction values were shown to be contaminated).
# ============================================================
import sys, os, glob, json, shutil
import numpy as np

CODE = None
for p in glob.glob("/kaggle/input/**/enhanced_ews_pipeline.py", recursive=True):
    if "_auc_from_scores" in open(p).read():
        CODE = p; break
assert CODE, "Fixed enhanced_ews_pipeline.py (with _auc_from_scores) not found."
shutil.copy(CODE, "/kaggle/working/"); sys.path.insert(0, "/kaggle/working")
from enhanced_ews_pipeline import (calibrate_population_threshold, apply_onset_threshold,
                                   detect_persistent_failure, _auc_from_scores)
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None
print("using:", CODE)

EXPERIMENTS = [
    ("Linear-radiography",    "linshift_raw_signals_", False),
    ("Linear-histopathology", "exp2lin_raw_signals_",  True),
    ("FDA-radiography",       "raw_signals_",          False),
    ("FDA-histopathology",    "exp2_raw_signals_",     True),
]
ARCHS = ["ResNet18", "DenseNet121", "ViT_B16"]
DETECTORS = [("Mahalanobis","mahalanobis_matrix",False), ("Energy","energy_matrix",False),
             ("MCDropout","mc_dropout_var_matrix",False), ("MSP","msp_matrix",True)]
TAU, PERSIST, NOMINAL = 0.70, 2, 0.05

ALL = glob.glob("/kaggle/input/**/*.npz", recursive=True)
print(f"{len(ALL)} npz files found (expect 36)\n")
out_all = {}

for name, prefix, is_auc in EXPERIMENTS:
    files = [f for f in ALL if os.path.basename(f).startswith(prefix)]
    runs = {a: [] for a in ARCHS}
    for f in sorted(files):
        arch = next((a for a in ARCHS if a in os.path.basename(f)), None)
        if arch:
            z = np.load(f, allow_pickle=True)
            runs[arch].append({k: z[k] for k in z.files})
    n = sum(len(v) for v in runs.values())
    print(f"=== {name} ({n} runs, metric_is_auc={is_auc}) ===")
    if n != 9:
        print("   !! expected 9 runs -- SKIPPING\n"); continue
    flat = [r for a in ARCHS for r in runs[a]]
    alphas = np.asarray(flat[0]["alphas"])

    # ---------- (A) DETECTION AUROC: alpha=0 scores vs each shifted level ----------
    print("  (A) Detection AUROC (alpha=0 vs shifted levels), pooled over runs:")
    auroc_summary = {}
    for dname, key, invert in DETECTORS:
        per_level = []
        for r in flat:
            m = np.asarray(r[key], dtype=float)
            if invert: m = -m
            base = m[0, :]
            for a in range(1, m.shape[0]):
                if roc_auc_score is None: continue
                y = np.concatenate([np.zeros(len(base)), np.ones(m.shape[1])])
                s = np.concatenate([base, m[a, :]])
                try: per_level.append(float(roc_auc_score(y, s)))
                except ValueError: pass
        if per_level:
            auroc_summary[dname] = {"min": float(np.min(per_level)),
                                    "max": float(np.max(per_level)),
                                    "mean": float(np.mean(per_level)),
                                    "median": float(np.median(per_level))}
            print(f"      {dname:12}: range {np.min(per_level):.3f}-{np.max(per_level):.3f}, "
                  f"mean {np.mean(per_level):.3f}")

    # ---------- calibrated Shell onset (representation family), per run ----------
    def perf_curve(r):
        if is_auc:
            pm = np.asarray(r["prob_matrix"], dtype=float)
            y = np.asarray(r["panel_labels"]).ravel()
            return np.asarray([_auc_from_scores(y, pm[i, :]) for i in range(pm.shape[0])])
        return np.asarray(r["correct_matrix"], dtype=float).mean(axis=1)

    thr = {}
    for dname, key, invert in DETECTORS[:3]:      # representation family only
        pool = []
        for r in flat:
            s = np.asarray(r[key], dtype=float).mean(axis=1)
            pool.append(-s if invert else s)
        thr[dname] = calibrate_population_threshold(pool, nominal_fpr=NOMINAL, persistence=PERSIST)

    # ---------- (B) POTENTIALLY INTERCEPTED FAILURES ----------
    print("  (B) Potentially intercepted failures (calibrated Shell onset):")
    inter = {}
    for arch in ARCHS:
        tot_fail = tot_after = 0
        for r in runs[arch]:
            perf = perf_curve(r)
            fail = detect_persistent_failure(alphas, perf, TAU, PERSIST)
            if not fail["sustained"]:
                continue                      # censored: no failure occurred, nothing to intercept
            onsets = []
            for dname, key, invert in DETECTORS[:3]:
                s = np.asarray(r[key], dtype=float).mean(axis=1)
                if invert: s = -s
                o = apply_onset_threshold(alphas, s, thr[dname], persistence=PERSIST)
                if o["onset_idx"] is not None: onsets.append(o["onset_idx"])
            cm = np.asarray(r["correct_matrix"], dtype=float)
            astar_idx = int(np.argmin(np.abs(alphas - fail["alpha_star"])))
            wrong = (cm[astar_idx:, :] == 0).sum()
            # A failure with NO detector ever firing must still be counted in the
            # denominator, contributing zero interceptions -- it is the worst case
            # for this metric, not a case to discard. The previous version used
            # `continue` here, silently dropping such runs from BOTH numerator and
            # denominator, which inflated the fraction by excluding exactly the
            # failures that were least protected. Verified against a real
            # discrepancy: an architecture with one unwarned, no-onset run showed
            # interception 1.000 despite Safety-Shell coverage of only 0.667 on the
            # same runs, because that run's failure was silently excluded here.
            tot_fail += wrong
            if onsets and min(onsets) < astar_idx:
                tot_after += wrong            # every post-alpha* error follows the alert
        frac = (tot_after / tot_fail) if tot_fail else float("nan")
        inter[arch] = {"n_incorrect_at_or_after_failure": int(tot_fail),
                       "n_after_alert": int(tot_after),
                       "potentially_intercepted_fraction": frac}
        print(f"      {arch:12}: {frac if np.isnan(frac) else round(frac,3)} "
              f"({tot_after}/{tot_fail} incorrect predictions followed an alert)")

    out_all[name] = {"detection_auroc": auroc_summary,
                     "potentially_intercepted": inter,
                     "calibrated_thresholds": thr}
    print()

with open("/kaggle/working/auroc_and_interception.json", "w") as fh:
    json.dump(out_all, fh, indent=2, default=str)
print(">>> saved /kaggle/working/auroc_and_interception.json <<<")
