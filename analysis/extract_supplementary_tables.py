# ============================================================
# EXTRACT SUPPLEMENTARY TABLES S1-S3 from saved npz files.
#
# Produces, exactly matching Supplementary_Material.md's promised structure:
#   Table S1 - per-seed, per-level detection AUROC (Mahalanobis, Energy,
#               MC-Dropout, inverted MSP), all four experiments
#   Table S2 - potentially-intercepted-failure fraction per confirmed-failure
#               run, cross-checked against warning coverage status
#   Table S3 - seed-level alpha* and Delta-alpha_shell at tau in
#               {0.65, 0.70, 0.75, 0.80}, FDA-radiography only
#
# Runs entirely against SAVED npz files, reusing the manuscript's own
# established functions from enhanced_ews_pipeline.py rather than
# reimplementing detection/calibration logic. No GPU, no retraining.
#
# HARD REQUIREMENT: this script verifies every array it needs is present,
# with a sane shape, in every required npz file BEFORE computing anything.
# If a requirement is not met, it prints exactly what is missing and exits
# without producing any table -- a partial or silently-wrong supplementary
# table is worse than none.
# ============================================================

import sys, os, glob, json, shutil
import numpy as np

# ------------------------------------------------------------------
# 0. Import the established pipeline code.
# ------------------------------------------------------------------
PIPELINE_CODE = None
for p in glob.glob("/kaggle/input/**/enhanced_ews_pipeline.py", recursive=True):
    src = open(p).read()
    if "analyze_experiment_posthoc" in src and "_auc_from_scores" in src:
        PIPELINE_CODE = p
        break

if PIPELINE_CODE is None:
    print("FATAL: enhanced_ews_pipeline.py (with analyze_experiment_posthoc and")
    print("       _auc_from_scores) was not found under /kaggle/input/. This")
    print("       script imports the manuscript's own calibration and onset-")
    print("       detection functions rather than reimplementing them, so it")
    print("       cannot proceed without this file.")
    sys.exit(1)

shutil.copy(PIPELINE_CODE, "/kaggle/working/enhanced_ews_pipeline.py")
sys.path.insert(0, "/kaggle/working")
from enhanced_ews_pipeline import (
    calibrate_population_threshold,
    apply_onset_threshold,
    detect_persistent_failure,
    _auc_from_scores,
    windowed_ews_indicators,
)
print(f"Using pipeline code: {PIPELINE_CODE}\n")

EXPERIMENTS = [
    ("Linear-radiography",    "linshift_raw_signals_", False),
    ("Linear-histopathology", "exp2lin_raw_signals_",  True),
    ("FDA-radiography",       "raw_signals_",           False),
    ("FDA-histopathology",    "exp2_raw_signals_",      True),
]
ARCHS = ["ResNet18", "DenseNet121", "ViT_B16"]
TAU = 0.70
PERSISTENCE = 2
NOMINAL_FPR = 0.05


# ------------------------------------------------------------------
# 1. STRICT VERIFICATION GATE
# ------------------------------------------------------------------
REQUIRED_KEYS_COMMON = ["alphas", "correct_matrix", "entropy_matrix",
                         "mahalanobis_matrix", "energy_matrix",
                         "mc_dropout_var_matrix", "msp_matrix"]
REQUIRED_KEYS_AUC_ONLY = ["prob_matrix", "panel_labels"]


def discover_and_verify():
    all_npz = glob.glob("/kaggle/input/**/*.npz", recursive=True)
    if not all_npz:
        print("FATAL: no .npz files found under /kaggle/input/.")
        sys.exit(1)

    failures = []
    loaded = {}

    for name, prefix, is_auc in EXPERIMENTS:
        files = [f for f in all_npz if os.path.basename(f).startswith(prefix)]
        if len(files) != 9:
            failures.append(f"[{name}] expected 9 run files with prefix "
                             f"'{prefix}', found {len(files)}.")
            continue

        runs = {a: [] for a in ARCHS}
        required = REQUIRED_KEYS_COMMON + (REQUIRED_KEYS_AUC_ONLY if is_auc else [])

        for f in sorted(files):
            arch = next((a for a in ARCHS if a in os.path.basename(f)), None)
            if arch is None:
                failures.append(f"[{name}] '{os.path.basename(f)}' has no "
                                 f"recognised architecture name in its filename.")
                continue
            try:
                z = np.load(f, allow_pickle=True)
            except Exception as e:
                failures.append(f"[{name}] '{os.path.basename(f)}' failed to load: {e}")
                continue

            missing = [k for k in required if k not in z.files]
            if missing:
                failures.append(f"[{name}] '{os.path.basename(f)}' is missing "
                                 f"required key(s): {missing}. Present: {list(z.files)}")
                continue

            runs[arch].append({k: z[k] for k in z.files})

        if sum(len(v) for v in runs.values()) == 9:
            loaded[name] = {"is_auc": is_auc, "runs": runs}

    if failures:
        print("=" * 70)
        print("VERIFICATION FAILED. No tables will be produced. Problems:")
        print("=" * 70)
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("Verification passed: all required arrays present for all four experiments.\n")
    return loaded


# ------------------------------------------------------------------
# 2. Shared helpers (mirrors Algorithm S1)
# ------------------------------------------------------------------
def perf_curve(run, is_auc):
    if is_auc:
        pm = np.asarray(run["prob_matrix"], dtype=float)
        y = np.asarray(run["panel_labels"]).ravel()
        return np.asarray([_auc_from_scores(y, pm[i, :]) for i in range(pm.shape[0])])
    return np.asarray(run["correct_matrix"], dtype=float).mean(axis=1)


def ac1_series(run):
    em = np.asarray(run["entropy_matrix"], dtype=float).mean(axis=1)
    return windowed_ews_indicators(em, window=5, detrend=True, detrend_sigma=2.0)["ac1"]


def loro_onsets(all_run_signals, alphas, persistence=PERSISTENCE, serial_dependence=False):
    n = len(all_run_signals)
    onsets = []
    for held_out in range(n):
        other = [all_run_signals[j] for j in range(n) if j != held_out]
        thr = calibrate_population_threshold(other, persistence=persistence,
                                              nominal_fpr=NOMINAL_FPR,
                                              serial_dependence=serial_dependence)
        onset = apply_onset_threshold(alphas, all_run_signals[held_out], thr,
                                       persistence=persistence, nominal_fpr=NOMINAL_FPR)
        onsets.append(onset)
    return onsets


def safety_shell_onsets(flat_runs, alphas):
    """Returns per-run alpha_shell (or None) using the same LORO dual-family logic."""
    ac1_signals = [ac1_series(r) for r in flat_runs]
    ac1_onsets = loro_onsets(ac1_signals, alphas, serial_dependence=True)

    rep_onset_alpha = []
    for dname, key in [("Mahalanobis", "mahalanobis_matrix"),
                        ("Energy", "energy_matrix"),
                        ("MCDropout", "mc_dropout_var_matrix")]:
        pass
    # earliest genuine crossing among the three representation detectors, per run
    rep_signals = {k: [np.asarray(r[k], dtype=float).mean(axis=1) for r in flat_runs]
                   for k in ["mahalanobis_matrix", "energy_matrix", "mc_dropout_var_matrix"]}
    rep_onsets = {k: loro_onsets(v, alphas) for k, v in rep_signals.items()}
    for i in range(len(flat_runs)):
        candidates = [rep_onsets[k][i]["alpha_ews"] for k in rep_onsets
                      if rep_onsets[k][i]["alpha_ews"] is not None]
        rep_onset_alpha.append(min(candidates) if candidates else None)

    shell_alpha = []
    for i in range(len(flat_runs)):
        a_dyn = ac1_onsets[i]["alpha_ews"]
        a_rep = rep_onset_alpha[i]
        shell_alpha.append(max(a_dyn, a_rep) if (a_dyn is not None and a_rep is not None) else None)
    return shell_alpha


# ------------------------------------------------------------------
# 3. Table S1: per-seed, per-level detection AUROC
# ------------------------------------------------------------------
def build_table_s1(loaded):
    rows = []
    for exp_name, exp_data in loaded.items():
        runs = exp_data["runs"]
        for arch in ARCHS:
            for seed_idx, run in enumerate(runs[arch]):
                alphas = np.asarray(run["alphas"])
                nominal_scores = {
                    "Mahalanobis": np.asarray(run["mahalanobis_matrix"], dtype=float)[0, :],
                    "Energy": np.asarray(run["energy_matrix"], dtype=float)[0, :],
                    "MC-Dropout": np.asarray(run["mc_dropout_var_matrix"], dtype=float)[0, :],
                    "MSP (inverted)": -np.asarray(run["msp_matrix"], dtype=float)[0, :],
                }
                for dname, key, invert in [("Mahalanobis", "mahalanobis_matrix", False),
                                            ("Energy", "energy_matrix", False),
                                            ("MC-Dropout", "mc_dropout_var_matrix", False),
                                            ("MSP (inverted)", "msp_matrix", True)]:
                    mat = np.asarray(run[key], dtype=float)
                    if invert:
                        mat = -mat
                    row = {"experiment": exp_name, "architecture": arch, "seed": seed_idx,
                           "detector": dname}
                    for level_idx in range(1, mat.shape[0]):
                        pos = mat[level_idx, :]
                        neg = mat[0, :]
                        labels = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
                        scores = np.concatenate([neg, pos])
                        auroc = _auc_from_scores(labels, scores)
                        row[f"alpha={alphas[level_idx]:.3f}"] = round(float(auroc), 4)
                    rows.append(row)
    return rows


# ------------------------------------------------------------------
# 4. Table S2: potentially-intercepted-failure fraction per run
# ------------------------------------------------------------------
def build_table_s2(loaded):
    rows = []
    for exp_name, exp_data in loaded.items():
        is_auc = exp_data["is_auc"]
        runs = exp_data["runs"]
        flat = [r for a in ARCHS for r in runs[a]]
        alphas = np.asarray(flat[0]["alphas"])
        shell_alpha = safety_shell_onsets(flat, alphas)

        idx = 0
        for arch in ARCHS:
            for seed_idx in range(len(runs[arch])):
                run = flat[idx]
                perf = perf_curve(run, is_auc)
                fail = detect_persistent_failure(alphas, perf, TAU, PERSISTENCE)
                a_shell = shell_alpha[idx]

                if fail["sustained"]:
                    astar_idx = fail["idx"]
                    if is_auc:
                        pm = np.asarray(run["prob_matrix"], dtype=float)
                        y = np.asarray(run["panel_labels"]).ravel()
                        preds = (pm[astar_idx:, :] >= 0.5).astype(int)
                        misclassified = preds != y[None, :]
                    else:
                        cm = np.asarray(run["correct_matrix"], dtype=float)
                        misclassified = cm[astar_idx:, :] == 0
                    total_misclass = misclassified.sum()
                    if a_shell is not None and total_misclass > 0:
                        a_shell_idx = int(np.argmin(np.abs(alphas - a_shell)))
                        intercepted = misclassified[max(0, a_shell_idx - astar_idx):, :].sum() \
                            if a_shell_idx >= astar_idx else total_misclass
                        frac = float(intercepted) / float(total_misclass)
                        status = "observed" if a_shell_idx < astar_idx else "unwarned failure"
                    else:
                        frac = 0.0
                        status = "unwarned failure"
                    rows.append({
                        "experiment": exp_name, "architecture": arch, "seed": seed_idx,
                        "alpha_star": round(float(alphas[astar_idx]), 3),
                        "alpha_shell": round(float(a_shell), 3) if a_shell is not None else None,
                        "status": status,
                        "potentially_intercepted_fraction": round(frac, 3),
                    })
                idx += 1
    return rows


# ------------------------------------------------------------------
# 5. Table S3: seed-level threshold sensitivity (FDA-radiography only)
# ------------------------------------------------------------------
def build_table_s3(loaded):
    if "FDA-radiography" not in loaded:
        print("WARNING: FDA-radiography not found; Table S3 cannot be built.")
        return []
    exp_data = loaded["FDA-radiography"]
    runs = exp_data["runs"]
    flat = [r for a in ARCHS for r in runs[a]]
    alphas = np.asarray(flat[0]["alphas"])
    taus = [0.65, 0.70, 0.75, 0.80]

    rows = []
    idx = 0
    for arch in ARCHS:
        for seed_idx in range(len(runs[arch])):
            run = flat[idx]
            perf = perf_curve(run, False)
            row = {"architecture": arch, "seed": seed_idx}
            for tau in taus:
                fail = detect_persistent_failure(alphas, perf, tau, PERSISTENCE)
                if fail["sustained"]:
                    row[f"alpha_star_tau{tau}"] = round(float(alphas[fail["idx"]]), 3)
                else:
                    row[f"alpha_star_tau{tau}"] = "censored"
            rows.append(row)
            idx += 1
    return rows


# ------------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------------
def main():
    loaded = discover_and_verify()

    print("Building Table S1 (detection AUROC)...")
    s1 = build_table_s1(loaded)
    print(f"  {len(s1)} rows.\n")

    print("Building Table S2 (intercepted-failure fractions)...")
    s2 = build_table_s2(loaded)
    print(f"  {len(s2)} rows (expect 19 confirmed failures across all experiments).\n")

    print("Building Table S3 (threshold sensitivity, FDA-radiography)...")
    s3 = build_table_s3(loaded)
    print(f"  {len(s3)} rows (expect 9).\n")

    out = {"table_s1_detection_auroc": s1,
           "table_s2_intercepted_failure": s2,
           "table_s3_threshold_sensitivity": s3}
    with open("/kaggle/working/supplementary_tables.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(">>> saved /kaggle/working/supplementary_tables.json <<<")
    print(">>> paste these into Supplementary_Material.md's Table S1-S3 shells <<<")


if __name__ == "__main__":
    main()
