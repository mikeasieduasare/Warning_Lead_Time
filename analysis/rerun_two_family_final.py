# ============================================================
# FINAL RERUN: two-family Safety Shell fix + LORO-consistent false-alarm test.
#
# Fixes applied since the last rerun:
#   1. Primary Delta-alpha/coverage now REQUIRES a genuine onset from BOTH
#      the dynamical family (AC1) and the representation-based family
#      (Mahalanobis/Energy/MCDropout), matching Methods 4.9. An earlier
#      version silently used the representation family alone.
#   2. AC1 is now LORO-calibrated (serial_dependence=True), consistent with
#      the other three detectors that feed the primary result.
#   3. The false-alarm test (Table 2, Safety Shell row) now uses each run's
#      OWN LORO thresholds and a surrogate pool excluding that run's own
#      data -- consistent with the primary result -- instead of a single
#      pooled threshold. Reported as a per-run distribution (min/mean/max),
#      not a single number, since 9 genuinely different thresholds no longer
#      have one false-alarm rate.
#   4. A string-matching bug that made the Shell's false-alarm rate exactly
#      0.000 by mathematical construction (regardless of data) is fixed.
#
# No GPU needed -- reads the already-saved npz files.
# ============================================================
import sys, os, glob, json, shutil
import numpy as np

CODE = None
for p in glob.glob("/kaggle/input/**/enhanced_ews_pipeline.py", recursive=True):
    src = open(p).read()
    if all(s in src for s in ["calibrated_threshold_crossing\") == \"calibrated_threshold_crossing",
                              "AC1's own genuine calibrated crossing"]) or \
       ("compute_safety_shell_onset_stratified(alphas, ac1_onset, calibration_independent_onsets)" in src):
        CODE = p; break
assert CODE, "Pipeline with the two-family fix not found -- upload the latest version."
shutil.copy(CODE, "/kaggle/working/enhanced_ews_pipeline.py")
sys.path.insert(0, "/kaggle/working")
print("using:", CODE)
from enhanced_ews_pipeline import (analyze_experiment_posthoc, false_alarm_rate_analysis,
                                   calibrate_leave_one_out_thresholds, calibrate_population_threshold,
                                   windowed_ews_indicators)

EXPERIMENTS = [
    ("Linear-radiography",    "linshift_raw_signals_", False),
    ("Linear-histopathology", "exp2lin_raw_signals_",  True),
    ("FDA-radiography",       "raw_signals_",          False),
    ("FDA-histopathology",    "exp2_raw_signals_",     True),
]
ARCHS = ["ResNet18", "DenseNet121", "ViT_B16"]
ALL_NPZ = glob.glob("/kaggle/input/**/*.npz", recursive=True)
print(f"{len(ALL_NPZ)} npz files found (expect 36)\n")

OUT = "/kaggle/working/two_family_final"
os.makedirs(OUT, exist_ok=True)

for name, prefix, is_auc in EXPERIMENTS:
    files = [f for f in ALL_NPZ if os.path.basename(f).startswith(prefix)]
    raw = {a: [] for a in ARCHS}
    for f in sorted(files):
        arch = next((a for a in ARCHS if a in os.path.basename(f)), None)
        if arch:
            z = np.load(f, allow_pickle=True)
            raw[arch].append({k: z[k] for k in z.files})
    counts = {a: len(raw[a]) for a in ARCHS}
    print(f"=== {name} ({sum(counts.values())} runs) ===")
    if sum(counts.values()) != 9:
        print("   !! expected 9 -- SKIPPING\n"); continue

    # --- primary result: now uses the corrected two-family Shell internally ---
    res = analyze_experiment_posthoc(raw, threshold=0.70, n_boot=2000, metric_is_auc=is_auc)
    print("  PRIMARY RESULT (two-family corrected):")
    for arch, d in res["by_arch"].items():
        da = d["delta_alpha"]
        print(f"    {arch:12}: Delta={da['delta_alpha_point']} CI={da['delta_alpha_ci95']} "
              f"coverage={da['warning_coverage']} status={da['status_counts']}")

    # --- LORO-consistent false-alarm test: per-run thresholds, excluded-pool surrogates ---
    all_runs = [r for a in ARCHS for r in raw[a]]
    for r in all_runs:
        em = np.asarray(r["entropy_matrix"], dtype=float)
        r["ac1_detrended"] = windowed_ews_indicators(em.mean(axis=1))["ac1"]
    alphas = all_runs[0]["alphas"]
    detector_specs = [("Mahalanobis","mahalanobis_matrix",False),("Energy","energy_matrix",False),
                      ("MCDropout","mc_dropout_var_matrix",False),("AC1","ac1_detrended",False,True)]
    loro = calibrate_leave_one_out_thresholds(all_runs, detector_specs, nominal_fpr=0.05)
    msp_thr = calibrate_population_threshold(
        [-np.asarray(r["msp_matrix"],dtype=float).mean(axis=1) for r in all_runs], nominal_fpr=0.05)

    print("  LORO-CONSISTENT FALSE-ALARM RATES (per run, own threshold, excluded-pool surrogate):")
    print("  Reports BOTH design variants for direct comparison:")
    print("    - SingleFamilyRepresentation: alert if ANY of Mahalanobis/Energy/MCDropout fires")
    print("    - SafetyShell: alert only if BOTH that family AND AC1 independently fire")
    shell_rates, maha_rates, ac1_rates, single_family_rates = [], [], [], []
    for i, run in enumerate(all_runs):
        pool_excl_i = {
            "mahalanobis": [all_runs[j]["mahalanobis_matrix"].mean(axis=1) for j in range(9) if j!=i],
            "energy": [all_runs[j]["energy_matrix"].mean(axis=1) for j in range(9) if j!=i],
            "mc_dropout": [all_runs[j]["mc_dropout_var_matrix"].mean(axis=1) for j in range(9) if j!=i],
            "ac1": [all_runs[j]["ac1_detrended"] for j in range(9) if j!=i],
            "msp": [-all_runs[j]["msp_matrix"].mean(axis=1) for j in range(9)],
            "setsize": [all_runs[j]["setsize_matrix"].mean(axis=1) for j in range(9)] if "setsize_matrix" in run else [np.zeros(len(alphas))]*9,
        }
        thr_i = {**loro[i], "MSP": msp_thr}
        far = false_alarm_rate_analysis(alphas,
            {"ac1": run["ac1_detrended"],
             "setsize": run["setsize_matrix"].mean(axis=1) if "setsize_matrix" in run else np.zeros(len(alphas)),
             "msp": -run["msp_matrix"].mean(axis=1), "energy": run["energy_matrix"].mean(axis=1),
             "mahalanobis": run["mahalanobis_matrix"].mean(axis=1), "mc_dropout": run["mc_dropout_var_matrix"].mean(axis=1)},
            n_surrogates=400, seed=i, calibrated_thresholds=thr_i, signal_pools=pool_excl_i)
        shell_rates.append(far["SafetyShell"]); maha_rates.append(far["Mahalanobis"]); ac1_rates.append(far["AC1"])
        single_family_rates.append(far["SingleFamilyRepresentation"])
        print(f"    run {i}: SingleFamily={far['SingleFamilyRepresentation']:.3f}  "
              f"DualFamily(Shell)={far['SafetyShell']:.3f}  (Maha={far['Mahalanobis']:.3f} AC1={far['AC1']:.3f})")

    print(f"  SingleFamily FAR across 9 runs: min={min(single_family_rates):.3f} mean={np.mean(single_family_rates):.3f} max={max(single_family_rates):.3f}")
    print(f"  DualFamily(Shell) FAR across 9 runs: min={min(shell_rates):.3f} mean={np.mean(shell_rates):.3f} max={max(shell_rates):.3f}")
    print(f"  AC1  FAR across 9 runs: min={min(ac1_rates):.3f} mean={np.mean(ac1_rates):.3f} max={max(ac1_rates):.3f}")

    with open(os.path.join(OUT, f"{prefix.rstrip('_')}_TWOFAMILY.json"), "w") as fh:
        json.dump({"primary": res, "shell_far_per_run": shell_rates,
                  "single_family_far_per_run": single_family_rates,
                  "maha_far_per_run": maha_rates, "ac1_far_per_run": ac1_rates},
                 fh, indent=2, default=str)
    print()

shutil.make_archive("/kaggle/working/two_family_final", "zip", OUT)
print(">>> Download /kaggle/working/two_family_final.zip <<<")
