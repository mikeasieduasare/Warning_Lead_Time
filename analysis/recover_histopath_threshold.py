# ============================================================
# Reverse-engineers the exact probability threshold used to classify
# histopathology predictions as correct/incorrect (PI item B19), by testing
# candidate thresholds against the saved prob_matrix and panel_labels and
# checking which one exactly reproduces the saved correct_matrix.
#
# This is more reliable than trusting an old script's comment: it validates
# directly against what the saved data shows was actually done.
#
# No GPU needed -- reads the already-saved npz files for the two
# histopathology experiments only (FDA-histopathology, Linear-histopathology).
# ============================================================
import numpy as np, glob, os

EXPERIMENTS = [
    ("FDA-histopathology",    "exp2_raw_signals_"),
    ("Linear-histopathology", "exp2lin_raw_signals_"),
]
ARCHS = ["ResNet18", "DenseNet121", "ViT_B16"]
ALL_NPZ = glob.glob("/kaggle/input/**/*.npz", recursive=True)

# Candidate thresholds to test -- 0.5 is the standard default; a wider sweep
# is included in case a non-standard operating point was used.
CANDIDATES = [0.5] + [round(x, 2) for x in np.arange(0.05, 1.00, 0.05) if x != 0.5]

for name, prefix in EXPERIMENTS:
    files = [f for f in ALL_NPZ if os.path.basename(f).startswith(prefix)]
    print(f"=== {name} ({len(files)} files) ===")
    if not files:
        print("  !! no files found -- check prefix/path\n"); continue

    all_correct_matches = {t: True for t in CANDIDATES}
    total_checked = 0

    for f in sorted(files):
        z = np.load(f, allow_pickle=True)
        if "prob_matrix" not in z.files or "correct_matrix" not in z.files or "panel_labels" not in z.files:
            print(f"  !! {os.path.basename(f)} missing required keys -- skipping")
            continue
        prob = np.asarray(z["prob_matrix"], dtype=float)      # (n_alpha, n_panel)
        correct = np.asarray(z["correct_matrix"], dtype=float) # (n_alpha, n_panel), 1.0/0.0
        labels = np.asarray(z["panel_labels"]).astype(int)     # (n_panel,) or (n_alpha, n_panel)
        if labels.ndim == 1:
            labels = np.tile(labels, (prob.shape[0], 1))

        total_checked += prob.size
        for t in CANDIDATES:
            pred = (prob > t).astype(int)
            reconstructed_correct = (pred == labels).astype(float)
            if not np.allclose(reconstructed_correct, correct, atol=1e-6):
                all_correct_matches[t] = False

    matching = [t for t, ok in all_correct_matches.items() if ok]
    print(f"  Checked {total_checked} (image, shift-level) predictions across {len(files)} runs.")
    if matching:
        print(f"  Threshold(s) that EXACTLY reproduce the saved correct_matrix: {matching}")
    else:
        print(f"  !! No candidate threshold in {CANDIDATES} exactly reproduces correct_matrix.")
        print(f"     This may mean: (a) a non-0.05-grid threshold was used, (b) correct_matrix")
        print(f"     was derived from argmax over a full softmax vector rather than a single")
        print(f"     probability threshold, or (c) prob_matrix stores a different quantity than")
        print(f"     assumed (e.g. logit rather than probability). Re-run with a finer grid or")
        print(f"     inspect prob_matrix's value range directly if this occurs.")
    print()
