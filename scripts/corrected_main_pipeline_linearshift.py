# =================================================================
# PARALLEL SECONDARY EXPERIMENT: PIXEL-SPACE LINEAR BLEND SHIFT
#
# Identical framework to corrected_main_pipeline.py (same architectures,
# same Safety Shell design, same statistical machinery) but with the shift
# MECHANISM replaced: x_alpha = (1-alpha)*x_source + alpha*x_target
# (direct pixel-space interpolation) instead of FDA low-frequency
# amplitude blending. Run ALONGSIDE, not instead of, the FDA experiment --
# see linear_pixel_blend_shift's docstring in cross_domain_shift.py for
# the full rationale (testing whether a more literal, monotonic shift
# mechanism produces cleaner classic CSD precursors than FDA did, and
# whether Warning Lead Time conclusions generalize across shift
# mechanisms, not just datasets/tasks).
#
# All outputs use a "linshift_" filename prefix so they can coexist in the
# same output directory as the FDA experiment's results without collision.
#
# Everything else in this file (Safety Shell family-stratified voting,
# detrended AC1, multi-seed pooling, threshold sensitivity, MC Dropout,
# the two summary figures) is UNCHANGED from corrected_main_pipeline.py --
# see that file's docstring and the enhanced_ews_pipeline.py module for
# the full rationale behind each of these design choices.
#
# VERIFICATION BOUNDARY: identical to corrected_main_pipeline.py -- the
# shift mechanism itself (linear_pixel_blend_shift) was verified with a
# numpy-equivalent test (exact midpoint match, perfectly proportional
# deviation at every alpha). The torch/GPU-dependent training pipeline has
# not been end-to-end run in this sandbox (no GPU/torch available here).
#
# REQUIRES: pydicom (pip install pydicom -q) and the Kaggle competition
# dataset "rsna-pneumonia-detection-challenge" added as a notebook input.
# =================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

import medmnist
from medmnist import INFO

from enhanced_ews_pipeline import (
    analyze_experiment_posthoc,
    build_fixed_panel_indices,
    windowed_ews_indicators,
    calculate_transition_metrics,
    composite_ews_agreement,
    detect_sustained_onset,
    detect_fixed_threshold_onset,
    mc_dropout_predict,
    baseline_onset_detection,
    compare_lead_times,
    detection_auroc_vs_alpha,
    register_feature_hook,
    calibrate_mahalanobis,
    mahalanobis_scores,
    permutation_test_lead_time_diff,
    compute_safety_shell_onset_stratified,
    compute_blocked_silent_counts,
    paired_bootstrap_blocked_fraction_test,
    threshold_sensitivity_analysis,
    false_alarm_rate_analysis,
    pool_across_seeds,
    seed_variance_summary,
)
from cross_domain_shift import RSNATargetPool, assign_fixed_partners, linear_pixel_blend_shift
from publication_utils import (
    set_publication_style, export_table,
    plot_accuracy_by_seed, plot_detrended_ac1,
    plot_threshold_sensitivity, plot_detector_comparison,
    plot_pairwise_significance_heatmap, plot_summary_figure_1_reliability,
    plot_summary_figure_2_detector_effectiveness,
    save_raw_signals_npz,
)

# =================================================================
# SETTINGS
# =================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
ALPHA_STEPS = np.linspace(0, 1.0, 20)
PANEL_SIZE = 400
N_BOOT = 500
N_BOOT_THRESHOLD = 100          # per-seed, per-threshold bootstrap (cheaper; pooled across seeds after)
FAILURE_THRESHOLD = 0.70        # primary/headline threshold
THRESHOLD_GRID = (0.65, 0.70, 0.75, 0.80)
SEEDS = (0, 1, 2)
SEED = 42                       # controls panel/partner selection (shared across all seeds/architectures)
RSNA_ROOT = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
RSNA_POOL_SIZE = 500
MC_DROPOUT_SAMPLES = 10         # smoke-test lever: see corrected_main_pipeline.py for rationale
TRAIN_EPOCHS = 10
TRAIN_LR = 1e-4
MIN_ACCEPTABLE_VAL_ACC = 0.75
MAX_TRAIN_SAMPLES = None        # smoke-test only: cap the training set size. None = full dataset (real run).
MAX_VAL_SAMPLES = None          # smoke-test only: cap the validation set size. None = full dataset (real run).
ARCHITECTURES = ["ResNet18", "DenseNet121", "ViT_B16"]


# =================================================================
# DATA
# =================================================================
def get_data():
    """PneumoniaMNIST source domain (D_s)."""
    data_flag = "pneumoniamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    train_dataset = DataClass(split="train", transform=transform, download=True)
    val_dataset = DataClass(split="val", transform=transform, download=True)
    test_dataset = DataClass(split="test", transform=transform, download=True)
    num_classes = len(info["label"])
    return train_dataset, val_dataset, test_dataset, num_classes


def get_denormalize_renormalize():
    """Converts between model-normalized space (mean.5/std.5) and [0,1]
    pixel space, since the FDA shift and RSNA pool both operate in [0,1]."""
    def to_unit(x):
        return torch.clamp(x * 0.5 + 0.5, 0, 1)

    def to_model_space(x):
        return (x - 0.5) / 0.5

    return to_unit, to_model_space


def build_target_partners(panel_size, device, root=RSNA_ROOT,
                           pool_size=RSNA_POOL_SIZE, seed=SEED):
    """Fixed RSNA target pool + fixed partner assignment, reused across
    EVERY architecture and seed for a fair, controlled comparison."""
    pool = RSNATargetPool(root=root, pool_size=pool_size, seed=seed)
    partner_idx = assign_fixed_partners(panel_size, len(pool), seed=seed)
    partners = torch.stack([pool.get(i) for i in partner_idx])
    return partners.to(device)


def apply_clinical_shift(images, target_partners, alpha, to_unit, to_model_space):
    """Pixel-space linear blend shift -- see linear_pixel_blend_shift docstring
    in cross_domain_shift.py for the full rationale vs. the FDA experiment."""
    unit_images = to_unit(images)
    shifted = linear_pixel_blend_shift(unit_images, target_partners, alpha)
    return to_model_space(shifted)


# =================================================================
# ARCHITECTURES
# =================================================================
def initialize_single_model(arch_name, num_classes):
    """Builds ONE architecture by name. Called once per (architecture, seed)
    so the random head-init and any seed-dependent layer behavior differs
    across seeds, as intended for the multi-seed variance analysis."""
    from torchvision import models as tvm

    # A Dropout layer is inserted before the final classifier for EVERY
    # architecture, uniformly -- this is what makes MC Dropout (Gal &
    # Ghahramani, 2016) usable and COMPARABLE across all three, since
    # torchvision's ResNet18/DenseNet121 have no dropout by default
    # (unlike ViT-B16, which already has internal dropout). Without this,
    # MC Dropout would only be available for one of the three
    # architectures, which was the exact reason it was scoped out
    # earlier. register_feature_hook still works unchanged: it hooks the
    # module assigned to model.fc/classifier/heads.head (now a Sequential
    # wrapping Dropout+Linear) and captures ITS input, i.e. the same
    # pre-dropout penultimate features as before.
    if arch_name == "ResNet18":
        m = tvm.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(m.fc.in_features, num_classes))
    elif arch_name == "DenseNet121":
        m = tvm.densenet121(weights="IMAGENET1K_V1")
        m.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(m.classifier.in_features, num_classes))
    elif arch_name == "ViT_B16":
        m = tvm.vit_b_16(weights="IMAGENET1K_V1")
        m.heads.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(m.heads.head.in_features, num_classes))
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    return m.to(DEVICE)


# =================================================================
# FINE-TUNING ON THE SOURCE DOMAIN TASK
# =================================================================
def train_source_model(model, train_ds, val_ds, epochs=TRAIN_EPOCHS, lr=TRAIN_LR,
                        device=DEVICE, batch_size=BATCH_SIZE, min_acceptable_acc=MIN_ACCEPTABLE_VAL_ACC,
                        name="", max_train_samples=None, max_val_samples=None):
    """
    Fine-tunes the FULL network on the source-domain diagnostic task.
    Without this, the classifier head never learns the task and accuracy
    sits near chance regardless of alpha (the bug in the original pipeline).
    Raises if best validation accuracy never clears `min_acceptable_acc`.

    `max_train_samples`/`max_val_samples`: if set, trains/validates on a
    random subset instead of the full split -- use for a genuinely fast
    smoke test; leave None (full dataset) for the real run.
    """
    if max_train_samples is not None and max_train_samples < len(train_ds):
        idx = np.random.RandomState(0).choice(len(train_ds), max_train_samples, replace=False)
        train_ds = Subset(train_ds, idx)
    if max_val_samples is not None and max_val_samples < len(val_ds):
        idx = np.random.RandomState(0).choice(len(val_ds), max_val_samples, replace=False)
        val_ds = Subset(val_ds, idx)

    # num_workers=0 (the old default) meant fully synchronous data loading --
    # every image read from disk and decoded one at a time, blocking the GPU
    # between batches. Parallelizing this across worker processes lets the
    # GPU stay busy while the next batch loads.
    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True,
                               persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=(num_workers > 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long().view(-1)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long().view(-1)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"[{name}] epoch {epoch+1}/{epochs} | train_loss={running_loss/len(train_ds):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_val_acc < min_acceptable_acc:
        raise RuntimeError(
            f"[{name}] best validation accuracy ({best_val_acc:.4f}) never reached "
            f"{min_acceptable_acc}. Running the alpha-sweep on this model would produce "
            f"an uninterpretable alpha_star. Increase TRAIN_EPOCHS, lower TRAIN_LR, or "
            f"inspect the training curve above before proceeding."
        )

    print(f"[{name}] fine-tuning complete. Best val_acc = {best_val_acc:.4f}")
    return model, best_val_acc


# =================================================================
# FIXED-PANEL TRAJECTORY COLLECTION
# =================================================================
def collect_panel_trajectories(model, arch_name, mahalanobis_params, test_ds, panel_indices,
                                target_partners, alphas, q_hat, device, to_unit, to_model_space,
                                mc_dropout_samples=10):
    """
    Runs the SAME fixed panel through the model at every alpha, returning
    per-image entropy / correctness / conformal-set-size / MSP / Energy /
    Mahalanobis / MC-Dropout matrices (n_alphas x panel_size). Image
    identity is preserved across alpha so AC1/variance/skewness are
    computed over a genuine alpha-indexed time series, and so bootstrap
    resampling and the Safety Shell's blocked/silent counting can reuse
    these cached matrices without re-running inference.

    MC Dropout adds `mc_dropout_samples` extra stochastic forward passes
    PER BATCH on top of the single deterministic pass already needed for
    everything else -- budget for a real slowdown proportional to this,
    specifically in this collection step.
    """
    _num_workers = min(4, os.cpu_count() or 2)
    loader = DataLoader(Subset(test_ds, panel_indices), batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=_num_workers, pin_memory=True,
                         persistent_workers=(_num_workers > 0))
    entropy_matrix, correct_matrix, setsize_matrix = [], [], []
    msp_matrix, energy_matrix, mahalanobis_matrix, mc_dropout_var_matrix = [], [], [], []

    feature_handle, feature_holder = register_feature_hook(model, arch_name)

    model.eval()
    try:
        for alpha in tqdm(alphas, desc=f"{arch_name} alpha sweep"):
            step_entropy, step_correct, step_setsize = [], [], []
            step_msp, step_energy, step_maha, step_mc_var = [], [], [], []
            partner_ptr = 0
            for inputs, labels in loader:
                b = inputs.size(0)
                batch_partners = target_partners[partner_ptr:partner_ptr + b]
                partner_ptr += b

                inputs = apply_clinical_shift(inputs.to(device), batch_partners, alpha, to_unit, to_model_space)
                labels = labels.to(device).long().view(-1)

                with torch.no_grad():
                    logits = model(inputs)  # triggers the feature hook too
                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    pred_sets = (probs > (1 - q_hat)).float().sum(dim=1)
                    msp = probs.max(dim=1)[0].cpu().numpy()
                    energy = (-torch.logsumexp(logits, dim=1)).cpu().numpy()

                penult_features = feature_holder["features"].cpu().numpy()
                maha = mahalanobis_scores(penult_features, mahalanobis_params)
                _, mc_var = mc_dropout_predict(model, inputs, n_samples=mc_dropout_samples)

                step_entropy.extend(ent.cpu().numpy())
                step_correct.extend((preds == labels).float().cpu().numpy())
                step_setsize.extend(pred_sets.cpu().numpy())
                step_msp.extend(msp)
                step_energy.extend(energy)
                step_maha.extend(maha)
                step_mc_var.extend(mc_var)

            entropy_matrix.append(step_entropy)
            correct_matrix.append(step_correct)
            setsize_matrix.append(step_setsize)
            msp_matrix.append(step_msp)
            energy_matrix.append(step_energy)
            mahalanobis_matrix.append(step_maha)
            mc_dropout_var_matrix.append(step_mc_var)
    finally:
        feature_handle.remove()

    return {
        "alphas": np.asarray(alphas),
        "entropy_matrix": np.array(entropy_matrix),
        "correct_matrix": np.array(correct_matrix),
        "setsize_matrix": np.array(setsize_matrix),
        "msp_matrix": np.array(msp_matrix),
        "energy_matrix": np.array(energy_matrix),
        "mahalanobis_matrix": np.array(mahalanobis_matrix),
        "mc_dropout_var_matrix": np.array(mc_dropout_var_matrix),
    }


# =================================================================
# PER-(ARCHITECTURE, SEED) ANALYSIS
# =================================================================
def analyze_one_run(arch_name, seed, model, n_classes, train_ds, val_ds, test_ds,
                     panel_indices, target_partners, cal_loader, to_unit, to_model_space):
    """
    Fine-tunes, calibrates, collects trajectories, and computes every metric
    (EWS/Delta-alpha, Safety Shell, baselines, bootstrap CIs, threshold
    sensitivity) for ONE (architecture, seed) combination. Returns a dict
    bundling everything needed for both the per-seed table and pooling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, val_acc = train_source_model(
        model, train_ds, val_ds, epochs=TRAIN_EPOCHS, lr=TRAIN_LR,
        min_acceptable_acc=MIN_ACCEPTABLE_VAL_ACC,
        max_train_samples=MAX_TRAIN_SAMPLES, max_val_samples=MAX_VAL_SAMPLES,
        name=f"{arch_name}_seed{seed}"
    )

    # --- calibration: conformal q_hat, Mahalanobis stats ---
    model.eval()
    cal_features, cal_labels_all = [], []
    feature_handle, feature_holder = register_feature_hook(model, arch_name)
    with torch.no_grad():
        scores = []
        for inputs, labels in cal_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long().view(-1)
            probs = torch.softmax(model(inputs), dim=1)
            true_prob = probs[range(len(labels)), labels]
            scores.extend((1 - true_prob).cpu().numpy())
            cal_features.append(feature_holder["features"].cpu().numpy())
            cal_labels_all.extend(labels.cpu().numpy())
    feature_handle.remove()
    q_hat = np.quantile(scores, 0.95)
    mahalanobis_params = calibrate_mahalanobis(
        np.concatenate(cal_features, axis=0), np.array(cal_labels_all), num_classes=n_classes
    )

    traj = collect_panel_trajectories(model, arch_name, mahalanobis_params, test_ds, panel_indices,
                                       target_partners, ALPHA_STEPS, q_hat, DEVICE, to_unit, to_model_space,
                                       mc_dropout_samples=MC_DROPOUT_SAMPLES)

    alphas = traj["alphas"]
    entropy_matrix, correct_matrix, setsize_matrix = traj["entropy_matrix"], traj["correct_matrix"], traj["setsize_matrix"]
    state_series = entropy_matrix.mean(axis=1)
    accuracy = correct_matrix.mean(axis=1)

    ews = windowed_ews_indicators(state_series)  # detrend=True by default now
    metrics = calculate_transition_metrics(alphas, accuracy, ews["ac1"], threshold=FAILURE_THRESHOLD)
    ac1_onset = detect_sustained_onset(alphas, ews["ac1"])
    composite = composite_ews_agreement(alphas, ews, ac1_onset["onset_idx"])

    # --- baselines: MSP, Energy, Mahalanobis, MC Dropout ---
    msp_means = traj["msp_matrix"].mean(axis=1)
    energy_means = traj["energy_matrix"].mean(axis=1)
    maha_means = traj["mahalanobis_matrix"].mean(axis=1)
    mc_dropout_means = traj["mc_dropout_var_matrix"].mean(axis=1)  # higher variance = more anomalous, correct convention already
    onset_baselines = baseline_onset_detection(
        alphas, msp_means, energy_means, mahalanobis_means=maha_means,
        extra_signals={"mc_dropout": mc_dropout_means}
    )
    baseline_lead = compare_lead_times(metrics["alpha_star"], onset_baselines)
    baseline_auroc = {
        "msp_auroc": detection_auroc_vs_alpha(traj["msp_matrix"][0], traj["msp_matrix"][-1]),
        "energy_auroc": detection_auroc_vs_alpha(traj["energy_matrix"][0], traj["energy_matrix"][-1]),
        "mahalanobis_auroc": detection_auroc_vs_alpha(traj["mahalanobis_matrix"][0], traj["mahalanobis_matrix"][-1]),
        "mc_dropout_auroc": detection_auroc_vs_alpha(traj["mc_dropout_var_matrix"][0], traj["mc_dropout_var_matrix"][-1]),
    }

    # --- Safety Shell: FAMILY-STRATIFIED redesign (supersedes flat m-of-n
    # voting). Requires >=1 genuine vote from EACH of two conceptually
    # distinct families:
    #   - Dynamical/EWS family: AC1 sustained crossing, OR composite
    #     convergent_csd_evidence validating the (possibly fallback) AC1 point.
    #   - Calibration-independent family: earliest genuine among
    #     {Mahalanobis, Energy, MC Dropout}.
    # Conformal set-size is DELIBERATELY EXCLUDED from voting -- diagnosis
    # showed it shares the same softmax-calibration blind spot as MSP under
    # this shift. Still computed below for diagnostic/raw-signal reporting.
    # See compute_safety_shell_onset_stratified for full rationale.
    setsize_means = setsize_matrix.mean(axis=1)
    conformal_onset = detect_fixed_threshold_onset(alphas, setsize_means, threshold=1.0, persistence=2)
    calibration_independent_onsets = {
        "mahalanobis": onset_baselines["mahalanobis_onset"],
        "energy": onset_baselines["energy_onset"],
        "mc_dropout": onset_baselines["mc_dropout_onset"],
    }
    shell = compute_safety_shell_onset_stratified(alphas, ac1_onset, composite, calibration_independent_onsets)
    shell_lead_time = (metrics["alpha_star"] - shell["shell_alpha_ews"]) if metrics["alpha_star"] > shell["shell_alpha_ews"] else 0.0

    false_alarm_rates = false_alarm_rate_analysis(
        alphas,
        {"ac1": ews["ac1"], "setsize": setsize_means, "msp": msp_means, "energy": energy_means,
         "mahalanobis": maha_means, "mc_dropout": mc_dropout_means},
        n_surrogates=200, seed=SEED + seed,
    )

    # --- blocked / silent failure counts: Shell vs. each baseline, head-to-head ---
    detector_onsets = {
        "SafetyShell": shell["shell_onset_idx"],
        "MSP": onset_baselines["msp_onset"]["onset_idx"],
        "Energy": onset_baselines["energy_onset"]["onset_idx"],
        "Mahalanobis": onset_baselines["mahalanobis_onset"]["onset_idx"],
        "MCDropout": onset_baselines["mc_dropout_onset"]["onset_idx"],
    }
    detector_counts = {det: compute_blocked_silent_counts(correct_matrix, idx) for det, idx in detector_onsets.items()}

    # --- paired bootstrap significance: Shell vs. each baseline on blocked fraction ---
    shell_vs_baseline_sig = {}
    for det_name in ["MSP", "Energy", "Mahalanobis", "MCDropout"]:
        shell_vs_baseline_sig[det_name] = paired_bootstrap_blocked_fraction_test(
            correct_matrix, detector_onsets["SafetyShell"], detector_onsets[det_name],
            n_boot=N_BOOT, seed=SEED, return_samples=True
        )

    # --- bootstrap CIs for alpha*, lead_time, AND shell_lead_time (reuses cached matrices) ---
    rng = np.random.RandomState(SEED + seed)
    boot_alpha_star, boot_lead, boot_shell_lead = [], [], []
    for _ in range(N_BOOT):
        cols = rng.choice(PANEL_SIZE, PANEL_SIZE, replace=True)
        b_state = entropy_matrix[:, cols].mean(axis=1)
        b_acc = correct_matrix[:, cols].mean(axis=1)
        b_maha = traj["mahalanobis_matrix"][:, cols].mean(axis=1)
        b_energy = traj["energy_matrix"][:, cols].mean(axis=1)
        b_mc = traj["mc_dropout_var_matrix"][:, cols].mean(axis=1)
        b_ews = windowed_ews_indicators(b_state)
        bm = calculate_transition_metrics(alphas, b_acc, b_ews["ac1"], threshold=FAILURE_THRESHOLD)
        b_ac1_onset = detect_sustained_onset(alphas, b_ews["ac1"])
        b_composite = composite_ews_agreement(alphas, b_ews, b_ac1_onset["onset_idx"])
        b_calib_onsets = {
            "mahalanobis": detect_sustained_onset(alphas, b_maha),
            "energy": detect_sustained_onset(alphas, b_energy),
            "mc_dropout": detect_sustained_onset(alphas, b_mc),
        }
        b_shell = compute_safety_shell_onset_stratified(alphas, b_ac1_onset, b_composite, b_calib_onsets)
        b_shell_lead = (bm["alpha_star"] - b_shell["shell_alpha_ews"]) if bm["alpha_star"] > b_shell["shell_alpha_ews"] else 0.0

        boot_alpha_star.append(bm["alpha_star"])
        boot_lead.append(bm["lead_time"])
        boot_shell_lead.append(b_shell_lead)

    # --- threshold sensitivity (reuses cached matrices, cheap) ---
    threshold_rows = threshold_sensitivity_analysis(
        alphas, entropy_matrix, correct_matrix, thresholds=THRESHOLD_GRID,
        n_boot=N_BOOT_THRESHOLD, seed=SEED + seed
    )

    return {
        "arch_name": arch_name, "seed": seed, "val_acc": val_acc,
        "traj": traj, "ews": ews, "metrics": metrics, "ac1_onset": ac1_onset, "conformal_onset": conformal_onset,
        "composite": composite, "baseline_lead": baseline_lead, "baseline_auroc": baseline_auroc,
        "shell": shell, "shell_lead_time": shell_lead_time, "false_alarm_rates": false_alarm_rates,
        "detector_counts": detector_counts, "shell_vs_baseline_sig": shell_vs_baseline_sig,
        "boot_alpha_star": boot_alpha_star, "boot_lead": boot_lead, "boot_shell_lead": boot_shell_lead,
        "threshold_rows": threshold_rows,
    }


def ci_of(arr):
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2


# =================================================================
# MAIN
# =================================================================
def main(output_dir=None, seeds=None, thresholds=None):
    if seeds is None:
        seeds = SEEDS
    if thresholds is None:
        thresholds = THRESHOLD_GRID
    if output_dir is None:
        output_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    set_publication_style()

    train_ds, val_ds, test_ds, n_classes = get_data()

    # Panel and RSNA partners are FIXED across every architecture and seed
    # for a controlled, apples-to-apples comparison.
    panel_indices = build_fixed_panel_indices(test_ds, panel_size=PANEL_SIZE, seed=SEED)
    to_unit, to_model_space = get_denormalize_renormalize()
    target_partners = build_target_partners(PANEL_SIZE, DEVICE, root=RSNA_ROOT, pool_size=RSNA_POOL_SIZE)

    cal_indices = np.random.RandomState(SEED).choice(len(train_ds), 500, replace=False)
    cal_loader = DataLoader(Subset(train_ds, cal_indices), batch_size=BATCH_SIZE,
                             num_workers=min(4, os.cpu_count() or 2), pin_memory=True)

    per_seed_rows = []
    runs_by_arch = {arch: [] for arch in ARCHITECTURES}
    raw_signals = {arch: [] for arch in ARCHITECTURES}  # results["raw_signals"] -- see below

    for arch_name in ARCHITECTURES:
        for seed in seeds:
            print(f"\n{'='*60}\n{arch_name} | seed {seed}\n{'='*60}")
            model = initialize_single_model(arch_name, n_classes)
            run = analyze_one_run(arch_name, seed, model, n_classes, train_ds, val_ds, test_ds,
                                   panel_indices, target_partners, cal_loader, to_unit, to_model_space)
            runs_by_arch[arch_name].append(run)

            # --- Raw alpha-indexed signals, saved to disk IMMEDIATELY (survives
            # a lost/restarted kernel session -- exactly the situation that
            # otherwise forces a full, expensive re-run just to inspect a
            # signal) AND kept in-memory as results["raw_signals"] for direct
            # diagnostic checks in the same session, e.g.:
            #   results["raw_signals"]["ResNet18"][0]["setsize_means"]
            traj = run["traj"]
            signals_dict = {
                "alphas": traj["alphas"],
                "accuracy": traj["correct_matrix"].mean(axis=1),
                "ac1_detrended": run["ews"]["ac1"],
                "setsize_means": traj["setsize_matrix"].mean(axis=1),
                "msp_means": traj["msp_matrix"].mean(axis=1),
                "energy_means": traj["energy_matrix"].mean(axis=1),
                "mahalanobis_means": traj["mahalanobis_matrix"].mean(axis=1),
                "mc_dropout_means": traj["mc_dropout_var_matrix"].mean(axis=1),
                # PER-IMAGE MATRICES (spec v3 Section 10): required for the
                # hierarchical bootstrap, potentially-intercepted-failure counts,
                # and the real-data CSD pre-collapse-slope audit (Section 12).
                # Shape (n_alpha, n_panel). Saving these makes every downstream
                # re-analysis recomputable without retraining.
                "entropy_matrix": traj["entropy_matrix"],
                "correct_matrix": traj["correct_matrix"],
                "msp_matrix": traj["msp_matrix"],
                "energy_matrix": traj["energy_matrix"],
                "mahalanobis_matrix": traj["mahalanobis_matrix"],
                "mc_dropout_var_matrix": traj["mc_dropout_var_matrix"],
                "setsize_matrix": traj["setsize_matrix"],
            }
            save_raw_signals_npz(output_dir, arch_name, seed, signals_dict, prefix="linshift_")
            raw_signals[arch_name].append(signals_dict)

            m, comp = run["metrics"], run["composite"]
            per_seed_rows.append({
                "Model": arch_name, "seed": seed, "val_accuracy": run["val_acc"],
                "alpha_star": m["alpha_star"], "alpha_star_sustained": m["alpha_star_sustained"],
                "alpha_ews": m["alpha_ews"], "onset_method": m["onset_method"],
                "lead_time": m["lead_time"], "shell_lead_time": run["shell_lead_time"],
                "shell_agreement_reached": run["shell"]["agreement_reached"],
                "shell_dynamical_vote_source": run["shell"]["dynamical_vote_source"],
                "shell_calibration_independent_vote_source": run["shell"]["calibration_independent_vote_source"],
                "sharpness": m["sharpness"], "fit_r_squared": m["fit_r_squared"],
                "composite_n_agree": comp["n_agree"], "convergent_csd_evidence": comp["convergent_csd_evidence"],
                **run["baseline_lead"], **run["baseline_auroc"],
                "shell_blocked": run["detector_counts"]["SafetyShell"]["blocked"],
                "shell_silent": run["detector_counts"]["SafetyShell"]["silent"],
                "shell_blocked_fraction": run["detector_counts"]["SafetyShell"]["blocked_fraction"],
                "msp_blocked_fraction": run["detector_counts"]["MSP"]["blocked_fraction"],
                "energy_blocked_fraction": run["detector_counts"]["Energy"]["blocked_fraction"],
                "mahalanobis_blocked_fraction": run["detector_counts"]["Mahalanobis"]["blocked_fraction"],
                "mc_dropout_blocked_fraction": run["detector_counts"]["MCDropout"]["blocked_fraction"],
                "shell_vs_msp_p": run["shell_vs_baseline_sig"]["MSP"]["p_value"],
                "shell_vs_energy_p": run["shell_vs_baseline_sig"]["Energy"]["p_value"],
                "shell_vs_mahalanobis_p": run["shell_vs_baseline_sig"]["Mahalanobis"]["p_value"],
                "shell_vs_mc_dropout_p": run["shell_vs_baseline_sig"]["MCDropout"]["p_value"],
                "far_shell": run["false_alarm_rates"]["SafetyShell"],
                "far_ac1": run["false_alarm_rates"]["AC1"],
                "far_conformal": run["false_alarm_rates"]["Conformal"],
                "far_msp": run["false_alarm_rates"]["MSP"],
                "far_energy": run["false_alarm_rates"]["Energy"],
                "far_mahalanobis": run["false_alarm_rates"]["Mahalanobis"],
                "far_mc_dropout": run["false_alarm_rates"]["MCDropout"],
            })

    per_seed_df = pd.DataFrame(per_seed_rows)
    print("\n" + "=" * 70)
    print("     PER-SEED DETAIL TABLE (raw, 3 seeds x 3 architectures)")
    print("=" * 70)
    print(per_seed_df.to_string(index=False))
    export_table(per_seed_df, output_dir, "linshift_table_per_seed_detail")

    # =================================================================
    # AGGREGATE ACROSS SEEDS PER ARCHITECTURE (pooled bootstrap + seed variance)
    # =================================================================
    agg_rows = []
    boot_lead_pool_by_arch = {}
    for arch_name in ARCHITECTURES:
        runs = runs_by_arch[arch_name]
        val_acc_summary = seed_variance_summary([r["val_acc"] for r in runs])
        alpha_star_summary = seed_variance_summary([r["metrics"]["alpha_star"] for r in runs])

        pooled_lead = pool_across_seeds([r["boot_lead"] for r in runs])
        pooled_shell_lead = pool_across_seeds([r["boot_shell_lead"] for r in runs])
        boot_lead_pool_by_arch[arch_name] = np.concatenate([r["boot_lead"] for r in runs])

        n_sustained = sum(r["metrics"]["alpha_star_sustained"] for r in runs)
        agg_rows.append({
            "Model": arch_name,
            "val_accuracy_mean": val_acc_summary["mean"], "val_accuracy_std_across_seeds": val_acc_summary["std"],
            "alpha_star_mean_across_seeds": alpha_star_summary["mean"],
            "alpha_star_std_across_seeds": alpha_star_summary["std"],
            "n_seeds_with_sustained_collapse": f"{n_sustained}/{len(runs)}",
            "lead_time_pooled_mean": pooled_lead["pooled_mean"], "lead_time_pooled_ci95": pooled_lead["ci95"],
            "shell_lead_time_pooled_mean": pooled_shell_lead["pooled_mean"], "shell_lead_time_pooled_ci95": pooled_shell_lead["ci95"],
            "mean_shell_blocked_fraction": np.mean([r["detector_counts"]["SafetyShell"]["blocked_fraction"] for r in runs]),
            "mean_mahalanobis_blocked_fraction": np.mean([r["detector_counts"]["Mahalanobis"]["blocked_fraction"] for r in runs]),
            "mean_energy_blocked_fraction": np.mean([r["detector_counts"]["Energy"]["blocked_fraction"] for r in runs]),
            "mean_msp_blocked_fraction": np.mean([r["detector_counts"]["MSP"]["blocked_fraction"] for r in runs]),
            "mean_mc_dropout_blocked_fraction": np.mean([r["detector_counts"]["MCDropout"]["blocked_fraction"] for r in runs]),
        })
    agg_df = pd.DataFrame(agg_rows)
    print("\n" + "=" * 70)
    print("     POOLED SUMMARY ACROSS SEEDS (training + resampling variance combined)")
    print("=" * 70)
    print(agg_df.to_string(index=False))
    export_table(agg_df, output_dir, "linshift_table_pooled_summary")

    unresolved = agg_df[agg_df["n_seeds_with_sustained_collapse"].apply(lambda s: s.startswith("0"))]["Model"].tolist()
    if unresolved:
        print(f"\n*** WARNING: {unresolved} showed NO seed with a sustained accuracy drop below "
              f"{FAILURE_THRESHOLD} within the tested alpha range. Their lead_time figures reflect "
              f"fallback (censored) values, not measured collapse intervals. ***")

    # =================================================================
    # PAIRWISE ARCHITECTURE SIGNIFICANCE  (pooled across seeds)
    # =================================================================
    pairwise_rows = []
    p_matrix = np.full((len(ARCHITECTURES), len(ARCHITECTURES)), np.nan)
    for i, a in enumerate(ARCHITECTURES):
        for j, b in enumerate(ARCHITECTURES):
            if i < j:
                res = permutation_test_lead_time_diff(boot_lead_pool_by_arch[a], boot_lead_pool_by_arch[b])
                pairwise_rows.append({"comparison": f"{a} vs {b}", "observed_lead_time_diff": res["observed_diff"],
                                       "p_value": res["p_value"], "significant_at_0.05": res["significant"]})
                p_matrix[i, j] = p_matrix[j, i] = res["p_value"]
    pairwise_df = pd.DataFrame(pairwise_rows)
    print("\n" + "=" * 70)
    print("     PAIRWISE ARCHITECTURE SIGNIFICANCE (pooled bootstrap, permutation test)")
    print("=" * 70)
    print(pairwise_df.to_string(index=False))
    export_table(pairwise_df, output_dir, "linshift_table_pairwise_architecture_significance")

    # =================================================================
    # SAFETY SHELL vs. BASELINES  (pooled across seeds, paired significance)
    # =================================================================
    shell_comparison_rows = []
    for arch_name in ARCHITECTURES:
        runs = runs_by_arch[arch_name]
        for det_name in ["MSP", "Energy", "Mahalanobis", "MCDropout"]:
            frac_a_pool = np.concatenate([r["shell_vs_baseline_sig"][det_name]["frac_a_samples"] for r in runs])
            frac_b_pool = np.concatenate([r["shell_vs_baseline_sig"][det_name]["frac_b_samples"] for r in runs])
            diffs = frac_a_pool - frac_b_pool
            valid = ~np.isnan(diffs)
            diffs = diffs[valid]
            no_failures = np.all(np.isnan(frac_a_pool)) or np.all(np.isnan(frac_b_pool))
            p_value = float(min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)) if len(diffs) else float("nan")
            with np.errstate(invalid="ignore"):
                shell_mean = float(np.nanmean(frac_a_pool)) if not np.all(np.isnan(frac_a_pool)) else float("nan")
                base_mean = float(np.nanmean(frac_b_pool)) if not np.all(np.isnan(frac_b_pool)) else float("nan")
                obs_diff = float(np.nanmean(diffs)) if len(diffs) else float("nan")
            shell_comparison_rows.append({
                "Model": arch_name, "comparison": f"SafetyShell vs {det_name}",
                "shell_blocked_fraction_mean": shell_mean,
                f"{det_name.lower()}_blocked_fraction_mean": base_mean,
                "observed_diff": obs_diff,
                "ci95": ci_of(diffs), "p_value": p_value, "significant_at_0.05": p_value < 0.05 if not np.isnan(p_value) else False,
                "no_failures_observed": no_failures,
            })
    shell_comparison_df = pd.DataFrame(shell_comparison_rows)
    print("\n" + "=" * 70)
    print("     SAFETY SHELL vs. OOD BASELINES (blocked-failure fraction, paired bootstrap)")
    print("=" * 70)
    print(shell_comparison_df.to_string(index=False))
    export_table(shell_comparison_df, output_dir, "linshift_table_safety_shell_vs_baselines")

    # =================================================================
    # FALSE-ALARM-RATE COMPARISON
    # =================================================================
    far_rows = []
    for arch_name in ARCHITECTURES:
        runs = runs_by_arch[arch_name]
        far_row = {"Model": arch_name}
        for det_key, det_label in [("SafetyShell", "shell"), ("AC1", "ac1"), ("Conformal", "conformal"),
                                    ("MSP", "msp"), ("Energy", "energy"), ("Mahalanobis", "mahalanobis"),
                                    ("MCDropout", "mc_dropout")]:
            vals = [r["false_alarm_rates"][det_key] for r in runs]
            far_row[f"far_{det_label}_mean"] = float(np.mean(vals))
            far_row[f"far_{det_label}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        far_rows.append(far_row)
    false_alarm_df = pd.DataFrame(far_rows)
    print("\n" + "=" * 70)
    print("     FALSE-ALARM-RATE COMPARISON (surrogate-null test, lower = better)")
    print("=" * 70)
    print(false_alarm_df.to_string(index=False))
    export_table(false_alarm_df, output_dir, "linshift_table_false_alarm_rates")

    # =================================================================
    # THRESHOLD SENSITIVITY  (pooled across seeds per architecture)
    # =================================================================
    threshold_summary_rows = []
    threshold_rows_pooled_by_arch = {}
    for arch_name in ARCHITECTURES:
        runs = runs_by_arch[arch_name]
        pooled_rows = []
        for t_idx, tau in enumerate(thresholds):
            alpha_stars = [r["threshold_rows"][t_idx]["alpha_star"] for r in runs]
            pooled_lead_samples = np.concatenate([r["threshold_rows"][t_idx]["boot_lead_samples"] for r in runs])
            pooled_alpha_star_samples = np.concatenate([r["threshold_rows"][t_idx]["boot_alpha_star_samples"] for r in runs])
            row = {
                "threshold": tau,
                "alpha_star": float(np.mean(alpha_stars)),
                "alpha_star_ci95": ci_of(pooled_alpha_star_samples),
                "lead_time": float(np.mean([r["threshold_rows"][t_idx]["lead_time"] for r in runs])),
                "lead_time_ci95": ci_of(pooled_lead_samples),
            }
            pooled_rows.append(row)
            threshold_summary_rows.append({"Model": arch_name, **row})
        threshold_rows_pooled_by_arch[arch_name] = pooled_rows
    threshold_df = pd.DataFrame(threshold_summary_rows)
    print("\n" + "=" * 70)
    print("     THRESHOLD SENSITIVITY ANALYSIS (tau = 0.65-0.80, pooled across seeds)")
    print("=" * 70)
    print(threshold_df.to_string(index=False))
    export_table(threshold_df, output_dir, "linshift_table_threshold_sensitivity")

    # =================================================================
    # FIGURES
    # =================================================================
    n_arch = len(ARCHITECTURES)
    fig1, axes1 = plt.subplots(n_arch, 2, figsize=(13, 4.2 * n_arch))
    fig2, axes2 = plt.subplots(1, n_arch, figsize=(5.2 * n_arch, 4.5))
    fig3, axes3 = plt.subplots(1, n_arch, figsize=(5.2 * n_arch, 4.5))

    curves_by_arch, alpha_star_by_arch, alpha_ews_by_arch, sustained_by_arch = {}, {}, {}, {}

    for i, arch_name in enumerate(ARCHITECTURES):
        runs = runs_by_arch[arch_name]
        alphas = runs[0]["traj"]["alphas"]

        acc_by_seed = [r["traj"]["correct_matrix"].mean(axis=1) for r in runs]
        plot_accuracy_by_seed(axes1[i, 0], alphas, acc_by_seed, FAILURE_THRESHOLD, f"{arch_name}: Accuracy vs alpha (3 seeds)")

        ac1_by_seed = [r["ews"]["ac1"] for r in runs]
        mean_onset_alpha = float(np.nanmean([r["metrics"]["alpha_ews"] for r in runs]))
        plot_detrended_ac1(axes1[i, 1], alphas, ac1_by_seed, mean_onset_alpha, f"{arch_name}: Detrended AC1 (3 seeds)")

        plot_threshold_sensitivity(axes2[i], threshold_rows_pooled_by_arch[arch_name], f"{arch_name}: tau sensitivity")

        det_names = ["SafetyShell", "Mahalanobis", "Energy", "MSP", "MCDropout"]
        blocked_fracs = [np.mean([r["detector_counts"][d]["blocked_fraction"] for r in runs]) for d in det_names]
        ci_lo, ci_hi = [], []
        for d in det_names:
            vals = [r["detector_counts"][d]["blocked_fraction"] for r in runs]
            ci_lo.append(min(vals)); ci_hi.append(max(vals))
        plot_detector_comparison(axes3[i], det_names, blocked_fracs, ci_lo, ci_hi, f"{arch_name}: Detector comparison")

        # gather what the single summary figure needs
        curves_by_arch[arch_name] = np.mean(acc_by_seed, axis=0)
        alpha_star_by_arch[arch_name] = np.mean([r["metrics"]["alpha_star"] for r in runs])
        alpha_ews_by_arch[arch_name] = np.mean([r["metrics"]["alpha_ews"] for r in runs])
        sustained_by_arch[arch_name] = any(r["metrics"]["alpha_star_sustained"] for r in runs)

    fig1.savefig(os.path.join(output_dir, "linshift_fig_accuracy_and_ac1.png"))
    fig2.savefig(os.path.join(output_dir, "linshift_fig_threshold_sensitivity.png"))
    fig3.savefig(os.path.join(output_dir, "linshift_fig_detector_comparison.png"))

    fig4, ax4 = plt.subplots(figsize=(5.5, 5))
    plot_pairwise_significance_heatmap(ax4, ARCHITECTURES, p_matrix, "Pairwise architecture significance (p-values)")
    fig4.savefig(os.path.join(output_dir, "linshift_fig_pairwise_significance.png"))

    # THE two single-glance summary figures. Deliberately minimal (no
    # in-plot annotation text), and deliberately DIFFERENT in visual form
    # (line plot vs. bar chart) and color scheme (per-architecture vs.
    # per-detector) so they are immediately distinguishable and, together,
    # readable as the paper's central thesis in one pass: Figure 1 shows
    # WHICH architectures are fragile vs. robust and how much warning each
    # gives; Figure 2 shows HOW WELL the proposed Safety Shell catches
    # failures compared to every baseline.
    fig5, ax5 = plt.subplots(figsize=(10, 6.5))
    plot_summary_figure_1_reliability(
        ax5, runs_by_arch[ARCHITECTURES[0]][0]["traj"]["alphas"],
        curves_by_arch, alpha_star_by_arch, alpha_ews_by_arch, sustained_by_arch,
        threshold=FAILURE_THRESHOLD, metric_name="Accuracy"
    )
    fig5.savefig(os.path.join(output_dir, "linshift_fig_SUMMARY_1_reliability_trajectories.png"))

    det_names_summary = ["SafetyShell", "Mahalanobis", "Energy", "MSP", "MCDropout"]
    blocked_fraction_matrix = np.array([
        [np.mean([r["detector_counts"][d]["blocked_fraction"] for r in runs_by_arch[arch]])
         for d in det_names_summary]
        for arch in ARCHITECTURES
    ])
    fig6, ax6 = plt.subplots(figsize=(10, 6.5))
    plot_summary_figure_2_detector_effectiveness(ax6, ARCHITECTURES, det_names_summary, blocked_fraction_matrix)
    fig6.savefig(os.path.join(output_dir, "linshift_fig_SUMMARY_2_detector_effectiveness.png"))

    far_names_summary = ["SafetyShell", "AC1", "Conformal", "MSP", "Energy", "Mahalanobis", "MCDropout"]
    false_alarm_matrix = np.array([
        [np.mean([r["false_alarm_rates"][d] for r in runs_by_arch[arch]]) for d in far_names_summary]
        for arch in ARCHITECTURES
    ])
    fig7, ax7 = plt.subplots(figsize=(11, 6.5))
    plot_summary_figure_2_detector_effectiveness(ax7, ARCHITECTURES, far_names_summary, false_alarm_matrix)
    ax7.set_ylabel("False alarm rate (surrogate-null test, lower = better)")
    fig7.savefig(os.path.join(output_dir, "linshift_fig_false_alarm_rate_comparison.png"))

    plt.show()
    print(f"\nAll tables and figures saved to: {output_dir}")
    print(f"Raw per-run signals saved as raw_signals_<arch>_seed<N>.npz in {output_dir} "
          f"(recoverable via publication_utils.load_raw_signals even after a lost session).")

    # === POST-HOC CALIBRATED ANALYSIS (spec v3): the authoritative numbers.
    # Runs on the saved per-image matrices; calibrates thresholds on pooled
    # nominal data, then computes Warning Lead Time (hierarchical CI), joint
    # false-alarm rates, status/coverage, and the CSD audit. See
    # analyze_experiment_posthoc. metric_is_auc=False for this experiment.
    try:
        posthoc = analyze_experiment_posthoc(raw_signals, threshold=FAILURE_THRESHOLD,
                                             metric_is_auc=False)
    except Exception as _e:
        print(f"[post-hoc analysis skipped: {_e}]")
        posthoc = None

    return {
        "posthoc": posthoc,
        "per_seed_df": per_seed_df, "agg_df": agg_df, "pairwise_df": pairwise_df,
        "shell_comparison_df": shell_comparison_df, "threshold_df": threshold_df,
        "false_alarm_df": false_alarm_df,
        "runs_by_arch": runs_by_arch, "raw_signals": raw_signals,
    }


if __name__ == "__main__":
    main()
