# =================================================================
# CROSS-DOMAIN SHIFT MODULE
# Replaces synthetic Gaussian perturbation with a real source->target
# domain shift, implemented via Fourier Domain Adaptation (FDA):
# low-frequency AMPLITUDE spectrum is blended from PneumoniaMNIST (source)
# toward a fixed RSNA (target) partner image; PHASE (structure/anatomy)
# is preserved from the source image at all alpha levels.
#
# This is a literal instantiation of the paper's own formalism:
#     P_t(X) = T_alpha(P_s(X))
# with T_alpha now driven by real target-domain statistics instead of
# additive noise, and alpha=0 -> pure source, alpha=1 -> full low-freq
# style transfer from a real RSNA image, continuous in between.
#
# REQUIRES: pydicom (for RSNA DICOM files) and the Kaggle competition
# dataset "rsna-pneumonia-detection-challenge" added as a notebook input.
#   pip install pydicom -q
# =================================================================

import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

try:
    import pydicom
    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False


# =================================================================
# 1. RSNA TARGET-DOMAIN POOL
# =================================================================
class RSNATargetPool:
    """
    Loads a fixed, seeded pool of RSNA DICOM images, preprocessed with the
    SAME pipeline as the source domain (resize 224, grayscale->3ch, [0,1]
    scaling), to serve as amplitude-blending partners.

    On Kaggle, add the competition dataset as a notebook input:
        Add Data -> Competitions -> "RSNA Pneumonia Detection Challenge"
    Default root matches this user's confirmed Kaggle mount path
    (some Kaggle environments nest competition inputs under /competitions/;
    check `!ls /kaggle/input/` if this default doesn't match yours).
    """

    def __init__(self, root="/kaggle/input/competitions/rsna-pneumonia-detection-challenge",
                 pool_size=500, seed=42, image_subdir="stage_2_train_images"):
        if not _PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required to load RSNA DICOM images. "
                               "Run: pip install pydicom -q")

        img_dir = os.path.join(root, image_subdir)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"RSNA image directory not found at {img_dir}. "
                "Add the 'rsna-pneumonia-detection-challenge' dataset as a "
                "Kaggle notebook input, or pass a custom `root`."
            )

        all_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".dcm"))
        rng = np.random.RandomState(seed)
        n = min(pool_size, len(all_files))
        chosen = rng.choice(all_files, size=n, replace=False)
        self.paths = [os.path.join(img_dir, f) for f in chosen]

        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def get(self, idx):
        """Returns a (3, 224, 224) tensor in [0,1], cached after first read."""
        if idx in self._cache:
            return self._cache[idx]

        dcm = pydicom.dcmread(self.paths[idx])
        arr = dcm.pixel_array.astype(np.float32)
        arr = arr / (arr.max() + 1e-8)
        img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
        tensor = self.resize(img).repeat(3, 1, 1)  # (3,224,224), matches pipeline channels
        self._cache[idx] = tensor
        return tensor

    def preload_all(self):
        """Loads every pool image into memory as one (N,3,224,224) tensor."""
        return torch.stack([self.get(i) for i in range(len(self.paths))])


# =================================================================
# 2. FIXED SOURCE-TO-TARGET PARTNER ASSIGNMENT
# =================================================================
def assign_fixed_partners(panel_size, pool_size, seed=42):
    """
    Assigns ONE fixed RSNA partner index to each of the `panel_size` source
    images, reused at every alpha level. This is what preserves image
    identity across the alpha sweep (required for the panel/AC1 design):
    only the BLEND WEIGHT changes with alpha, never the partner.
    """
    rng = np.random.RandomState(seed)
    replace = pool_size < panel_size
    return rng.choice(pool_size, size=panel_size, replace=replace)


# =================================================================
# 3. FDA LOW-FREQUENCY AMPLITUDE BLEND  (the T_alpha operator)
# =================================================================
def fda_cross_domain_shift(source_batch, target_partner_batch, alpha, beta=0.10):
    """
    Batched Fourier Domain Adaptation blend.

    source_batch:          (B, C, H, W) tensor in [0,1], the fixed panel images
    target_partner_batch:  (B, C, H, W) tensor in [0,1], each row's FIXED RSNA partner
    alpha:                 scalar in [0,1], shift intensity
    beta:                  fraction of image size defining the low-frequency
                            "style" window that gets blended (default 0.10,
                            following the FDA paper's recommended range)

    Returns the shifted batch: source structure (phase) preserved, low-frequency
    amplitude linearly interpolated toward the target partner's amplitude.
    alpha=0 -> identical to source_batch. alpha=1 -> full low-freq replacement.
    """
    device = source_batch.device
    target_partner_batch = target_partner_batch.to(device)

    fft_src = torch.fft.fft2(source_batch, dim=(-2, -1))
    fft_trg = torch.fft.fft2(target_partner_batch, dim=(-2, -1))

    amp_src, pha_src = fft_src.abs(), fft_src.angle()
    amp_trg = fft_trg.abs()

    amp_src_shift = torch.fft.fftshift(amp_src, dim=(-2, -1))
    amp_trg_shift = torch.fft.fftshift(amp_trg, dim=(-2, -1))

    _, _, H, W = source_batch.shape
    b = max(1, int(min(H, W) * beta / 2))
    cy, cx = H // 2, W // 2

    amp_mixed_shift = amp_src_shift.clone()
    region_src = amp_src_shift[:, :, cy - b:cy + b, cx - b:cx + b]
    region_trg = amp_trg_shift[:, :, cy - b:cy + b, cx - b:cx + b]
    amp_mixed_shift[:, :, cy - b:cy + b, cx - b:cx + b] = (1 - alpha) * region_src + alpha * region_trg

    amp_mixed = torch.fft.ifftshift(amp_mixed_shift, dim=(-2, -1))
    fft_mixed = amp_mixed * torch.exp(1j * pha_src)
    shifted = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real

    return torch.clamp(shifted, 0.0, 1.0)


# =================================================================
# 3b. PIXEL-SPACE LINEAR BLEND  (parallel secondary shift experiment)
# =================================================================
def linear_pixel_blend_shift(source_batch, target_partner_batch, alpha):
    """
    Direct pixel-space linear interpolation: x_alpha = (1-alpha)*x_source
    + alpha*x_target. A SECOND, PARALLEL shift mechanism -- NOT a
    replacement for fda_cross_domain_shift -- run as its own separate
    experiment to test two things at once: (1) whether a more literal,
    monotonic mixing coefficient produces cleaner classic CSD precursors
    than FDA's low-frequency amplitude blend, and (2) whether the Warning
    Lead Time framework's conclusions generalize across shift MECHANISMS,
    not just across datasets/tasks.

    RATIONALE for why this might produce cleaner CSD precursors where FDA
    did not: FDA blending swaps only low-frequency amplitude while
    preserving source phase/structure -- a well-trained CNN's early
    frequency-sensitive filters may partially "correct for" this kind of
    input change confidently, rather than genuinely losing predictive
    information, which is a plausible explanation for why the FDA-shifted
    models were observed to become MORE confident (not less) as alpha
    increased (MSP AUROC below chance) -- leaving no genuine "loss of
    resilience" for CSD theory to find a precursor to. Direct pixel
    blending is monotonic and literal (alpha IS the textbook bifurcation
    parameter CSD theory assumes) and genuinely destroys/replaces
    diagnostic content rather than just restyling it, which should push
    entropy/uncertainty up in a smoother, more organic way as real
    information degrades.

    KNOWN TRADEOFF, stated plainly: intermediate alpha values here can
    look like a double-exposure/ghosting artifact rather than a plausible
    clinical image -- weaker face validity than FDA's "always looks like
    a real X-ray" property. This is exactly why it's run as a SEPARATE
    experiment rather than replacing FDA: the two mechanisms make
    different tradeoffs (clinical realism vs. textbook-clean bifurcation
    dynamics) and comparing their results directly is itself a finding.

    source_batch, target_partner_batch: (B, C, H, W) tensors in [0,1].
    alpha: scalar in [0,1]. alpha=0 -> identical to source. alpha=1 ->
    identical to the target partner.
    """
    device = source_batch.device
    target_partner_batch = target_partner_batch.to(device)
    blended = (1 - alpha) * source_batch + alpha * target_partner_batch
    return torch.clamp(blended, 0.0, 1.0)


# =================================================================
# 4. SANITY CHECK / SELF-TEST
# =================================================================
if __name__ == "__main__":
    # Quick numerical sanity check with synthetic tensors (no RSNA/pydicom
    # required) to confirm alpha=0 is identity and the blend is continuous,
    # for BOTH shift mechanisms.
    torch.manual_seed(0)
    src = torch.rand(4, 3, 224, 224)
    trg = torch.rand(4, 3, 224, 224)

    print("--- FDA low-frequency amplitude blend ---")
    out_a0 = fda_cross_domain_shift(src, trg, alpha=0.0)
    out_a1 = fda_cross_domain_shift(src, trg, alpha=1.0)
    out_a05 = fda_cross_domain_shift(src, trg, alpha=0.5)
    print("alpha=0 matches source (max abs diff):", (out_a0 - src).abs().max().item())
    print("alpha=1 differs from source (mean abs diff):", (out_a1 - src).abs().mean().item())
    print("alpha=0.5 is between the two (sanity):",
          ((out_a05 - out_a0).abs().mean() > 0).item(),
          ((out_a05 - out_a1).abs().mean() > 0).item())

    print("\n--- Pixel-space linear blend ---")
    lout_a0 = linear_pixel_blend_shift(src, trg, alpha=0.0)
    lout_a1 = linear_pixel_blend_shift(src, trg, alpha=1.0)
    lout_a05 = linear_pixel_blend_shift(src, trg, alpha=0.5)
    print("alpha=0 matches source (max abs diff):", (lout_a0 - src).abs().max().item())
    print("alpha=1 matches target (max abs diff):", (lout_a1 - trg).abs().max().item())
    print("alpha=0.5 is exactly the midpoint (sanity):",
          (lout_a05 - 0.5 * (src + trg)).abs().max().item())
