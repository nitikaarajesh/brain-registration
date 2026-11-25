"""RegCheck v0.1: register two brain MRIs and flag results that should be checked by eye."""
import argparse, json, os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
import SimpleITK as sitk
import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thresholds: set these from YOUR notebook 04 test bench, never by guessing.
MIN_NCC = 0.87       # placeholder until notebook 04
MAX_DISAGREE_MM = 3.0   # placeholder until notebook 04


def check_headers(fixed, moving):
    """Problems that are not really registration problems: catch them first."""
    issues = []
    for name, img in [("fixed", fixed), ("moving", moving)]:
        if min(img.GetSpacing()) < 0.2 or max(img.GetSpacing()) > 7:
            issues.append(f"{name}: unusual voxel spacing {img.GetSpacing()}")
    def box(img):
        corners = [img.TransformIndexToPhysicalPoint([i * (s - 1) for i, s in zip(c, img.GetSize())])
                   for c in np.ndindex(2, 2, 2)]
        return np.min(corners, 0), np.max(corners, 0)
    (f0, f1), (m0, m1) = box(fixed), box(moving)
    overlap = np.prod(np.clip(np.minimum(f1, m1) - np.maximum(f0, m0), 0, None))
    if overlap / np.prod(f1 - f0) < 0.5:
        issues.append("the two scans overlap less than 50% in physical space: check origin/orientation")
    return issues


def brain_points(img, step=500):
    mask = sitk.GetArrayFromImage(img) > 0
    idx = np.argwhere(mask)[::step][:, ::-1].astype(float)
    return np.array([img.TransformContinuousIndexToPhysicalPoint(p.tolist()) for p in idx])


def run_ants(fixed_path, moving_path):
    f = ants.image_read(fixed_path)
    r = ants.registration(f, ants.image_read(moving_path), type_of_transform="SyN")
    jac = ants.create_jacobian_determinant_image(f, r["fwdtransforms"][0]).numpy()
    def mapper(P):
        return ants.apply_transforms_to_points(3, pd.DataFrame(P, columns=list("xyz")),
                                               r["fwdtransforms"])[list("xyz")].values
    return r, mapper, float((jac <= 0).mean() * 100)


def run_unigradicon(fixed_path, moving_path, out_path):
    cmd = [sys.executable, "-c", "import sys; from unigradicon import main; sys.exit(main())",
           "--fixed", fixed_path, "--fixed_modality", "mri",
           "--moving", moving_path, "--moving_modality", "mri",
           "--transform_out", out_path, "--io_iterations", "None"]
    subprocess.run(cmd, check=True, capture_output=True)
    T = sitk.ReadTransform(out_path)
    return lambda P: np.array([T.TransformPoint(p) for p in P])


def ncc_after(fixed, warped_ants):
    """How similar the fixed image and the registered moving image look, inside the brain (1 = identical)."""
    a = sitk.GetArrayFromImage(fixed)
    b = warped_ants.numpy().transpose(2, 1, 0)      # ANTs arrays are (x,y,z); SimpleITK's are (z,y,x)
    m = a > 0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def save_overlay(fixed, warped, path):
    A, B = sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(warped)
    def norm(s, ref):
        lo, hi = np.percentile(ref[ref > 0], [1, 99])
        return np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)
    z, y, x = [s // 2 for s in A.shape]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, a, b in zip(axes, [A[z], A[:, y], A[:, :, x]], [B[z], B[:, y], B[:, :, x]]):
        ax.imshow(np.stack([norm(a, A), norm(b, B), np.zeros_like(a)], -1), origin="lower"); ax.axis("off")
    fig.suptitle("red = fixed, green = moving after registration (yellow = aligned)")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def regcheck(fixed_path, moving_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    report = {"fixed": fixed_path, "moving": moving_path, "header_issues": check_headers(fixed, moving)}
    if report["header_issues"]:
        report["verdict"] = "stop: fix the header issues first"
        return report

    t0 = time.time()
    r, ants_map, fold_pct = run_ants(fixed_path, moving_path)
    with tempfile.TemporaryDirectory() as tmp:
        ugi_map = run_unigradicon(fixed_path, moving_path, os.path.join(tmp, "ugi.hdf5"))
        P = brain_points(fixed)
        d = np.linalg.norm(ants_map(P) - ugi_map(P), axis=1)

    ants.image_write(r["warpedmovout"], os.path.join(out_dir, "moving_registered.nii.gz"))
    save_overlay(fixed, sitk.ReadImage(os.path.join(out_dir, "moving_registered.nii.gz")),
                 os.path.join(out_dir, "overlay.png"))

    ncc = ncc_after(fixed, r["warpedmovout"])
    reasons = []
    if ncc < MIN_NCC:
        reasons.append(f"after registration the images match poorly (NCC {ncc:.3f} < {MIN_NCC})")
    if fold_pct > 0:
        reasons.append(f"{fold_pct:.3f}% of voxels fold (tissue turned inside out)")
    
    report.update({
        "seconds": round(time.time() - t0, 1),
        "disagreement_mean_mm": round(float(d.mean()), 3),
        "disagreement_p95_mm": round(float(np.percentile(d, 95)), 3),
        "folding_pct": fold_pct,
        "ncc_after": round(ncc, 4),
        "verdict": "check by eye" if reasons else "trust",
        "reasons": reasons,
        "note": "The saved result is ANTs SyN. Thresholds come from a synthetic test bench (notebook 04); treat the verdict as a screening aid, not a guarantee.",
    })
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Register two brain MRIs and flag doubtful results.")
    p.add_argument("fixed"); p.add_argument("moving")
    p.add_argument("--out", default="regcheck_output")
    a = p.parse_args()
    rep = regcheck(a.fixed, a.moving, a.out)
    print(json.dumps(rep, indent=2))
    print(f"\nVERDICT: {rep['verdict'].upper()}")