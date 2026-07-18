"""Instrument verification record for SPEC-starlet-l1 (R12 quotable artifact).

Captures: the package's own test-suite result, the gen1 sum-reconstruction
identity, the gen2 reconstruct() defect (unused path), the shipped
noise-plane border artifact (spatial cv + rot90 non-commutation of shipped
SNR), and the D4 exactness of the plateau-normalized l1 convention.

Run under ~/wl-challenge-env. Writes results_p2/starlet_l1_instrument.json.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.starlet_l1_lib import combined_ranges, l1_curves  # noqa: E402

from wl_stats_torch import WLStatistics  # noqa: E402
from wl_stats_torch.starlet import Starlet2D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.expanduser("~/software/wl_stats_torch")
OUT = os.path.join(REPO, "results_p2", "starlet_l1_instrument.json")


def make_grf(n_maps, n=128, slope=-2.0, seed=0):
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amp = k ** (slope / 2.0)
    amp[0, 0] = 0.0
    maps = np.empty((n_maps, n, n))
    for i in range(n_maps):
        f = np.fft.fft2(rng.standard_normal((n, n))) * amp
        m = np.fft.ifft2(f).real
        maps[i] = m / m.std()
    return maps


def main():
    rec = {}

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p",
         "no:cacheprovider", "--override-ini", "addopts="],
        cwd=PKG, capture_output=True, text=True)
    rec["package_tests"] = r.stdout.strip().splitlines()[-1]

    maps = make_grf(2, seed=5)
    st = Starlet2D(n_scales=5, device=torch.device("cpu"), dtype=torch.float64)
    coeffs = st(torch.from_numpy(maps).unsqueeze(1))
    rec["gen1_sum_identity_err"] = float(
        np.max(np.abs(coeffs.sum(dim=1).numpy() - maps)))
    recon = st.reconstruct(coeffs, gen2=True).squeeze(1).numpy()
    rec["gen2_reconstruct_err_UNUSED_PATH"] = float(np.max(np.abs(recon - maps)))

    stats = WLStatistics(n_scales=5, device=torch.device("cpu"))
    out = stats.compute_wavelet_transform(torch.from_numpy(maps),
                                          noise_sigma=1.0)
    nl = out["noise_levels"].numpy()[0]
    rec["shipped_noise_plane_cv_per_scale"] = [
        float(nl[s].std() / nl[s].mean()) for s in range(5)]
    snr0 = out["snr"].numpy()
    stats_r = WLStatistics(n_scales=5, device=torch.device("cpu"))
    rot = np.ascontiguousarray(np.rot90(maps, 1, axes=(1, 2)))
    snr_r = stats_r.compute_wavelet_transform(
        torch.from_numpy(rot), noise_sigma=1.0)["snr"].numpy()
    rec["shipped_snr_rot90_commute_err_per_scale"] = [
        float(np.max(np.abs(np.rot90(snr0[:, s], 1, axes=(1, 2)) - snr_r[:, s])))
        for s in range(5)]
    rec["plateau_noise_per_scale_sigma1"] = [
        float(nl[s].max()) for s in range(5)]

    grf = make_grf(8, seed=3)
    sigma = float(grf.std())
    ranges = combined_ranges([grf], sigma)
    _, l1_ref = l1_curves(grf, sigma, ranges)
    _, l1_rot = l1_curves(np.ascontiguousarray(np.rot90(grf, 1, axes=(1, 2))),
                          sigma, ranges)
    rec["plateau_l1_rot90_max_rel_dev"] = float(max(
        np.max(np.abs(l1_rot[s] - l1_ref[s])) / max(1.0, np.abs(l1_ref[s]).max())
        for s in range(4)))

    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
