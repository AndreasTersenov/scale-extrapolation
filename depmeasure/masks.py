"""Oriented-context machinery for the B1 shape test.

Design (pre-declared in the B1 prereg): local orientation from the structure tensor
of the coarse map (Gaussian smoothing sigma=2 px), quantized to 4 classes
{0, 45, 90, 135} degrees. Context masks: isotropic disk vs an elongated ellipse
(aspect 4:1) at EXACTLY matched pixel count (farthest-pixels-first trimming).
Within the axis-aligned class pair {0, 90} and the diagonal pair {45, 135}, patches
are canonicalized by EXACT D4 grid transforms (90-degree rotation), so features are
comparable; the two class pairs are fit separately and pooled. The residual
axis-vs-diagonal mask-shape difference at matched area is a documented approximation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def ellipse_offsets(a, b, angle_deg):
    """Integer offsets inside an ellipse with semi-axes (a, b), rotated angle_deg."""
    rr = int(np.ceil(max(a, b)))
    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)
    out = []
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            u = ct * dx + st * dy      # along major axis
            v = -st * dx + ct * dy
            if (u / a) ** 2 + (v / b) ** 2 <= 1.0 + 1e-9:
                out.append((dy, dx))
    return sorted(out)


def match_area(offsets, n_target):
    """Trim to exactly n_target offsets, dropping farthest-from-center first."""
    offs = sorted(offsets, key=lambda p: (p[0] ** 2 + p[1] ** 2, p))
    if len(offs) < n_target:
        raise ValueError(f"mask has {len(offs)} < target {n_target} pixels")
    return sorted(offs[:n_target])


def elongated_offsets(n_target, aspect, angle_deg):
    """Elongated ellipse mask with EXACTLY n_target pixels (grow then trim).

    Semi-axes start at (a, a/aspect) with a chosen from the area target and are
    scaled up until the pixelized ellipse covers >= n_target offsets; the mask is
    then trimmed farthest-first to exactly n_target.
    """
    a = max(1.5, np.sqrt(n_target * aspect / np.pi))
    for _ in range(20):
        offs = ellipse_offsets(a, a / aspect, angle_deg)
        if len(offs) >= n_target:
            return match_area(offs, n_target)
        a *= 1.15
    raise RuntimeError("could not grow mask to target area")


def orientation_class(coarse, sigma=2.0):
    """Per-pixel orientation class in {0, 45, 90, 135} from the structure tensor.

    Returns (cls, coherence): cls in {0,1,2,3} for {0,45,90,135} degrees of the
    dominant ORIENTATION (major axis of local structure), coherence in [0,1].
    """
    c = np.asarray(coarse, dtype=np.float64)
    gy, gx = np.gradient(c)
    Jxx = gaussian_filter(gx * gx, sigma, mode="wrap")
    Jyy = gaussian_filter(gy * gy, sigma, mode="wrap")
    Jxy = gaussian_filter(gx * gy, sigma, mode="wrap")
    # gradient dominant angle; structure orientation is perpendicular to it
    theta_grad = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    theta = theta_grad + np.pi / 2.0
    deg = np.rad2deg(theta) % 180.0
    cls = np.floor(((deg + 22.5) % 180.0) / 45.0).astype(int)
    lam_diff = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lam_sum = Jxx + Jyy
    coherence = np.where(lam_sum > 0, lam_diff / (lam_sum + 1e-30), 0.0)
    return cls, coherence


def rotate_offsets_90(offsets):
    """Exact 90-degree rotation of an offset set: (dy, dx) -> (dx, -dy)."""
    return sorted((dx, -dy) for dy, dx in offsets)
