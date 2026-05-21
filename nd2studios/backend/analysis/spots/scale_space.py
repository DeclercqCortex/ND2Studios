from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace


def resolve_sigma(typical_diameter_px: float) -> tuple[float, float]:
    """Return (sigma_inner, sigma_outer) for DoG given the typical diameter.

    sigma_inner = d / (2*sqrt(2*ln2))  — FWHM-matched LoG scale (Lindeberg 1998).
    sigma_outer = 1.6 * sigma_inner    — Marr-Hildreth DoG approximation ratio.
    """
    sigma_in = typical_diameter_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma_in, 1.6 * sigma_in


def log_response(image: np.ndarray, sigma: float) -> np.ndarray:
    """Return sigma^2-normalised Laplacian of Gaussian response (float32).

    Bright spots produce positive response; dark spots produce negative.
    sigma^2 normalisation makes peak amplitude scale-independent (Lindeberg 1998).
    """
    img_f = image.astype(np.float32)
    # gaussian_laplace = ∇²(G_σ * I).  Bright blobs → negative peak (bowl shape).
    # Negate so bright spots → positive peaks.
    return -(sigma ** 2) * gaussian_laplace(img_f, sigma=sigma)


def dog_response(image: np.ndarray, sigma_in: float, sigma_out: float) -> np.ndarray:
    """Return Difference-of-Gaussians band-pass response (float32).

    DoG ≈ LoG when sigma_out / sigma_in ≈ 1.6 (Marr & Hildreth 1980).
    Bright spots produce positive response; dark spots produce negative.
    """
    img_f = image.astype(np.float32)
    return gaussian_filter(img_f, sigma=sigma_in) - gaussian_filter(img_f, sigma=sigma_out)
