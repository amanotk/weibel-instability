"""Field utility functions for PIC-NIX simulation data."""

import numpy as np


def calc_vector_potential_2d(bx, by, delh):
    """Compute 2D vector potential Az from cell-centered Bx, By.

    Solves Bx = ∂Az/∂y, By = −∂Az/∂x via FFT.  The source term
    R = ∂By/∂x − ∂Bx/∂y is computed with centered finite differences.
    The discrete Laplacian inverse uses the symbol that matches the
    centered-difference-of-centered-difference operator.

    Non-zero mean components of Bx and By are recovered as linear ramps
    (Az = B̄x·y − B̄y·x) and added back to the FFT solution, so the
    returned Az includes the full absolute-scale vector potential.

    Parameters
    ----------
    bx : (Ny, Nx) array
        Bx component on cell centers (e.g. after mean over z).
    by : (Ny, Nx) array
        By component on cell centers (e.g. after mean over z).
    delh : float
        Uniform grid spacing.

    Returns
    -------
    Az : (Ny, Nx) array
        Vector potential Az at cell centers including mean-field contribution.
    """
    Ny, Nx = bx.shape

    dBy_dx = (np.roll(by, -1, axis=1) - np.roll(by, 1, axis=1)) / (2.0 * delh)
    dBx_dy = (np.roll(bx, -1, axis=0) - np.roll(bx, 1, axis=0)) / (2.0 * delh)
    Bx_avg = np.mean(bx)
    By_avg = np.mean(by)
    R = dBy_dx - dBx_dy

    R_hat = np.fft.rfft2(R)

    nx = np.fft.fftfreq(Nx)
    ny = np.fft.fftfreq(Ny)
    KX, KY = np.meshgrid(nx, ny)
    KX = KX[:, : Nx // 2 + 1]
    KY = KY[:, : Nx // 2 + 1]

    D = (np.sin(2.0 * np.pi * KX) ** 2 + np.sin(2.0 * np.pi * KY) ** 2) / delh**2 + 1.0e-15
    Az = np.fft.irfft2(R_hat / D, s=(Ny, Nx))
    Az_By_avg = -By_avg * np.arange(Nx)[None, :] * delh
    Az_Bx_avg = +Bx_avg * np.arange(Ny)[:, None] * delh

    return Az + Az_By_avg + Az_Bx_avg
