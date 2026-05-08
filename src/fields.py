"""Field utility functions for PIC-NIX simulation data."""

import numpy as np


def calc_vector_potential_2d(bx, by, delh):
    """Compute 2D vector potential Az from cell-centered Bx, By.

    Solves Bx = ∂Az/∂y, By = −∂Az/∂x via FFT.  The source term
    R = ∂By/∂x − ∂Bx/∂y is computed with centered finite differences.
    The discrete Laplacian inverse uses the symbol that matches the
    centered-difference-of-centered-difference operator.

    Assumes periodic boundary conditions in both x and y directions.

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
        Vector potential Az at cell centers, zero-mean gauge fixed.
    """
    Ny, Nx = bx.shape

    dBy_dx = (np.roll(by, -1, axis=1) - np.roll(by, 1, axis=1)) / (2.0 * delh)
    dBx_dy = (np.roll(bx, -1, axis=0) - np.roll(bx, 1, axis=0)) / (2.0 * delh)
    R = dBy_dx - dBx_dy

    R_hat = np.fft.rfft2(R)

    nx = np.fft.fftfreq(Nx)
    ny = np.fft.fftfreq(Ny)
    KX, KY = np.meshgrid(nx, ny)
    KX = KX[:, : Nx // 2 + 1]
    KY = KY[:, : Nx // 2 + 1]

    D = (np.sin(2.0 * np.pi * KX) ** 2 + np.sin(2.0 * np.pi * KY) ** 2) / delh**2

    with np.errstate(divide="ignore", invalid="ignore"):
        Az_hat = np.where(D > 1e-15, R_hat / D, 0.0)

    return np.fft.irfft2(Az_hat, s=(Ny, Nx))
