"""Tests for field utility functions."""

import numpy as np
import pytest

from src.fields import calc_vector_potential_2d


@pytest.fixture
def uniform_grid():
    return 64, 48, 0.5


@pytest.fixture
def grid_coords(uniform_grid):
    Nx, Ny, delh = uniform_grid
    x = np.arange(Nx) * delh
    y = np.arange(Ny) * delh
    return np.meshgrid(x, y, indexing="xy"), Nx, Ny, delh


def reconstruct_fields(Az, delh):
    Bx = (np.roll(Az, -1, axis=0) - np.roll(Az, 1, axis=0)) / (2.0 * delh)
    By = -(np.roll(Az, -1, axis=1) - np.roll(Az, 1, axis=1)) / (2.0 * delh)
    return Bx, By


def test_single_mode_reconstruction(grid_coords):
    X, Y, Nx, Ny, delh = *grid_coords[0], *grid_coords[1:]
    Az_true = np.sin(2 * np.pi * X / (Nx * delh)) * np.sin(2 * np.pi * Y / (Ny * delh))
    Bx = (np.roll(Az_true, -1, axis=0) - np.roll(Az_true, 1, axis=0)) / (2.0 * delh)
    By = -(np.roll(Az_true, -1, axis=1) - np.roll(Az_true, 1, axis=1)) / (2.0 * delh)

    Az = calc_vector_potential_2d(Bx, By, delh)
    Bx_rec, By_rec = reconstruct_fields(Az, delh)

    assert np.allclose(Az, Az_true, atol=1e-14)
    assert np.allclose(Bx_rec, Bx, atol=1e-14)
    assert np.allclose(By_rec, By, atol=1e-14)


def test_multi_mode_reconstruction(grid_coords):
    X, Y, Nx, Ny, delh = *grid_coords[0], *grid_coords[1:]
    Az_true = (
        np.sin(2 * np.pi * X / (Nx * delh)) * np.sin(2 * np.pi * Y / (Ny * delh))
        + 0.3 * np.cos(4 * np.pi * X / (Nx * delh))
        + 0.5 * np.sin(3 * np.pi * X / (Nx * delh)) * np.cos(6 * np.pi * Y / (Ny * delh))
    )
    Bx = (np.roll(Az_true, -1, axis=0) - np.roll(Az_true, 1, axis=0)) / (2.0 * delh)
    By = -(np.roll(Az_true, -1, axis=1) - np.roll(Az_true, 1, axis=1)) / (2.0 * delh)

    Az = calc_vector_potential_2d(Bx, By, delh)

    assert np.allclose(Az, Az_true, atol=1e-14)


def test_gauge_invariance(grid_coords):
    X, Y, Nx, Ny, delh = *grid_coords[0], *grid_coords[1:]
    Az_true = np.sin(2 * np.pi * X / (Nx * delh)) * np.sin(2 * np.pi * Y / (Ny * delh))
    Bx = (np.roll(Az_true, -1, axis=0) - np.roll(Az_true, 1, axis=0)) / (2.0 * delh)
    By = -(np.roll(Az_true, -1, axis=1) - np.roll(Az_true, 1, axis=1)) / (2.0 * delh)

    Az_base = calc_vector_potential_2d(Bx, By, delh)

    Az_shifted = Az_true + 5.0
    Bx_s = (np.roll(Az_shifted, -1, axis=0) - np.roll(Az_shifted, 1, axis=0)) / (2.0 * delh)
    By_s = -(np.roll(Az_shifted, -1, axis=1) - np.roll(Az_shifted, 1, axis=1)) / (2.0 * delh)
    Az_from_shifted = calc_vector_potential_2d(Bx_s, By_s, delh)

    assert np.allclose(Az_from_shifted, Az_base, atol=1e-14)
