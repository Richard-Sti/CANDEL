# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Linear perturbation theory for computing velocity fields from density fields.
"""

import numpy as np


def get_kvectors(N, boxsize):
    """
    Return Fourier space wave vectors for full FFT.

    Parameters
    ----------
    N : int
        Grid resolution.
    boxsize : float
        Box size in Mpc/h.

    Returns
    -------
    kx, ky, kz : tuple of ndarray
        3D arrays of wave vector components.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=boxsize / N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=boxsize / N)
    kz = 2.0 * np.pi * np.fft.fftfreq(N, d=boxsize / N)
    return np.meshgrid(kx, ky, kz, indexing='ij')


def growth_rate(Omega_m):
    """
    Logarithmic growth rate f = Omega_m^0.55 (EdS approximation).

    Parameters
    ----------
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float
        Growth rate.
    """
    return Omega_m**0.55


def hubble_parameter(Omega_m, h, a):
    """
    Hubble parameter H(a) in km/s/Mpc.

    Parameters
    ----------
    Omega_m : float
        Matter density parameter.
    h : float
        Reduced Hubble constant.
    a : float
        Scale factor.

    Returns
    -------
    float
        Hubble parameter in km/s/Mpc.
    """
    Omega_L = 1.0 - Omega_m
    E_a = np.sqrt(Omega_m / a**3 + Omega_L)
    return 100.0 * h * E_a


def delta_to_velocity(delta, boxsize, Omega_m, h=1.0, a=1.0, beta=None):
    """
    Convert overdensity to velocity using linear theory.

    Linear theory gives: v_i(k) = i * f * H(a) * a * delta(k) * k_i / k^2
    If `beta` is provided, it replaces the growth rate `f` in the
    prefactor (useful for galaxy overdensity fields).

    Parameters
    ----------
    delta : ndarray, shape (N, N, N)
        Overdensity field.
    boxsize : float
        Box size in Mpc/h.
    Omega_m : float
        Matter density parameter.
    h : float, optional
        Reduced Hubble constant. Default is 1.0.
    a : float, optional
        Scale factor. Default is 1.0.

    Returns
    -------
    v_field : ndarray, shape (3, N, N, N)
        Velocity field in km/s.
    """
    N = delta.shape[0]
    f = growth_rate(Omega_m) if beta is None else beta
    H_a = hubble_parameter(Omega_m, h, a)
    prefactor = f * H_a * a

    delta_k = np.fft.fftn(delta)
    kx, ky, kz = get_kvectors(N, boxsize)
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0

    vx_k = 1j * prefactor * delta_k * kx / k_sq
    vy_k = 1j * prefactor * delta_k * ky / k_sq
    vz_k = 1j * prefactor * delta_k * kz / k_sq

    vx_k[0, 0, 0] = 0.0
    vy_k[0, 0, 0] = 0.0
    vz_k[0, 0, 0] = 0.0

    vx = np.fft.ifftn(vx_k).real
    vy = np.fft.ifftn(vy_k).real
    vz = np.fft.ifftn(vz_k).real

    return np.stack([vx, vy, vz], axis=0)
