"""
 * author Ohidul Islam
 * created on 20-11-2026-22h-40m
 * copyright 2026
"""

"""
Helper functions for the anti-reflective coating project.

"""

import numpy as np

# Physical constants (SI) used by the helpers below
C0 = 299_792_458.0                # speed of light in vacuum [m/s]
MU0 = 4e-7 * np.pi                # vacuum permeability [H/m]
EPS0 = 1.0 / (MU0 * C0**2)        # vacuum permittivity [F/m]


# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------
def wave_speed(eps_r):
    """v = c / sqrt(eps_r)."""
    return C0 / np.sqrt(eps_r)


def wavelength(eps_r, f):
    """lambda(f) = v / f inside a material."""
    return wave_speed(eps_r) / f


def impedance(eps_r):
    """eta = sqrt(mu0 / (eps0 * eps_r))."""
    return np.sqrt(MU0 / (EPS0 * eps_r))


# ---------------------------------------------------------------------------
# Coating stack
# ---------------------------------------------------------------------------
def validate_layers(layers, allowed_eps):
    """Check every layer uses an allowed eps_r and has positive thickness."""
    for eps_r, d in layers:
        if eps_r not in allowed_eps:
            raise ValueError(f"layer eps_r = {eps_r} not in {sorted(allowed_eps)}")
        if d <= 0:
            raise ValueError(f"layer thickness must be positive, got {d}")


def layer_boundaries(thicknesses):
    """b_0 = 0, b_m = d_1 + ... + d_m. Returns an array of length N + 1."""
    return np.concatenate(([0.0], np.cumsum(thicknesses)))


def print_summary(f0, eps_inc, eps_sub, layers, b):
    """Print constants, media, and the layer boundaries."""
    print(f"c0   = {C0:.6e} m/s")
    print(f"mu0  = {MU0:.6e} H/m")
    print(f"eps0 = {EPS0:.6e} F/m")
    print(f"eta0 = {impedance(1.0):.4f} ohm")
    print(f"f0   = {f0/1e6:.1f} MHz,  lambda0 = {wavelength(1.0, f0):.4f} m\n")

    print(f"incident:  eps_r = {eps_inc}, eta = {impedance(eps_inc):.2f} ohm, z < 0")
    for m, (eps_r, d) in enumerate(layers, start=1):
        print(f"layer {m}:   eps_r = {eps_r}, eta = {impedance(eps_r):.2f} ohm, "
              f"d = {d:.4f} m ({d / wavelength(eps_r, f0):.3f} lambda), "
              f"z in [{b[m-1]:.4f}, {b[m]:.4f}) m")
    print(f"substrate: eps_r = {eps_sub}, eta = {impedance(eps_sub):.2f} ohm, "
          f"z >= b_N = {b[-1]:.4f} m\n")
    print("boundaries b [m]:", np.array2string(b, precision=4))
