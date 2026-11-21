
import numpy as np

c = 299_792_458.0
mu0 = 4e-7 * np.pi
eps0 = 1 / (mu0 * c**2)


def init_fields(nz):
    """Ex on nz nodes, Hy on nz-1 half nodes."""
    return np.zeros(nz), np.zeros(nz - 1)


def update_H(Ex, Hy, dt, dz):
    Hy -= dt / (mu0 * dz) * (Ex[1:] - Ex[:-1])


def update_E(Ex, Hy, eps_r, dt, dz):
    Ex[1:-1] -= dt / (eps0 * eps_r[1:-1] * dz) * (Hy[1:] - Hy[:-1])


def mur_coeff(eps_edge, dt, dz):
    v = c / np.sqrt(eps_edge)
    return (v * dt - dz) / (v * dt + dz)


def step(Ex, Hy, eps_r, dt, dz, kL, kR):
    """One time step: H, then E, then Mur on both ends."""
    E1, Em = Ex[1], Ex[-2]
    update_H(Ex, Hy, dt, dz)
    update_E(Ex, Hy, eps_r, dt, dz)
    Ex[0] = E1 + kL * (Ex[1] - Ex[0])
    Ex[-1] = Em + kR * (Ex[-2] - Ex[-1])
