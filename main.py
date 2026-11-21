"""
1D FDTD simulation of the multilayer coating.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 18

c = 299_792_458.0

# ---------------- settings ----------------
eps_i = 1.0
eps_s = 4.0
f0 = 20e6
f_max = 40e6

N = 4
eps = [2, 3, 2, 3]
d = [c / (4 * f0 * np.sqrt(e)) for e in eps]   # quarter-wave

ppw = 40              # cells per shortest wavelength
S = 0.5               # Courant number
n_air = 80            # air cells before the coating
n_sub = 150           # substrate cells after the coating
# ------------------------------------------

# grid
lam_min = c / (f_max * np.sqrt(max(eps + [eps_i, eps_s])))
dz = lam_min / ppw
dt = S * dz / c

cells = [round(dm / dz) for dm in d]          # layer thickness in cells
nz = n_air + sum(cells) + n_sub

# relative permittivity on the grid
eps_r = np.full(nz, eps_i)
j = n_air
for e, n in zip(eps, cells):
    eps_r[j:j + n] = e
    j += n
eps_r[j:] = eps_s

z = (np.arange(nz) - n_air) * dz              # z = 0 at the coating front

print(f"dz = {dz*100:.2f} cm, dt = {dt*1e9:.4f} ns, nz = {nz}")
for m in range(N):
    print(f"layer {m+1}: eps = {eps[m]}, d = {d[m]:.4f} m -> {cells[m]} cells "
          f"= {cells[m]*dz:.4f} m")

# plot
plt.figure(figsize=(8, 5))
plt.plot(z, eps_r, lw=2)
plt.xlabel("z (m)")
plt.ylabel(r"$\epsilon_r$")
plt.title(f"N = {N}")
plt.ylim(0, eps_s + 1)
plt.tight_layout()
plt.savefig("figs/geometry.png", dpi=150)
plt.show()
