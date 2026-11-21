
"""
Anti-reflective coating at 20 MHz -- 1D FDTD.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fdtd import FDTD1D, C0

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 18 

# ---------------- settings ----------------
eps_i = 1.0
eps_s = 4.0
f0 = 20e6
f_max = 40e6

N = 4
eps = [2, 3, 2, 3]
d = [C0 / (4 * f0 * np.sqrt(e)) for e in eps]   # quarter-wave

T_run = 2e-6                      
f = np.linspace(0, f_max, 401)
# ------------------------------------------

sim = FDTD1D(eps_i, eps_s, eps, d, f_max)
print(f"dz = {sim.dz*100:.2f} cm, dt = {sim.dt*1e9:.4f} ns, nz = {sim.nz}, cells = {sim.cells}")

b = np.cumsum([0] + sim.cells) * sim.dz       # layer boundaries on the grid
colors = {2: "#2a78d6", 3: "#eb6834"}


def draw_regions(ax, label_y):
    """Shade air, layers, substrate; mark the reflection monitor and source."""
    ax.axvspan(sim.z[0], 0, color="#e6e5e0", alpha=0.6, lw=0)
    for m in range(N):
        ax.axvspan(b[m], b[m + 1], color=colors[eps[m]], alpha=0.25, lw=0)
    ax.axvspan(b[-1], sim.z[-1], color="#c3c2b7", alpha=0.6, lw=0)
    for j, label, ls in [(sim.j_ref, "ref monitor", ":"), (sim.j_src, "source", "--")]:
        ax.axvline(sim.z[j], color="0.4", ls=ls, lw=1.2)
        ax.text(sim.z[j], label_y, label, rotation=90, ha="right", va="bottom",
                fontsize=11, color="0.4")
    ax.set_xlim(sim.z[0], sim.z[-1])


# geometry
fig, ax = plt.subplots(figsize=(12, 5))
draw_regions(ax, 0.25)
ax.plot(sim.z, sim.eps_r, color="k", lw=2, drawstyle="steps-mid")
for m in range(N):
    ax.text((b[m] + b[m + 1]) / 2, eps[m] + 0.12, f"{m+1}", ha="center", fontsize=14)
ax.text(sim.z[sim.j_src] / 2, 0.25, f"air\n$\\epsilon_r$={eps_i:g}", ha="center", fontsize=14)
ax.text((b[-1] + sim.z[-1]) / 2, 0.25, f"substrate\n$\\epsilon_r$={eps_s:g}", ha="center", fontsize=14)
ax.plot([], [], color=colors[2], alpha=0.4, lw=10, label=r"$\epsilon_r$ = 2 layer")
ax.plot([], [], color=colors[3], alpha=0.4, lw=10, label=r"$\epsilon_r$ = 3 layer")
ax.set_ylim(0, eps_s + 0.8)
ax.set_xlabel("z (m)")
ax.set_ylabel(r"$\epsilon_r(z)$")
ax.legend(loc="upper left", frameon=False, fontsize=14)
fig.tight_layout()
fig.savefig("figs/geometry.png", dpi=150)

# snapshots
times = np.linspace(90e-9, 255e-9, 5)
fig, axes = plt.subplots(len(times), 1, figsize=(12, 12), sharex=True)
for ax, ts in zip(axes, times):
    while sim.n * sim.dt < ts:
        sim.step()
    draw_regions(ax, -1.5)
    ax.plot(sim.z, sim.Ex, color="k", lw=1.8)
    ax.set_ylim(-1.6, 1.6)
    ax.set_ylabel(r"$E_x$")
    ax.text(0.99, 0.92, f"t = {sim.n * sim.dt * 1e9:.1f} ns", transform=ax.transAxes,
            ha="right", va="top", fontsize=14)
axes[-1].set_xlabel("z (m)")
fig.tight_layout()
fig.savefig("figs/snapshots.png", dpi=150)
sim.reset()

# reflection
t, E_ref, E_inc = sim.run(int(T_run / sim.dt))
gamma = sim.reflection(t, E_ref, E_inc, f)
G0 = sim.reflection(t, E_ref, E_inc, np.array([f0]))[0]
Gamma_bare = (np.sqrt(eps_i) - np.sqrt(eps_s)) / (np.sqrt(eps_i) + np.sqrt(eps_s))
print(f"|Gamma(f0)| = {abs(G0):.4f}, |Gamma(f0)|^2 = {100 * abs(G0)**2:.3f} %")

plt.figure(figsize=(8, 6))
plt.plot(f / 1e6, np.abs(gamma)**2, lw=2, label="with coating")
plt.plot(f / 1e6, np.full_like(f, Gamma_bare**2), "--", lw=2, label="without coating")
plt.plot(f0 / 1e6, abs(G0)**2, "o")
plt.xlabel("Frequency (MHz)")
plt.ylabel(r"$|\Gamma_{in}|^2$")
plt.title(f"FDTD, N = {N}")
plt.legend()
plt.tight_layout()
plt.savefig("figs/gamma_fdtd.png", dpi=150)

# animation
sim.reset()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sim.z, sim.eps_r / eps_s, color="0.7")
line, = ax.plot(sim.z, sim.Ex, lw=2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("z (m)")
ax.set_ylabel(r"$E_x$")


def update(_):
    for _ in range(10):
        sim.step()
    line.set_ydata(sim.Ex)
    ax.set_title(f"t = {sim.n * sim.dt * 1e9:.0f} ns")
    return line,


anim = FuncAnimation(fig, update, frames=int(300e-9 / sim.dt) // 10, interval=20, repeat=False)
plt.show()
