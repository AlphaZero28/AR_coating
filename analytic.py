"""
Analytical reflection coefficient of the multilayer coating
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"      # Times-style math to match
plt.rcParams["font.size"] = 18

c = 299_792_458.0

# ---------------- settings ----------------
eps_i = 1.0                # incident medium (air)
eps_s = 4.0                # substrate
f0 = 20e6                  # design frequency

N = 4                                  
eps = [2, 3, 2, 3]                     
d = [c / (4 * f0 * np.sqrt(e)) for e in eps]   # thickness of each layer(quarter-wave)
# d = [2.65, 2.16, 2.65, 2.16]         

f = np.linspace(0, 40e6, 801)          
# ------------------------------------------

# reflection coefficient at each interface
n = np.sqrt([eps_i] + eps + [eps_s])
Gamma = [(n[m] - n[m + 1]) / (n[m] + n[m + 1]) for m in range(N + 1)]


def gamma_in(f):
    """ total reflection coefficient at frequency f."""
    total = 0
    phi = 0                                            # phi_0 = 0
    for m in range(N + 1):
        if m > 0:
            theta = 2 * np.pi * f * np.sqrt(eps[m - 1]) * d[m - 1] / c   
            phi = phi + 2 * theta                                        
        total = total + Gamma[m] * np.exp(-1j * phi)                    
    return total


# print the numbers
for m in range(N + 1):
    print(f"Gamma_{m} = {Gamma[m]:+.4f}")
G0 = gamma_in(f0)

# without coating: air directly on the substrate
Gamma_bare = (np.sqrt(eps_i) - np.sqrt(eps_s)) / (np.sqrt(eps_i) + np.sqrt(eps_s))
print(f"without coating: Gamma = {Gamma_bare:+.4f}, |Gamma|^2 = {Gamma_bare**2:.4f}")


# plot
plt.figure(figsize=(8, 6))
plt.plot(f / 1e6, np.abs(gamma_in(f))**2, lw=2, label="with coating")
plt.plot(f / 1e6, np.full_like(f, Gamma_bare**2), "--", lw=2, label="without coating")
plt.plot(f0 / 1e6, abs(G0)**2, "o")
plt.legend(loc="upper right")
plt.xlabel("Frequency (MHz)")
plt.ylabel(r" $|\Gamma_{in}|^2$")
plt.title(f"N = {N}")
plt.tight_layout()
plt.savefig("figs/gamma_analytic.png", dpi=150)
plt.show()
