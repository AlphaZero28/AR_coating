

import numpy as np

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 1 / (MU0 * C0**2)


class FDTD1D:
    def __init__(self, eps_i, eps_s, eps, d, f_max, ppw=40, S=0.5, gap=20, n_sub=150):
        self.eps_i = eps_i

        # grid
        self.dz = C0 / (f_max * np.sqrt(max(eps + [eps_i, eps_s]))) / ppw
        self.dt = S * self.dz / C0
        self.cells = [max(1, round(dm / self.dz)) for dm in d]

        self.j_ref = gap                       # reflection monitor
        self.j_src = 2 * gap                   
        
        self.j0 = 4 * gap                      # coating front (z = 0)
        j_b = self.j0 + np.cumsum([0] + self.cells)      # interfaces
        self.nz = j_b[-1] + n_sub
        self.z = (np.arange(self.nz) - self.j0) * self.dz

        # eps_r on the grid (interface nodes get the average)
        self.eps_r = np.full(self.nz, float(eps_i))
        for m, e in enumerate(eps):
            self.eps_r[j_b[m]:j_b[m + 1]] = e
        self.eps_r[j_b[-1]:] = eps_s
        sides = self.eps_r.copy()
        self.eps_r[j_b] = 0.5 * (sides[j_b - 1] + sides[j_b])

        # update and Mur coefficients
        self.ch = self.dt / (MU0 * self.dz)
        self.ce = self.dt / (EPS0 * self.eps_r * self.dz)
        self.ce_i = self.dt / (EPS0 * eps_i * self.dz)
        mur = lambda e: (C0 / np.sqrt(e) * self.dt - self.dz) / (C0 / np.sqrt(e) * self.dt + self.dz)
        self.kL, self.kR = mur(eps_i), mur(eps_s)

        # Gaussian pulse, spectrum down to 1% at f_max
        self.tau = np.sqrt(np.log(100)) / (np.pi * f_max)
        self.t0 = 5 * self.tau

        self.reset()

    def reset(self):
        self.Ex, self.Hy = np.zeros(self.nz), np.zeros(self.nz - 1)
        self.Ei, self.Hi = np.zeros(self.nz), np.zeros(self.nz - 1)   # incident-only grid
        self.n = 0

    def source(self, t):
        return np.exp(-((t - self.t0) / self.tau) ** 2)

    def step(self):
        js = self.j_src
        E1, Em, Ei1 = self.Ex[1], self.Ex[-2], self.Ei[-2]
        Ei_src = self.Ei[js]

        # H update 
        self.Hy -= self.ch * (self.Ex[1:] - self.Ex[:-1])
        self.Hi -= self.ch * (self.Ei[1:] - self.Ei[:-1])
        self.Hy[js - 1] += self.ch * Ei_src

        # E update 
        self.Ex[1:-1] -= self.ce[1:-1] * (self.Hy[1:] - self.Hy[:-1])
        self.Ex[js] += self.ce[js] * self.Hi[js - 1]
        self.Ei[1:-1] -= self.ce_i * (self.Hi[1:] - self.Hi[:-1])
        self.Ei[0] = self.source((self.n + 1) * self.dt)

        # Mur boundaries
        self.Ex[0] = E1 + self.kL * (self.Ex[1] - self.Ex[0])
        self.Ex[-1] = Em + self.kR * (self.Ex[-2] - self.Ex[-1])
        self.Ei[-1] = Ei1 + self.kL * (self.Ei[-2] - self.Ei[-1])
        self.n += 1

    def run(self, n_steps):
        """Record reflected and incident field at the reflection monitor."""
        t, E_ref, E_inc = np.empty(n_steps), np.empty(n_steps), np.empty(n_steps)
        for k in range(n_steps):
            self.step()
            t[k] = self.n * self.dt
            E_ref[k] = self.Ex[self.j_ref]
            E_inc[k] = self.Ei[self.j_ref]
        return t, E_ref, E_inc

    def reflection(self, t, E_ref, E_inc, f):
        """Gamma(f) at z = 0 from the DFT of the recorded fields."""
        kernel = np.exp(-2j * np.pi * np.outer(f, t))
        beta = 2 * np.pi * f * np.sqrt(self.eps_i) / C0
        return (kernel @ E_ref) / (kernel @ E_inc) * np.exp(-2j * beta * self.z[self.j_ref])
