"""A team of independent (mu/mu_w, lambda)-CMA-ES workers, vectorised over agents (maximisation).

Standard CMA-ES (Hansen's tutorial defaults: weights, c_c, c_s, c_1, c_mu, d_s), one instance per
agent. ``restart`` re-centres chosen workers with an isotropic distribution; ``generation`` runs one
generation for the active workers and returns their sampled points and values. A worker whose step
size times its largest axis falls below ``tol`` is marked converged (inactive) until it is restarted.
"""
from __future__ import annotations

import numpy as np


class CMAWorkers:
    def __init__(self, n: int, dim: int, lam: int | None = None, tol: float = 1e-12) -> None:
        self.n, self.d = n, dim
        self.lam = lam or 4 + int(3 * np.log(dim))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w**2)
        d, mueff = dim, self.mueff
        self.cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        self.cs = (mueff + 2) / (d + mueff + 5)
        self.c1 = 2 / ((d + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (d + 1)) - 1) + self.cs
        self.chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
        self.tol = tol
        self.m = np.zeros((n, d))
        self.sigma = np.ones(n)
        self.C = np.tile(np.eye(d), (n, 1, 1))
        self.B = self.C.copy()
        self.D = np.ones((n, d))
        self.inv_sqrt = self.C.copy()
        self.pc = np.zeros((n, d))
        self.ps = np.zeros((n, d))
        self.gen = np.zeros(n)
        self.active = np.ones(n, dtype=bool)

    def restart(self, idx: np.ndarray | int, centers: np.ndarray, sigma: float) -> None:
        idx = np.atleast_1d(idx)
        self.m[idx] = np.atleast_2d(centers)
        self.sigma[idx] = sigma
        eye = np.eye(self.d)
        self.C[idx] = eye
        self.B[idx] = eye
        self.inv_sqrt[idx] = eye
        self.D[idx] = 1.0
        self.pc[idx] = 0.0
        self.ps[idx] = 0.0
        self.gen[idx] = 0
        self.active[idx] = True

    def generation(self, rng: np.random.Generator, evaluate, low: float, high: float, max_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One generation for the active workers: returns (agent indices, points [a, lam, d], values [a, lam])."""
        a = np.flatnonzero(self.active)
        if not len(a):
            return a, np.empty((0, self.lam, self.d)), np.empty((0, self.lam))
        z = rng.standard_normal((len(a), self.lam, self.d))
        y = np.einsum("ald,aed->ale", z * self.D[a][:, None, :], self.B[a])
        points = np.clip(self.m[a][:, None, :] + self.sigma[a][:, None, None] * y, low, high)
        y = (points - self.m[a][:, None, :]) / self.sigma[a][:, None, None]
        values = evaluate(points.reshape(-1, self.d)).reshape(len(a), self.lam)

        order = np.argsort(-values, axis=1)[:, : self.mu]
        chosen = np.take_along_axis(y, order[:, :, None], axis=1)
        yw = np.einsum("m,amd->ad", self.w, chosen)
        self.m[a] = self.m[a] + self.sigma[a][:, None] * yw
        self.ps[a] = (1 - self.cs) * self.ps[a] + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.einsum("ade,ae->ad", self.inv_sqrt[a], yw)
        self.gen[a] += 1
        ps_norm = np.linalg.norm(self.ps[a], axis=1)
        hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen[a])) / self.chi < 1.4 + 2 / (self.d + 1)
        self.pc[a] = (1 - self.cc) * self.pc[a] + (hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff))[:, None] * yw
        rank_one = np.einsum("ad,ae->ade", self.pc[a], self.pc[a])
        rank_mu = np.einsum("m,amd,ame->ade", self.w, chosen, chosen)
        old = self.C[a]
        self.C[a] = ((1 - self.c1 - self.cmu) * old + self.c1 * (rank_one + ((1 - hsig) * self.cc * (2 - self.cc))[:, None, None] * old)
                     + self.cmu * rank_mu)
        self.sigma[a] = np.minimum(self.sigma[a] * np.exp((self.cs / self.damps) * (ps_norm / self.chi - 1)), max_sigma)
        sym = (self.C[a] + np.transpose(self.C[a], (0, 2, 1))) / 2
        eig, vec = np.linalg.eigh(sym)
        eig = np.maximum(eig, 1e-20)
        self.C[a] = sym
        self.B[a] = vec
        self.D[a] = np.sqrt(eig)
        self.inv_sqrt[a] = np.einsum("ade,ae,afe->adf", vec, 1 / self.D[a], vec)
        self.active[a] = self.sigma[a] * self.D[a].max(axis=1) > self.tol
        return a, points, values
