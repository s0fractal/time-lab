"""q9-field pilot core (superseded by run.py, kept so the pilots rerun): the crystal's rotors on a periodic D-dimensional lattice.

Each site carries one rotor theta with q5-bound's local physics (friction gamma, bath sigma, torque F, pin
-m g_pin sin(m theta)) and couples to its 2D nearest neighbours by g sin(theta_j - theta_i); no attention. One site, the
origin, is a pacemaker: its torque is F + delta."""
import numpy as np


def coupling(theta, g):
    f = np.zeros_like(theta)
    for ax in range(1, theta.ndim):
        f += np.sin(np.roll(theta, -1, ax) - theta) + np.sin(np.roll(theta, 1, ax) - theta)
    return g * f


def green(L, D, mu2=0.0):
    """G on the L^D torus: (-lap + mu2) G = delta_0 (mu2 = 0: minus the uniform 1/L^D, zero mean)."""
    k = 2 * np.pi * np.fft.fftfreq(L)
    lam = sum(np.meshgrid(*([2 - 2 * np.cos(k)] * D), indexing="ij"))
    with np.errstate(divide="ignore"):
        inv = 1.0 / (lam + mu2)
    if mu2 == 0:
        inv[(0,) * D] = 0.0
    return np.real(np.fft.ifftn(inv))


def distance(L, D):
    i = np.arange(L)
    m = np.minimum(i, L - i).astype(float)
    return np.sqrt(sum(x ** 2 for x in np.meshgrid(*([m] * D), indexing="ij")))


class Lattice:
    def __init__(self, p, L, D, B, delta, seed):
        self.p, self.h = p, p["dt"] / p["S"]
        self.g = p["g_lat"]
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.c1 = np.exp(-p["gamma"] * self.h)
        self.c2 = np.sqrt(p["sigma"] ** 2 * (1 - self.c1 ** 2))
        shape = (B,) + (L,) * D
        self.theta = np.zeros(shape)
        self.P = np.zeros(shape)
        self.X = np.zeros(shape)                         # unwrapped displacement
        self.Fi = np.full((1,) + (L,) * D, p["F"])
        self.Fi[(0,) * (D + 1)] += delta
        self.c = coupling(self.theta, self.g)
        self.f = self._force()

    def _force(self):
        p = self.p
        return self.c - p["m_pin"] * p["g_pin"] * np.sin(p["m_pin"] * self.theta) + self.Fi

    def step(self):
        h = self.h
        self.P = self.P + 0.5 * h * self.f
        self.theta = self.theta + 0.5 * h * self.P
        self.X = self.X + 0.5 * h * self.P
        self.P = self.c1 * self.P + self.c2 * self.rng.standard_normal(self.P.shape)
        self.theta = self.theta + 0.5 * h * self.P
        self.X = self.X + 0.5 * h * self.P
        self.theta = np.remainder(self.theta + np.pi, 2 * np.pi) - np.pi
        self.c = coupling(self.theta, self.g)
        self.f = self._force()
        self.P = self.P + 0.5 * h * self.f


def observe(p, L, D, B, delta, warm, t_total, seed, every=1.0):
    """Time averages over [warm, t_total], sampled every `every`: phi = <X_i - mean X> (B, sites), the mean and variance of
    X_i - X_0 (B, sites), <c_i> (B, sites), and the pacemaker's lead X_0 - mean X at warm and at t_total (B, 2)."""
    lat = Lattice(p, L, D, B, delta, seed)
    h = lat.h
    n_warm, n_total, k = int(round(warm / h)), int(round(t_total / h)), int(round(every / h))
    ax = tuple(range(1, D + 1))
    o = (slice(None),) + (0,) * D
    s1 = s2 = s3 = sc = 0.0
    m = 0
    lead = np.zeros((B, 2))
    for i in range(1, n_total + 1):
        lat.step()
        if i == n_warm:
            lead[:, 0] = lat.X[o] - lat.X.mean(ax)
        if i > n_warm and i % k == 0:
            xb = lat.X.mean(ax, keepdims=True)
            rel = lat.X - lat.X[o].reshape((B,) + (1,) * D)
            s1 = s1 + (lat.X - xb)
            s2 = s2 + rel
            s3 = s3 + rel ** 2
            sc = sc + lat.c
            m += 1
    lead[:, 1] = lat.X[o] - lat.X.mean(ax)
    mu = s2 / m
    return {"phi": s1 / m, "rel_mean": mu, "rel_var": s3 / m - mu ** 2, "c": sc / m, "lead": lead,
            "speed": lat.X.mean() / t_total}
