"""Welten und Daten: Newsvendor mit Merkmalen. Nachfrage D = max(0, mu(x) + sigma(x) * z), z standardnormal, Merkmale standardnormal.

Welten (Form der Vormessung, neu geschrieben; Zufall nur über SplitMix64):
  linear           mu = 60 + 10 * (b . x), sigma = 8
  streuung_linear  mu wie linear, sigma = max(1, 6 + 4 * x1)
  betrag           mu wie linear, sigma = 4 + 6 * |x1|
  quadratisch      mu = 60 + 10 * (b . x) + 8 * (x1^2 - 1), sigma = 8
b ist ein Zufallsvektor der Länge 1 (je Seed fest). Das Orakel kennt mu und sigma und bestellt das wahre bedingte tau-Quantil max(0, mu + sigma * z_tau)."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

import gws_constants as C
from gws_rng import normal_block, uniform_block

_SALT = {"beta": 1, "x": 2, "z": 3, "xt": 4, "zt": 5}
_STEP = 1_000_003  # Abstand der Ströme eines Seeds; Seeds der App liegen weit auseinander, die Ströme überlappen nicht


def _stream(seed: int, name: str) -> int:
    return seed * 7_919 * _STEP + _SALT[name] * _STEP + 12_345


@dataclass(frozen=True)
class World:
    kind: str
    beta: tuple

    def mu(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        m = C.MU0 + C.MU_SLOPE * (x @ np.asarray(self.beta))
        if self.kind == "quadratisch":
            m = m + C.MU_QUAD * (x[:, 0] ** 2 - 1.0)
        return m

    def sigma(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if self.kind == "streuung_linear":
            a, b, lo = C.SIGMA_LIN
            return np.maximum(lo, a + b * x[:, 0])
        if self.kind == "betrag":
            a, b = C.SIGMA_ABS
            return a + b * np.abs(x[:, 0])
        return np.full(len(x), C.SIGMA_CONST)

    def draw(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.mu(x) + self.sigma(x) * z)

    def oracle_quantile(self, x: np.ndarray, tau: float) -> np.ndarray:
        """Wahres tau-Quantil von D gegeben x: D = max(0, Y), also max(0, Quantil von Y). (Exakt für tau > P(Y < 0); sonst ist 0 das Quantil.)"""
        return np.maximum(0.0, self.mu(x) + self.sigma(x) * ndtri(tau))


def make_world(kind: str, p: int, seed: int) -> World:
    if kind not in C.WELTEN:
        raise ValueError(f"unbekannte Welt: {kind}")
    if p < 1:
        raise ValueError("p muss mindestens 1 sein")
    beta = uniform_block(_stream(seed, "beta"), p) * 2.0 - 1.0
    norm = float(np.linalg.norm(beta))
    if norm == 0.0:  # praktisch unmöglich; definierter Rückfall: Richtung der ersten Achse
        beta = np.zeros(p)
        beta[0] = 1.0
        norm = 1.0
    return World(kind, tuple((beta / norm).tolist()))


@dataclass(frozen=True)
class Data:
    world: World
    p: int
    x: np.ndarray       # Training (n, p)
    d: np.ndarray       # Nachfragen (n,)
    xt: np.ndarray      # Testfälle (m, p)
    dt: np.ndarray      # Nachfragen der Testfälle (m,)

    @property
    def n(self) -> int:
        return len(self.d)


def make_data(kind: str, p: int, n: int, seed: int, ntest: int = C.NTEST_LIVE) -> Data:
    """Trainingsdaten und Testfälle einer Instanz; alle Ströme hängen nur von (seed, p, n, ntest) ab, nie von der Plattform."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    world = make_world(kind, p, seed)
    x = normal_block(_stream(seed, "x"), n * p).reshape(n, p)
    z = normal_block(_stream(seed, "z"), n)
    xt = normal_block(_stream(seed, "xt"), ntest * p).reshape(ntest, p)
    zt = normal_block(_stream(seed, "zt"), ntest)
    return Data(world, p, x, world.draw(x, z), xt, world.draw(xt, zt))
