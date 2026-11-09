"""SplitMix64 (Vigna), reine Ganzzahl-Arithmetik: Portfolio-Konvention für allen Zufall der Demos.

Herkunft: wörtlich aus `saa-demo/saa_rng.py` kopiert (nur der Modulkopf ist neu), keine Abhängigkeit zu dieser Demo.

`uniform_block` liefert die ersten `count` Gleichverteilten des Stroms vektorisiert (numpy uint64 läuft modulo 2^64 über),
`normal_block` macht daraus Standardnormalverteilte per Inversion (`scipy.special.ndtri`): auf jeder Plattform dieselben Zahlen."""

import numpy as np
from scipy.special import ndtri

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def uniform_block(seed: int, count: int) -> np.ndarray:
    """Die ersten `count` Gleichverteilten auf [0, 1) des Stroms `SplitMix64(seed)` (Form (count,))."""
    k = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & _MASK) + k * np.uint64(_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def normal_block(seed: int, count: int) -> np.ndarray:
    """Standardnormalverteilte (Inversion der Gleichverteilten, an den Rändern auf (1e-12, 1 - 1e-12) begrenzt)."""
    u = np.clip(uniform_block(seed, count), 1e-12, 1.0 - 1e-12)
    return ndtri(u)
