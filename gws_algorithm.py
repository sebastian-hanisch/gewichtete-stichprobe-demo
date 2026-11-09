"""Kern: Bestellregeln des Newsvendors mit Merkmalen und ihre Mehrkosten gegen das Orakel.

Regeln (alle liefern eine Bestellmenge je Testfall):
  kq_empirisch  Kleinste Quadrate für den Mittelwert, dazu das empirische tau-Quantil der Residuen (zweistufig: erst schätzen, dann bestellen)
  direkt        lineare Quantilregression zum Niveau tau (Pinball-Verlust als lineares Programm, scipy/HiGHS), entscheidet direkt auf den Kosten
  knn           gewichtete Stichprobe mit k nächsten Nachbarn, k per Leave-one-out
  kern          gewichtete Stichprobe mit Gauß-Kerngewichten, Bandbreite per Leave-one-out
Zufallswald-Gewichte werden hier NICHT gerechnet (kein scikit-learn zur Laufzeit); ihre Zahlen stehen nur als eingefrorene Vormessung in `gws_vormessung.json`.

Mehrkosten gegen das Orakel = (Kosten der Regel - Kosten des Orakels) / Kosten des Orakels, in Prozent, gemessen auf den Testfällen (dieselben Testfälle für alle Regeln)."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

import gws_constants as C
import gws_weights as W
from gws_scenario import Data


def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate mit Achsenabschnitt; Koeffizienten (1 + p,)."""
    xa = np.hstack([np.ones((len(x), 1)), x])
    return np.linalg.lstsq(xa, y, rcond=None)[0]


def predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + np.atleast_2d(x) @ coef[1:]


def quantreg(x: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Lineare Quantilregression: min tau * Σu + (1 - tau) * Σv  mit  y - X c = u - v, u, v >= 0 (Koeffizienten (1 + p,)); scipy.optimize.linprog, HiGHS."""
    n, p = x.shape
    xa = np.hstack([np.ones((n, 1)), x])
    cvec = np.concatenate([np.zeros(p + 1), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    a = sparse.hstack([sparse.csr_matrix(xa), sparse.identity(n), -sparse.identity(n)], format="csr")
    bounds = [(None, None)] * (p + 1) + [(0, None)] * (2 * n)
    r = linprog(cvec, A_eq=a, b_eq=y, bounds=bounds, method="highs")
    if not r.success:  # darf bei zulässigen, beschränkten Daten nicht vorkommen; kein stiller Rückfall
        raise RuntimeError(f"Quantilregression nicht gelöst: {r.message}")
    return r.x[: p + 1]


def cost(q: np.ndarray, d: np.ndarray, tau: float) -> float:
    """Mittlere Newsvendor-Kosten b * (D - q)+ + h * (q - D)+ mit (b, h) = KOSTEN_BH[tau]."""
    b, h = C.KOSTEN_BH[tau]
    return float(np.mean(b * np.maximum(d - q, 0.0) + h * np.maximum(q - d, 0.0)))


def extra_cost_percent(q: np.ndarray, data: Data, tau: float) -> float:
    c0 = cost(data.world.oracle_quantile(data.xt, tau), data.dt, tau)
    return 100.0 * (cost(q, data.dt, tau) - c0) / c0


@dataclass(frozen=True)
class Weighted:
    """Ergebnis einer gewichteten Regel: Bestellmengen der Testfälle und der gewählte Parameter (k beziehungsweise h)."""
    q: np.ndarray
    param: float
    losses: tuple
    grid: tuple


def distances(data: Data) -> tuple[np.ndarray, np.ndarray]:
    """Quadratische Abstände Training-Training und Test-Training; einmal rechnen und an mehrere Regeln weitergeben (`dist=`)."""
    return W.sqdist(data.x, data.x), W.sqdist(data.xt, data.x)


def knn_rule(data: Data, tau: float, k: int | None = None, dist=None) -> Weighted:
    """kNN-Gewichte; k per Leave-one-out aus KGRID (nur k <= n - 1), oder ein fest vorgegebenes k."""
    d2_tr, d2_te = distances(data) if dist is None else dist
    ks = W.k_grid(data.n)
    losses = W.loo_losses_knn(d2_tr, data.d, tau, ks)
    kb = W.loo_choice(losses, ks)[0] if k is None else int(k)
    q = W.weighted_quantile(W.knn_weights(W.neighbour_order(d2_te), kb, data.n), data.d, tau)
    return Weighted(q, float(kb), tuple(losses), tuple(ks))


def kernel_rule(data: Data, tau: float, h: float | None = None, dist=None) -> Weighted:
    """Gauß-Kerngewichte; Bandbreite per Leave-one-out aus dem Raster (auf p umgerechnet), oder eine fest vorgegebene (schon umgerechnete) Bandbreite."""
    d2_tr, d2_te = distances(data) if dist is None else dist
    hs = W.h_grid(data.p)
    losses = W.loo_losses_kernel(d2_tr, data.d, tau, hs)
    hb = W.loo_choice(losses, hs)[0] if h is None else float(h)
    q = W.weighted_quantile(W.kernel_weights(d2_te, hb), data.d, tau)
    return Weighted(q, float(hb), tuple(losses), tuple(hs))


def kq_empirical_rule(data: Data, tau: float) -> np.ndarray:
    coef = ols(data.x, data.d)
    res = data.d - predict_linear(coef, data.x)
    return predict_linear(coef, data.xt) + float(np.quantile(res, tau))


def direct_rule(data: Data, tau: float) -> np.ndarray:
    return predict_linear(quantreg(data.x, data.d, tau), data.xt)


@dataclass(frozen=True)
class GridResult:
    """Mehrkosten jedes Rasterwerts (fest vorgegeben) und die Leave-one-out-Wahl auf einer Instanz (Experiment 4)."""
    grid: tuple          # k beziehungsweise Bandbreiten (kernel: auf p umgerechnet)
    loo_losses: tuple    # Leave-one-out-Pinball-Verlust je Rasterwert
    extras: tuple        # Mehrkosten gegen das Orakel in % je Rasterwert (Testfälle)
    chosen: int          # Index der Leave-one-out-Wahl (kleinster Verlust, bei Gleichstand der erste)

    @property
    def chosen_extra(self) -> float:
        return self.extras[self.chosen]

    @property
    def best_extra(self) -> float:
        return min(self.extras)


def knn_grid(data: Data, tau: float) -> GridResult:
    d2_tr, d2_te = distances(data)
    ks = W.k_grid(data.n)
    losses = W.loo_losses_knn(d2_tr, data.d, tau, ks)
    order = W.neighbour_order(d2_te)
    extras = [extra_cost_percent(W.weighted_quantile(W.knn_weights(order, k, data.n), data.d, tau), data, tau) for k in ks]
    return GridResult(tuple(ks), tuple(losses), tuple(extras), W.loo_choice(losses, ks)[1])


def kernel_grid(data: Data, tau: float) -> GridResult:
    d2_tr, d2_te = distances(data)
    hs = W.h_grid(data.p)
    losses = W.loo_losses_kernel(d2_tr, data.d, tau, hs)
    extras = [extra_cost_percent(W.weighted_quantile(W.kernel_weights(d2_te, h), data.d, tau), data, tau) for h in hs]
    return GridResult(tuple(hs), tuple(losses), tuple(extras), W.loo_choice(losses, hs)[1])
