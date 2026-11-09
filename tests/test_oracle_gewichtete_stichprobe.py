"""Orakel-Tests der Demo „Gewichtete Stichprobe“: jede Kernroutine gegen einen ANDEREN Rechenweg (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12).

  gewichtetes Quantil (kNN)   gegen Aufzählung mit exakten Brüchen (`fractions`): kleinster Wert v mit F(v) = Σ w_i [D_i <= v] >= tau, Gleichstände in x und D enthalten
  gewichtetes Quantil (Kern)  gegen Aufzählung mit `math.exp` und `math.fsum` (reines Python, ohne numpy)
  Leave-one-out-Wahl          gegen eine Schleife über Fälle und Rasterwerte in reinem Python
  Quantilregression (LP)      gegen Aufzählung aller Geraden durch zwei Datenpunkte (der Optimalwert liegt bei einer Ecke)
  Kleinste Quadrate           gegen scipy.stats.linregress
  wahres Quantil (Orakel)     gegen numerische Minimierung der erwarteten Kosten auf einem feinen Raster
  Zufallswald-Gewichte        werden in der Demo nicht gerechnet (kein scikit-learn zur Laufzeit): kein Orakel nötig, die Zahlen sind eingefrorene Vormessung

Die Zufallsinstanzen sind klein (n <= 14) und fest; ein Lauf mit 200 Instanzen steht nicht im Repo (Scratch)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import linregress, norm

import gws_algorithm as A
import gws_constants as C
import gws_weights as W
from gws_rng import normal_block, uniform_block
from gws_scenario import World


def _weighted_quantile_by_enumeration(weights, demands, tau):
    """Kleinster Nachfragewert v, dessen Gewicht F(v) = Σ_{D_i <= v} w_i mindestens tau erreicht (Summe exakt in Fractions oder per fsum)."""
    total = math.fsum if isinstance(tau, float) else sum
    for v in sorted(set(demands)):
        if total([w for w, d in zip(weights, demands) if d <= v]) >= tau:
            return v
    return max(demands)


def _small_instance(seed, n, p, grid_x=True):
    """Kleine Instanz aus SplitMix64: ganzzahlige x (viele Gleichstände im Abstand) und ganzzahlige Nachfragen (Gleichstände in D)."""
    u = uniform_block(seed, n * (p + 1))
    x = [[int(u[i * (p + 1) + j] * 4) for j in range(p)] for i in range(n)]
    d = [int(u[i * (p + 1) + p] * 6) + 1 for i in range(n)]
    q = [int(v * 4) for v in uniform_block(seed + 7777, p)]
    return x, d, q


def test_knn_quantile_matches_enumeration_with_exact_fractions():
    """40 Instanzen mit Gleichständen: Nachbarn nach (exaktem Abstand, Nummer), Gewicht 1/k als Bruch; verglichen mit der Rechnung der Demo."""
    for seed in range(40):
        n, p = 3 + seed % 10, 1 + seed % 3
        x, d, q = _small_instance(seed, n, p)
        for k in (1, 2, 3, n):
            for tau in (Fraction(1, 2), Fraction(4, 5), Fraction(19, 20), Fraction(1)):
                dist = sorted((sum((a - b) ** 2 for a, b in zip(q, xi)), i) for i, xi in enumerate(x))
                chosen = {i for _, i in dist[: min(k, n)]}
                w = [Fraction(1, min(k, n)) if i in chosen else Fraction(0) for i in range(n)]
                expected = _weighted_quantile_by_enumeration(w, d, tau)
                xa, da, qa = np.array(x, float), np.array(d, float), np.array([q], float)
                got = W.weighted_quantile(W.knn_weights(W.neighbour_order(W.sqdist(qa, xa)), k), da, float(tau))[0]
                assert got == expected, (seed, k, tau)


def test_kernel_quantile_matches_enumeration_in_pure_python():
    """Kerngewichte mit math.exp, Summen mit fsum; stetige x (keine Gleichstände im Abstand), 40 Instanzen, vier Bandbreiten, drei tau."""
    for seed in range(40):
        n, p = 4 + seed % 9, 1 + seed % 2
        x = normal_block(seed + 1, n * p).reshape(n, p)
        d = np.floor(uniform_block(seed + 99, n) * 40) + 1.0
        q = normal_block(seed + 555, p)
        for h in (0.3, 0.8, 2.0, 6.0):
            raw = [math.exp(-sum((a - b) ** 2 for a, b in zip(q, xi)) / (2 * h * h)) for xi in x]
            s = math.fsum(raw)
            w = [r / s for r in raw]
            for tau in (0.5, 0.8, 0.95):
                expected = _weighted_quantile_by_enumeration(w, d.tolist(), tau)
                got = W.weighted_quantile(W.kernel_weights(W.sqdist(q[None, :], x), h), d, tau)[0]
                assert got == expected, (seed, h, tau)


def _loo_loss_pure(x, d, tau, make_weights):
    """Leave-one-out-Pinball in reinem Python: je Fall i die Gewichte über die übrigen Fälle, Quantil durch Aufzählung, Verlust gemittelt (fsum)."""
    n = len(d)
    losses = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        w = make_weights([sum((a - b) ** 2 for a, b in zip(x[i], x[j])) for j in idx], idx)
        qv = _weighted_quantile_by_enumeration(w, [d[j] for j in idx], tau)
        r = d[i] - qv
        losses.append(max(tau * r, (tau - 1) * r))
    return math.fsum(losses) / n


def test_leave_one_out_losses_and_choice_match_a_pure_python_loop():
    for seed in range(12):
        n, p = 12 + seed % 3, 1 + seed % 2
        xs = normal_block(seed + 31, n * p).reshape(n, p)
        d = np.floor(uniform_block(seed + 301, n) * 50) + 1.0
        x_list, d_list = xs.tolist(), d.tolist()
        tau = (0.5, 0.8, 0.95)[seed % 3]
        d2 = W.sqdist(xs, xs)
        ks = [2, 3, 5, 8, 11]
        got = W.loo_losses_knn(d2, d, tau, ks)
        want_k = []
        for k in ks:
            def knn_w(dists, idx, k=k):
                order = sorted(range(len(idx)), key=lambda t: (dists[t], idx[t]))[: min(k, len(idx))]
                return [1 / min(k, len(idx)) if t in order else 0.0 for t in range(len(idx))]
            want_k.append(_loo_loss_pure(x_list, d_list, tau, knn_w))
        assert got == pytest.approx(want_k, rel=1e-9, abs=1e-9)
        assert W.loo_choice(got, ks)[0] == ks[int(np.argmin(want_k))] or abs(min(want_k) - sorted(want_k)[1]) < 1e-9
        hs = [0.4, 0.9, 2.0]
        got_h = W.loo_losses_kernel(d2, d, tau, hs)
        want_h = []
        for h in hs:
            def kern_w(dists, idx, h=h):
                raw = [math.exp(-0.5 * dd / (h * h)) for dd in dists]
                s = math.fsum(raw)
                return [r / s for r in raw]
            want_h.append(_loo_loss_pure(x_list, d_list, tau, kern_w))
        assert got_h == pytest.approx(want_h, rel=1e-9, abs=1e-9)


def _pinball_sum(coef, x, y, tau):
    return sum(max(tau * (yi - coef[0] - coef[1] * xi), (tau - 1) * (yi - coef[0] - coef[1] * xi)) for xi, yi in zip(x, y))


def test_quantile_regression_optimum_matches_enumeration_of_all_lines_through_two_points():
    """Eindimensionales x: der Optimalwert der Quantilregression wird an einer Geraden durch zwei Datenpunkte angenommen (Ecke des LP); Aufzählung aller Paare."""
    for seed in range(25):
        n = 6 + seed % 7
        x = (normal_block(seed + 11, n) * 2).tolist()
        y = (np.floor(uniform_block(seed + 12, n) * 30) + 1.0 + 3 * np.array(x)).tolist()
        tau = (0.5, 0.8, 0.95)[seed % 3]
        best = math.inf
        for i, j in itertools.combinations(range(n), 2):
            slope = (y[j] - y[i]) / (x[j] - x[i])
            best = min(best, _pinball_sum((y[i] - slope * x[i], slope), x, y, tau))
        coef = A.quantreg(np.array(x).reshape(-1, 1), np.array(y), tau)
        assert _pinball_sum(coef, x, y, tau) == pytest.approx(best, rel=1e-7, abs=1e-7), seed


def test_ols_matches_scipy_linregress():
    for seed in range(20):
        n = 8 + seed
        x = normal_block(seed + 3, n)
        y = 4.0 - 2.5 * x + normal_block(seed + 4, n)
        ref = linregress(x, y)
        coef = A.ols(x.reshape(-1, 1), y)
        assert coef == pytest.approx([ref.intercept, ref.slope], abs=1e-9)


def test_oracle_quantile_minimises_the_expected_cost_on_a_fine_grid():
    """Wahres Quantil gegen numerische Minimierung der erwarteten Kosten: Integral über die Normalverteilung von Y (D = max(0, Y)) auf einem Raster von 40001 Punkten."""
    b = (1.0, 0.0, 0.0)
    for kind in C.WELTEN:
        world = World(kind, b)
        for xval in (-1.5, 0.0, 0.7):
            x = np.array([[xval, 0.3, -0.2]])
            mu, sig = float(world.mu(x)[0]), float(world.sigma(x)[0])
            ys = np.linspace(mu - 8 * sig, mu + 8 * sig, 20001)
            dens = norm.pdf(ys, mu, sig)
            dens = dens / dens.sum()
            d = np.maximum(0.0, ys)
            for tau in C.TAUS:
                bc, hc = C.KOSTEN_BH[tau]
                q0 = float(world.oracle_quantile(x, tau)[0])
                qs = np.arange(max(0.0, q0 - 6.0), q0 + 6.0, 0.1)      # Fenster um das behauptete Optimum: die Kostenfunktion ist konvex, ein lokales Minimum ist das globale
                costs = [(dens * (bc * np.maximum(d - q, 0) + hc * np.maximum(q - d, 0))).sum() for q in qs]
                assert q0 == pytest.approx(qs[int(np.argmin(costs))], abs=0.12), (kind, xval, tau)
