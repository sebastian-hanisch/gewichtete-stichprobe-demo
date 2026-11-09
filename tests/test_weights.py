"""Gewichte und gewichtetes Quantil: Handbeispiel von Hand, Randfälle und Eigenschaften (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

import math

import numpy as np
import pytest

import gws_weights as W


def _q_knn(x, d, q, k, tau):
    d2 = W.sqdist(np.array([[q]]), x)
    return float(W.weighted_quantile(W.knn_weights(W.neighbour_order(d2), k), d, tau)[0])


def _q_kern(x, d, q, h, tau):
    d2 = W.sqdist(np.array([[q]]), x)
    return float(W.weighted_quantile(W.kernel_weights(d2, h), d, tau)[0])


# --- Handbeispiel (Handrechnung in tests/conftest.py) -----------------------------------------------------------------------------------------

def test_hand_example_knn_three_neighbours_gives_30(hand):
    x, d = hand
    d2 = W.sqdist(np.array([[3.2]]), x)
    assert np.allclose(d2, [[4.84, 1.44, 0.04, 0.64, 3.24]])                      # (3,2 - x)^2 von Hand
    w = W.knn_weights(W.neighbour_order(d2), 3)
    assert np.allclose(w, [[0, 1 / 3, 1 / 3, 1 / 3, 0]])
    assert W.weighted_quantile(w, d, 0.8)[0] == 30.0
    assert float((w[0] * d).sum()) == pytest.approx(19.0)                          # Mittel der drei Nachbarn: (12 + 15 + 30) / 3
    assert float(d.mean()) == pytest.approx(19.8)


def test_hand_example_gauss_kernel_bandwidth_one_gives_30(hand):
    x, d = hand
    d2 = W.sqdist(np.array([[3.2]]), x)
    w = W.kernel_weights(d2, 1.0)[0]
    raw = np.array([math.exp(-0.5 * (3.2 - xi) ** 2) for xi in (1, 2, 3, 4, 5)])
    assert np.allclose(w, raw / raw.sum())
    assert np.allclose(w, [0.0359, 0.1963, 0.3953, 0.2928, 0.0798], atol=5e-5)     # die Handwerte
    order = np.argsort(d)
    assert np.allclose(np.cumsum(w[order]), [0.0359, 0.2321, 0.6274, 0.9202, 1.0], atol=1e-4)
    assert W.weighted_quantile(w[None, :], d, 0.8)[0] == 30.0


def test_k_equal_n_gives_equal_weights_and_the_quantile_of_the_whole_sample(hand):
    x, d = hand
    d2 = W.sqdist(np.array([[3.2]]), x)
    w = W.knn_weights(W.neighbour_order(d2), 5)
    assert np.allclose(w, 0.2)
    assert [W.weighted_quantile(w, d, t)[0] for t in (0.2, 0.4, 0.5, 0.8, 1.0)] == [10.0, 12.0, 15.0, 30.0, 32.0]


def test_k_equal_one_returns_the_value_of_the_nearest_neighbour(hand):
    x, d = hand
    assert _q_knn(x, d, 3.2, 1, 0.8) == 15.0           # nächster: x = 3
    assert _q_knn(x, d, 4.6, 1, 0.1) == 32.0           # nächster: x = 5, unabhängig von tau
    assert _q_knn(x, d, 1.1, 1, 0.99) == 10.0


def test_tau_one_gives_the_maximum_demand_with_positive_weight(hand):
    x, d = hand
    assert _q_knn(x, d, 3.2, 3, 1.0) == 30.0           # Nachbarn 3, 4, 2: Maximum 30, nicht das Maximum aller (32)
    assert _q_knn(x, d, 3.2, 5, 1.0) == 32.0
    assert _q_kern(x, d, 3.2, 1.0, 1.0) == 32.0        # alle Gewichte positiv


# --- Eigenschaften ------------------------------------------------------------------------------------------------------------------------------------

def test_weights_sum_to_one_and_are_nonnegative():
    rng = np.random.default_rng(3)
    d2 = rng.uniform(0, 9, size=(20, 15))
    for k in (1, 4, 15, 40):
        w = W.knn_weights(W.neighbour_order(d2), k)
        assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0).all()
        assert ((w > 0).sum(axis=1) == min(k, 15)).all()
    for h in (0.05, 0.7, 5.0):
        w = W.kernel_weights(d2, h)
        assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0).all()


def test_quantile_is_monotone_in_tau(hand):
    x, d = hand
    taus = np.linspace(0.05, 1.0, 20)
    for q in (1.0, 2.5, 3.2, 4.9):
        for fn in (lambda t: _q_knn(x, d, q, 3, t), lambda t: _q_kern(x, d, q, 0.8, t)):
            vals = [fn(t) for t in taus]
            assert all(a <= b for a, b in zip(vals, vals[1:])), (q, vals)


def test_scaling_all_demands_scales_the_quantile(hand):
    x, d = hand
    for lam in (0.5, 3.0, 100.0):
        assert _q_knn(x, lam * d, 3.2, 3, 0.8) == pytest.approx(lam * 30.0)
        assert _q_kern(x, lam * d, 3.2, 1.0, 0.8) == pytest.approx(lam * 30.0)


def test_permuting_the_observations_changes_nothing(hand):
    x, d = hand
    for perm in ([4, 3, 2, 1, 0], [2, 0, 4, 1, 3]):
        xp, dp = x[perm], d[perm]
        for q in (1.4, 3.2, 4.7):
            assert _q_knn(xp, dp, q, 3, 0.8) == _q_knn(x, d, q, 3, 0.8)
            assert _q_kern(xp, dp, q, 1.0, 0.8) == _q_kern(x, d, q, 1.0, 0.8)


def test_large_bandwidth_gives_equal_weights(hand):
    x, d = hand
    w = W.kernel_weights(W.sqdist(np.array([[3.2]]), x), 1e6)
    assert np.allclose(w, 0.2, atol=1e-9)


def test_pinball_by_hand():
    # tau = 0,8, D = (10, 20), q = 15: Überschuss 5 (Kosten 0,2 * 5) und Mangel 5 (Kosten 0,8 * 5) -> Mittel (1,0 + 4,0) / 2 = 2,5
    assert W.pinball(np.array([15.0, 15.0]), np.array([10.0, 20.0]), 0.8) == pytest.approx(2.5)


# --- Gleichstände und Randfälle ------------------------------------------------------------------------------------------------------------------

def test_ties_in_distance_are_broken_by_the_smaller_index():
    x = np.array([[2.0], [2.0], [2.0], [5.0]])
    d = np.array([1.0, 2.0, 3.0, 4.0])
    d2 = W.sqdist(np.array([[2.0]]), x)
    w = W.knn_weights(W.neighbour_order(d2), 2)
    assert w.tolist() == [[0.5, 0.5, 0.0, 0.0]]
    # zwei gleich weit entfernte Nachbarn links und rechts: der mit der kleineren Nummer (links, x = 2) kommt zuerst
    xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    w2 = W.knn_weights(W.neighbour_order(W.sqdist(np.array([[3.0]]), xs)), 2)
    assert w2.tolist() == [[0.0, 0.5, 0.5, 0.0, 0.0]]                       # Abstände 1, 0, 1: x = 3 (Abstand 0), dann x = 2 (Nummer 1 vor Nummer 3)


def test_ties_in_the_demands_do_not_depend_on_their_order():
    w = np.array([[0.25, 0.25, 0.25, 0.25]])
    assert W.weighted_quantile(w, np.array([5.0, 5.0, 7.0, 7.0]), 0.5)[0] == 5.0
    assert W.weighted_quantile(w, np.array([7.0, 5.0, 7.0, 5.0]), 0.5)[0] == 5.0
    assert W.weighted_quantile(w, np.array([7.0, 5.0, 7.0, 5.0]), 0.51)[0] == 7.0


def test_k_larger_than_n_means_all_cases_with_equal_weights():
    d2 = np.array([[3.0, 1.0, 2.0]])
    assert np.allclose(W.knn_weights(W.neighbour_order(d2), 10), 1 / 3)
    with pytest.raises(ValueError):
        W.knn_weights(W.neighbour_order(d2), 0)


def test_all_x_equal_gives_the_first_k_cases_for_knn_and_equal_weights_for_the_kernel():
    x = np.full((6, 2), 1.5)
    d2 = W.sqdist(np.array([[0.0, 0.0]]), x)
    assert W.knn_weights(W.neighbour_order(d2), 3).tolist() == [[1 / 3, 1 / 3, 1 / 3, 0, 0, 0]]
    assert np.allclose(W.kernel_weights(d2, 0.5), 1 / 6)


def test_kernel_with_tiny_bandwidth_puts_all_weight_on_the_nearest_case_without_nan(hand):
    x, d = hand
    w = W.kernel_weights(W.sqdist(np.array([[3.2]]), x), 1e-3)
    assert not np.isnan(w).any()
    assert w[0, 2] == pytest.approx(1.0) and w.sum() == pytest.approx(1.0)
    assert _q_kern(x, d, 3.2, 1e-3, 0.8) == 15.0
    with pytest.raises(ValueError):
        W.kernel_weights(np.array([[1.0]]), 0.0)


def test_single_observation():
    x, d = np.array([[0.0]]), np.array([7.0])
    for tau in (0.1, 1.0):
        assert _q_knn(x, d, 5.0, 3, tau) == 7.0
        assert _q_kern(x, d, 5.0, 0.2, tau) == 7.0


def test_leave_one_out_with_one_or_two_observations_does_not_fail():
    assert W.loo_losses_knn(np.zeros((1, 1)), np.array([3.0]), 0.8, [5, 10]) == [0.0, 0.0]
    d2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    d = np.array([4.0, 6.0])
    # jeder Fall wird aus dem einen anderen vorhergesagt: q = 6 für Fall 0 (Mangel -2: Verlust 0,2 * 2), q = 4 für Fall 1 (Überschuss 2: Verlust 0,8 * 2)
    assert W.loo_losses_knn(d2, d, 0.8, [5]) == pytest.approx([(0.2 * 2 + 0.8 * 2) / 2])
    assert W.k_grid(2) == [5]          # kein Rasterwert passt: der kleinste bleibt (wird auf die Zahl der Nachbarn begrenzt)
    assert W.k_grid(41) == [5, 10, 20, 40]
    assert W.loo_choice([3.0, 1.0, 1.0], [5, 10, 20]) == (10, 1)   # Gleichstand: der kleinere Rasterwert


def test_grids_scale_with_the_dimension():
    assert W.h_grid(3) == pytest.approx([0.3, 0.5, 0.8, 1.2, 2.0, 3.0])
    assert W.h_grid(12) == pytest.approx([0.6, 1.0, 1.6, 2.4, 4.0, 6.0])      # h * sqrt(12 / 3) = 2 h


def test_squared_distance_is_elementwise_and_exact_for_equal_points():
    a = np.array([[0.1, 0.2], [3.0, 4.0]])
    b = np.array([[0.1, 0.2], [0.4, 0.6]])
    d2 = W.sqdist(a, b)
    assert d2[0, 0] == 0.0
    assert d2[0, 1] == pytest.approx(0.09 + 0.16)
    assert d2[1, 1] == pytest.approx(2.6 ** 2 + 3.4 ** 2)
    assert W.sqdist(np.tile(a, (400, 1)), b).shape == (800, 2)               # mehr Zeilen als ein Block (256)
