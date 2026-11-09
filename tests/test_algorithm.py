"""Bestellregeln gegen Handrechnung: Kleinste Quadrate, Quantilregression, Kosten, Mehrkosten, kNN- und Kernregel auf kleinen Daten (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import gws_algorithm as A
import gws_constants as C
import gws_weights as W
from gws_scenario import Data, World, make_data


def test_ols_recovers_an_exact_line():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    coef = A.ols(x, 5.0 + 2.0 * x[:, 0])
    assert coef == pytest.approx([5.0, 2.0])
    assert A.predict_linear(coef, np.array([[10.0]]))[0] == pytest.approx(25.0)


def test_quantreg_on_three_points_by_hand():
    """Punkte (0, 0), (1, 1), (2, 10), tau = 0,5: die Gerade durch (0, 0) und (2, 10) hat die Beträge 0 + 4 + 0 = 4, die durch (0, 0) und (1, 1) hat 0 + 0 + 8 = 8,
    die durch (1, 1) und (2, 10) hat 8 + 0 + 0 = 8: der Median-Fit ist y = 5 x (Achsenabschnitt 0, Steigung 5)."""
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 10.0])
    coef = A.quantreg(x, y, 0.5)
    assert coef == pytest.approx([0.0, 5.0], abs=1e-7)


def test_quantreg_with_constant_demand_returns_the_sample_quantile_as_intercept():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.full(5, 7.0)
    assert A.quantreg(x, y, 0.8) == pytest.approx([7.0, 0.0], abs=1e-7)


def test_cost_by_hand():
    """tau = 0,8 hat b = 4, h = 1. Nachfragen (10, 20), Bestellmenge 15: Überschuss 5 (Kosten 5), Mangel 5 (Kosten 20) -> Mittel 12,5."""
    assert A.cost(np.array([15.0, 15.0]), np.array([10.0, 20.0]), 0.8) == pytest.approx(12.5)
    assert A.cost(np.array([15.0, 15.0]), np.array([10.0, 20.0]), 0.5) == pytest.approx(5.0)       # b = h = 1
    assert A.cost(np.array([0.0]), np.array([3.0]), 0.95) == pytest.approx(19 * 3)


def test_cost_equals_b_plus_h_times_pinball():
    rng = np.random.default_rng(2)
    d, q = rng.uniform(0, 100, 50), rng.uniform(0, 100, 50)
    for tau, (b, h) in C.KOSTEN_BH.items():
        assert A.cost(q, d, tau) == pytest.approx((b + h) * W.pinball(q, d, tau))
        assert b / (b + h) == pytest.approx(tau)


def test_the_oracle_has_zero_extra_cost_and_a_wrong_order_quantity_costs_more():
    data = make_data("linear", 3, 50, 1, ntest=500)
    q0 = data.world.oracle_quantile(data.xt, 0.8)
    assert A.extra_cost_percent(q0, data, 0.8) == 0.0
    assert A.extra_cost_percent(q0 + 10.0, data, 0.8) > 0.0


def test_weighted_rules_on_one_dimensional_hand_data():
    """Trainingsdaten aus dem Handbeispiel; Testfall x = 3,2 mit der eigenen Nachfrage 15 (nur zum Auswerten); k wird fest vorgegeben, Handwerte wie in conftest."""
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    d = np.array([10.0, 12.0, 15.0, 30.0, 32.0])
    data = Data(World("linear", (1.0,)), 1, x, d, np.array([[3.2]]), np.array([15.0]))
    assert A.knn_rule(data, 0.8, k=3).q.tolist() == [30.0]
    assert A.knn_rule(data, 0.8, k=1).q.tolist() == [15.0]
    assert A.kernel_rule(data, 0.8, h=1.0).q.tolist() == [30.0]
    assert A.knn_rule(data, 0.8, k=3).param == 3.0


def test_leave_one_out_choice_is_a_grid_value_and_matches_the_returned_losses():
    data = make_data("quadratisch", 3, 200, 5, ntest=100)
    r = A.knn_rule(data, 0.8)
    assert r.param in W.k_grid(200) and r.param == r.grid[int(np.argmin(r.losses))]
    kr = A.kernel_rule(data, 0.8)
    assert kr.param in W.h_grid(3) and kr.param == kr.grid[int(np.argmin(kr.losses))]


def test_grid_results_agree_with_the_rules():
    data = make_data("betrag", 3, 150, 2, ntest=200)
    g = A.knn_grid(data, 0.95)
    assert g.chosen_extra == pytest.approx(A.extra_cost_percent(A.knn_rule(data, 0.95).q, data, 0.95))
    assert g.best_extra == min(g.extras) <= g.chosen_extra
    gk = A.kernel_grid(data, 0.95)
    assert gk.chosen_extra == pytest.approx(A.extra_cost_percent(A.kernel_rule(data, 0.95).q, data, 0.95))
    assert len(gk.grid) == len(gk.loo_losses) == len(gk.extras) == len(C.HGRID)


def test_quantreg_failure_is_not_swallowed(monkeypatch):
    class Failed:
        success, message, x = False, "kaputt", None

    monkeypatch.setattr(A, "linprog", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError):
        A.quantreg(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]), 0.5)
