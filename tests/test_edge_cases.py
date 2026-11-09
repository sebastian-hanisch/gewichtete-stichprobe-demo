"""Randfälle, an denen die Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, n < k, alle x gleich, Gleichstände, Gewichte mit Nullen, Grenzen inklusive.

Die Einheiten der Gewichte (Gleichstände, k > n, winzige Bandbreite) stehen in tests/test_weights.py; hier die Regeln und die Instanzen darüber."""

import numpy as np
import pytest

import gws_algorithm as A
import gws_constants as C
import gws_evaluation as ev
from gws_scenario import Data, World, make_data


def _data(x, d, xt, dt):
    x, xt = np.array(x, float).reshape(len(x), -1), np.array(xt, float).reshape(len(xt), -1)
    return Data(World("linear", tuple([1.0] + [0.0] * (x.shape[1] - 1))), x.shape[1], x, np.array(d, float), xt, np.array(dt, float))


def test_exactly_two_training_cases_every_rule_runs():
    """n = 2: der kleinste Fall mit Leave-one-out (jeder Fall hat genau einen Nachbarn); k und Bandbreite kommen aus dem Raster."""
    data = _data([[0.0], [1.0]], [4.0, 6.0], [[0.2], [0.9]], [5.0, 5.0])
    kn, ke = A.knn_rule(data, 0.8), A.kernel_rule(data, 0.8)
    assert kn.q.shape == (2,) and ke.q.shape == (2,)
    assert set(kn.q) <= {4.0, 6.0} and set(ke.q) <= {4.0, 6.0}
    assert kn.q.tolist() == [6.0, 6.0]       # k wird auf 2 Fälle begrenzt: gleiche Gewichte, 0,8-Quantil von (4, 6) ist 6


def test_single_training_case_gives_its_demand_everywhere():
    data = _data([[0.0]], [9.0], [[-3.0], [4.0]], [8.0, 10.0])
    assert A.knn_rule(data, 0.95).q.tolist() == [9.0, 9.0]
    assert A.kernel_rule(data, 0.5).q.tolist() == [9.0, 9.0]


def test_k_larger_than_n_is_clipped_not_an_error():
    data = _data([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0], [[1.0]], [2.0])
    assert A.knn_rule(data, 0.5, k=50).q.tolist() == [2.0]     # alle drei mit Gewicht 1/3: Median 2


def test_all_features_equal_gives_the_unweighted_quantile_for_the_kernel_and_the_first_k_for_knn():
    data = _data([[1.0]] * 5, [5.0, 1.0, 4.0, 2.0, 3.0], [[1.0]], [3.0])
    assert A.kernel_rule(data, 0.6, h=0.5).q.tolist() == [3.0]        # gleiche Gewichte: 0,6-Quantil von 1..5
    assert A.knn_rule(data, 0.6, k=2).q.tolist() == [5.0]             # die zwei ersten Nummern (Nachfragen 5, 1): 0,6-Quantil von (1, 5) ist 5


def test_constant_demand_gives_that_demand_for_every_rule():
    data = _data([[0.0], [1.0], [2.0], [3.0]], [7.0] * 4, [[0.5], [2.5]], [7.0, 7.0])
    for tau in C.TAUS:
        assert A.knn_rule(data, tau).q.tolist() == [7.0, 7.0]
        assert A.kernel_rule(data, tau).q.tolist() == [7.0, 7.0]
        assert A.kq_empirical_rule(data, tau) == pytest.approx([7.0, 7.0])


def test_kernel_far_from_every_training_case_still_returns_a_demand():
    """Testfall weit außerhalb: alle Gewichte wären numerisch 0; der nächste Fall behält das Gewicht 1 (kein 0/0)."""
    data = _data([[0.0], [1.0]], [4.0, 6.0], [[1e6]], [6.0])
    q = A.kernel_rule(data, 0.5, h=0.3).q
    assert not np.isnan(q).any() and q.tolist() == [6.0]


def test_demand_is_never_negative_and_the_cost_ratio_is_defined_for_every_tau():
    data = make_data("linear", 3, 30, 3, ntest=50)
    assert (data.d >= 0).all()
    for tau in C.TAUS:
        assert A.extra_cost_percent(A.knn_rule(data, tau).q, data, tau) > -100.0


def test_unknown_tau_is_rejected_not_guessed():
    data = make_data("linear", 3, 30, 3, ntest=20)
    with pytest.raises(KeyError):
        A.cost(data.dt, data.dt, 0.77)


def test_slider_bounds_are_inclusive_for_the_hand_example():
    lo = ev.hand_example(C.HAND_X_MIN, C.HAND_TAU_MIN, C.HAND_K_MIN, C.HAND_H_MIN)
    hi = ev.hand_example(C.HAND_X_MAX, C.HAND_TAU_MAX, C.HAND_K_MAX, C.HAND_H_MAX)
    assert lo.q_knn == 10.0                               # x = 1, k = 1: der Fall x = 1
    assert hi.q_knn == 32.0                               # x = 5, k = 5: alle fünf Fälle, 0,95-Quantil 32
    assert abs(sum(lo.w_knn) - 1) < 1e-12 and abs(sum(hi.w_kern) - 1) < 1e-12


def test_hand_example_distance_ties_use_the_smaller_index():
    """Anfrage x = 3,0 und k = 2: Abstände 2, 1, 0, 1, 2; die zwei nächsten sind x = 3 und (Gleichstand 2 gegen 4) x = 2."""
    r = ev.hand_example(3.0, 0.8, 2, 1.0)
    assert r.w_knn == (0.0, 0.5, 0.5, 0.0, 0.0)
