"""Welten und Daten: Formeln von Hand an festen Punkten, Reproduzierbarkeit, Stutzung bei 0."""

import numpy as np
import pytest
from scipy.stats import norm

import gws_constants as C
from gws_scenario import World, make_data, make_world

Z95 = float(norm.ppf(0.95))


def test_mean_and_spread_by_hand_in_every_world():
    """b = (1, 0, 0): mu = 60 + 10 * x1 (+ 8 * (x1^2 - 1) quadratisch); sigma konstant 8, max(1, 6 + 4 x1) beziehungsweise 4 + 6 |x1|."""
    b = (1.0, 0.0, 0.0)
    x = np.array([[2.0, 5.0, -1.0], [-3.0, 0.0, 0.0]])
    lin = World("linear", b)
    assert lin.mu(x).tolist() == [80.0, 30.0] and lin.sigma(x).tolist() == [8.0, 8.0]
    quad = World("quadratisch", b)
    assert quad.mu(x).tolist() == [60 + 20 + 8 * 3, 60 - 30 + 8 * 8]            # 104 und 94
    assert quad.sigma(x).tolist() == [8.0, 8.0]
    sl = World("streuung_linear", b)
    assert sl.sigma(x).tolist() == [14.0, 1.0]                                  # 6 + 8 und max(1, 6 - 12)
    assert sl.mu(x).tolist() == lin.mu(x).tolist()
    ab = World("betrag", b)
    assert ab.sigma(x).tolist() == [16.0, 22.0]                                 # 4 + 12 und 4 + 18


def test_oracle_quantile_is_the_truncated_normal_quantile():
    w = World("linear", (1.0, 0.0))
    x = np.array([[0.0, 0.0], [-10.0, 0.0]])
    assert w.oracle_quantile(x, 0.5).tolist() == [60.0, 0.0]                    # zweiter Fall: Mittel -40, das Quantil 0,5 von max(0, Y) ist 0
    assert w.oracle_quantile(x, 0.95)[0] == pytest.approx(60.0 + 8.0 * Z95)


def test_data_are_reproducible_nonnegative_and_have_the_requested_shapes():
    d1 = make_data("quadratisch", 4, 50, 7, ntest=30)
    d2 = make_data("quadratisch", 4, 50, 7, ntest=30)
    assert d1.x.shape == (50, 4) and d1.d.shape == (50,) and d1.xt.shape == (30, 4) and d1.dt.shape == (30,)
    assert np.array_equal(d1.x, d2.x) and np.array_equal(d1.d, d2.d) and np.array_equal(d1.dt, d2.dt)
    assert (d1.d >= 0).all() and (d1.dt >= 0).all()
    assert not np.array_equal(d1.x, make_data("quadratisch", 4, 50, 8, ntest=30).x)


def test_direction_vector_has_length_one_and_depends_only_on_world_seed_and_p():
    for seed in (0, 1, 99999):
        beta = np.array(make_world("linear", 5, seed).beta)
        assert np.linalg.norm(beta) == pytest.approx(1.0)
    assert make_world("linear", 3, 4).beta == make_world("quadratisch", 3, 4).beta


def test_training_features_are_roughly_standard_normal():
    d = make_data("linear", 3, 5000, 3, ntest=10)
    assert np.abs(d.x.mean(axis=0)).max() < 0.06 and np.abs(d.x.std(axis=0) - 1.0).max() < 0.06


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        make_world("unbekannt", 3, 1)
    with pytest.raises(ValueError):
        make_world("linear", 0, 1)
    with pytest.raises(ValueError):
        make_data("linear", 3, 0, 1)


def test_all_world_names_have_labels_and_a_name_in_the_frozen_measurement():
    assert set(C.WELTEN) == set(C.WELT_NAMEN) == set(C.WELT_IN_VORMESSUNG)
