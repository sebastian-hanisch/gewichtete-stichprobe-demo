"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds."""

import pytest

import gws_evaluation as ev


def _band(m):
    return {"median": m, "min": m - 1.0, "max": m + 1.0}


def _cell(kq, direkt, knn, kern, wald):
    return {"kq": _band(kq), "direkt": _band(direkt), "knn": _band(knn), "kern": _band(kern), "wald": _band(wald), "k_median": 20}


S = ev.Settings("quadratisch", 3, 1000, 0.95, 35)


def test_de_uses_a_decimal_comma_and_never_prints_negative_zero():
    assert ev.de(30.44) == "30,4" and ev.de(0.95, 2) == "0,95" and ev.de(-0.04) == "0,0" and ev.de(-0.14) == "-0,1"
    assert ev.de_signed(22.14) == "+22,1" and ev.de_signed(-3.26) == "-3,3" and ev.de_signed(0.01) == "0,0"


def test_tolerance_is_one_point_or_five_percent_of_the_larger_value():
    assert ev.tolerance(0.3, 0.7) == 1.0
    assert ev.tolerance(20.0, 10.0) == 1.0                  # 5 % von 20 sind genau 1,0
    assert ev.tolerance(150.0, 146.7) == pytest.approx(7.5)


def test_verdict_welt_success_when_weights_win():
    kind, text = ev.verdict_welt(S, _cell(kq=134.9, direkt=136.5, knn=30.4, kern=29.6, wald=29.7))
    assert kind == "success"
    assert "**29,6 %**" in text and "Gauß-Kerngewichte" in text and "**134,9 %**" in text and "KQ + empirisches Quantil" in text
    assert "105,3 Prozentpunkte darunter" in text and "Wald-Gewichte (Vormessung mit scikit-learn, nicht live): 29,7 %." in text
    assert "deutlich darunter" not in text and "τ = 0,95" in text and "p = 3, n = 1000" in text


def test_verdict_welt_warning_when_the_parametric_rule_wins():
    kind, text = ev.verdict_welt(ev.Settings("linear", 10, 1000, 0.95, 35), _cell(kq=0.7, direkt=2.2, knn=23.2, kern=24.8, wald=19.3))
    assert kind == "warning"
    assert "(kNN-Gewichte) **23,2 %**" in text and "(KQ + empirisches Quantil) **0,7 %**" in text
    assert "22,5 Prozentpunkte darunter (bei dieser Datenmenge und Dimension)" in text


def test_verdict_welt_picks_the_better_of_the_two_parametric_rules_and_the_two_weights():
    kind, text = ev.verdict_welt(ev.Settings("streuung_linear", 3, 1000, 0.95, 35), _cell(kq=32.5, direkt=7.5, knn=18.6, kern=22.4, wald=23.1))
    assert kind == "warning" and "(direkt (Quantilregression)) **7,5 %**" in text and "(kNN-Gewichte) **18,6 %**" in text


def test_verdict_welt_threshold_is_exclusive_at_the_tolerance():
    # Unterschied genau 1,0 bei kleinen Werten: gleichauf; knapp darüber: Entscheidung
    assert ev.verdict_welt(S, _cell(kq=5.0, direkt=9.0, knn=4.0, kern=8.0, wald=7.0))[0] == "info"
    assert ev.verdict_welt(S, _cell(kq=5.0, direkt=9.0, knn=3.9, kern=8.0, wald=7.0))[0] == "success"
    assert ev.verdict_welt(S, _cell(kq=4.0, direkt=9.0, knn=5.0, kern=8.0, wald=7.0))[0] == "info"
    assert ev.verdict_welt(S, _cell(kq=3.9, direkt=9.0, knn=5.0, kern=8.0, wald=7.0))[0] == "warning"


def test_verdict_welt_ties_in_large_values_are_a_draw_and_name_a_clearly_better_forest():
    kind, text = ev.verdict_welt(ev.Settings("quadratisch", 10, 100, 0.95, 35), _cell(kq=149.8, direkt=214.5, knn=150.9, kern=146.7, wald=95.5))
    assert kind == "info" and "Der Unterschied von 3,1 Prozentpunkten liegt innerhalb der Gleichauf-Grenze" in text
    assert "95,5 %, deutlich darunter." in text


def _rows(p_lo, p_hi, kq, knn, kern):
    def row(p, a, b, c):
        return {"p": p, "kq": _band(a), "direkt": _band(a), "knn": _band(b), "kern": _band(c)}
    return [row(p_lo, kq[0], knn[0], kern[0]), row(p_hi, kq[1], knn[1], kern[1])]


def test_verdict_dim_warns_when_the_weights_rise_more_than_the_parametric_rule():
    kind, text = ev.verdict_dim(_rows(1, 10, kq=(1.2, 11.9), knn=(21.1, 46.6), kern=(15.0, 46.1)))
    assert kind == "warning"
    assert "Von p = 1 auf p = 10" in text and "von 21,1 % auf 46,6 % (Änderung +25,5 Prozentpunkte)" in text
    assert "der Gauß-Kerngewichte um +31,1 Prozentpunkte" in text and "von 1,2 % auf 11,9 % (Änderung +10,7 Prozentpunkte)" in text
    assert "stärker als die parametrische Regel" in text
    assert "Bei p = 10 liegen die kNN-Gewichte 34,7 Prozentpunkte über KQ + empirischem Quantil." in text
    kind2, text2 = ev.verdict_dim(_rows(1, 10, kq=(100.0, 140.0), knn=(15.0, 85.0), kern=(3.0, 83.0)))
    assert kind2 == "warning" and "Bei p = 10 liegen die kNN-Gewichte 55,0 Prozentpunkte unter KQ + empirischem Quantil." in text2


def test_verdict_dim_threshold_is_one_point_and_the_other_case_is_info():
    assert ev.verdict_dim(_rows(1, 10, kq=(0.0, 10.0), knn=(0.0, 11.0), kern=(0.0, 0.0)))[0] == "info"       # Anstieg 11 gegen 10: genau 1,0 Unterschied
    assert ev.verdict_dim(_rows(1, 10, kq=(0.0, 10.0), knn=(0.0, 11.1), kern=(0.0, 0.0)))[0] == "warning"
    kind, text = ev.verdict_dim(_rows(1, 10, kq=(0.0, 30.0), knn=(5.0, 10.0), kern=(0.0, 0.0)))
    assert kind == "info" and "mindestens so stark" in text
    assert "(Änderung 0,0 Prozentpunkte)" in ev.verdict_dim(_rows(1, 10, kq=(5.0, 5.0), knn=(0.0, 0.0), kern=(0.0, 0.0)))[1]


def test_verdict_wahl_names_both_gaps():
    res = {"knn": {"loo": 26.3, "best": 25.0}, "kern": {"loo": 22.5, "best": 22.5}}
    kind, text = ev.verdict_wahl(res)
    assert kind == "info"
    assert "26,3 % Mehrkosten (kNN) beziehungsweise 22,5 % (Kern)" in text and "25,0 % beziehungsweise 22,5 %" in text
    assert "Abstand von 1,3 beziehungsweise 0,0 Prozentpunkten" in text


def test_verdict_hand_names_quantiles_mean_and_coverage():
    kind, text = ev.verdict_hand(ev.hand_example(3.2, 0.8, 3, 1.0))
    assert kind == "info"
    assert "Mit k = 3 Nachbarn bestellt das gewichtete 0,80-Quantil **30**" in text and "(Bandbreite 1,0) **30**" in text
    assert "Das gewichtete Mittel der Nachbarn (19,0) deckt nur 66,7 % des Gewichts; verlangt sind mindestens 80 %" in text


def test_hand_example_results_by_hand():
    r = ev.hand_example(3.2, 0.8, 3, 1.0)
    assert r.q_knn == 30.0 and r.q_kern == 30.0 and r.mean_knn == pytest.approx(19.0) and r.mean_all == pytest.approx(19.8)
    assert r.cover_mean_knn == pytest.approx(2 / 3)                         # Mittel 19: gedeckt sind die Nachfragen 12 und 15 (Gewicht 1/3 + 1/3)
    r1 = ev.hand_example(3.2, 0.5, 1, 1.0)
    assert r1.q_knn == 15.0 and r1.cover_mean_knn == 1.0                    # ein Nachbar: Mittel = 15, vollständig gedeckt
    cdf = ev.hand_cdf(r.w_knn)
    assert [d for d, _ in cdf] == [10.0, 12.0, 15.0, 30.0, 32.0]
    assert [round(c, 4) for _, c in cdf] == [0.0, 0.3333, 0.6667, 1.0, 1.0]


def test_hand_cdf_merges_equal_demands(monkeypatch):
    import gws_constants as C
    monkeypatch.setattr(C, "HAND_D", (10.0, 10.0, 15.0, 30.0, 30.0))
    assert ev.hand_cdf((0.1, 0.2, 0.3, 0.2, 0.2)) == [(10.0, pytest.approx(0.3)), (15.0, pytest.approx(0.6)), (30.0, pytest.approx(1.0))]
