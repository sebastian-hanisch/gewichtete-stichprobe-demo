"""Jede Zahl der README: Zahl der Vormessung (eingefroren in gws_vormessung.json) oder Messung dieser Demo (feste Seeds, als Band oder Ungleichung).

Zwei Arten von Zahlen (DEMO-PLAYBOOK §11a Punkt 13):
  Vormessung   30 Seeds, 5000 Testfälle, Wald mit scikit-learn: die Zahlen stehen als exakte Werte in der JSON-Datei und werden hier exakt geprüft. Sie stammen aus
               vormessung/out_eo_v3_gewichtet.txt und vormessung/out_eo_v3b_kern.txt des Planungsordners (nicht im Repo) und werden in der Demo nie live gerechnet.
  Live         Seeds SWEEP_SEEDS (fünf feste Instanzen), 1000 Testfälle, numpy: nur Ungleichungen mit großem Abstand und Bänder, weil trainierte Zahlen (Leave-one-out-Wahl)
               plattformabhängig sein können."""

import json
from collections import Counter
from pathlib import Path

import pytest

import gws_constants as C
import gws_evaluation as ev

F = ev.frozen()


def _med(welt, p, n, tau, m):
    return F[(welt, p, n, tau)][m]["median"]


# --- Vormessung: Zellen und Aufbau ------------------------------------------------------------------------------------------------------------

def test_frozen_file_has_48_cells_each_with_five_methods_and_consistent_bands():
    assert len(F) == 48 == len(C.WELTEN) * len(C.P_OPTIONS) * len(C.N_OPTIONS) * len(C.TAUS)
    for key, c in F.items():
        assert key[0] in C.WELTEN
        for m in ev.METHODEN:
            assert c[m]["min"] <= c[m]["median"] <= c[m]["max"], (key, m)
        assert min(C.KGRID) <= c["k_median"] <= max(C.KGRID)     # Median der gewählten k über 30 Seeds: kann zwischen zwei Rasterwerten liegen (z. B. 60)
    raw = json.loads((Path(__file__).resolve().parent.parent / C.VORMESSUNG_DATEI).read_text(encoding="utf-8"))
    assert raw["seeds"] == C.VORMESSUNG_SEEDS == 30 and raw["testfaelle"] == C.VORMESSUNG_NTEST == 5000
    assert "Wald" in raw["quelle"] and "nicht live" in raw["quelle"]


# --- Vormessung: README-Tabelle „Befunde“ -----------------------------------------------------------------------------------------------------

def test_quadratic_truth_n1000_tau95_p3_and_p10():
    """README: Wahrheit quadratisch, n = 1000, τ = 0,95 (Median über 30 Seeds, Mehrkosten in %)."""
    got = {m: (_med("quadratisch", 3, 1000, 0.95, m), _med("quadratisch", 10, 1000, 0.95, m)) for m in ev.METHODEN}
    assert got == {"kq": (134.9, 140.0), "direkt": (136.5, 148.1), "knn": (30.4, 81.1), "wald": (29.7, 32.9), "kern": (29.6, 80.8)}


def test_linear_truth_n1000_tau95_p3_and_p10():
    got = {m: (_med("linear", 3, 1000, 0.95, m), _med("linear", 10, 1000, 0.95, m)) for m in ev.METHODEN}
    assert got == {"kq": (0.3, 0.7), "direkt": (0.6, 2.2), "knn": (9.3, 23.2), "wald": (12.4, 19.3), "kern": (9.1, 24.8)}


def test_other_two_worlds_n1000_tau95():
    assert {m: (_med("streuung_linear", 3, 1000, 0.95, m), _med("streuung_linear", 10, 1000, 0.95, m)) for m in ev.METHODEN} == {
        "kq": (32.5, 33.1), "direkt": (7.5, 9.7), "knn": (18.6, 44.4), "wald": (23.1, 41.4), "kern": (22.4, 45.5)}
    assert {m: (_med("betrag", 3, 1000, 0.95, m), _med("betrag", 10, 1000, 0.95, m)) for m in ev.METHODEN} == {
        "kq": (19.3, 19.0), "direkt": (20.7, 21.8), "knn": (12.1, 29.5), "wald": (17.0, 28.1), "kern": (14.3, 32.8)}


def test_kernel_beats_knn_at_quadratic_p3_n1000_for_tau_05_and_08():
    assert (_med("quadratisch", 3, 1000, 0.5, "kern"), _med("quadratisch", 3, 1000, 0.8, "kern")) == (14.6, 16.5)
    assert (_med("quadratisch", 3, 1000, 0.5, "knn"), _med("quadratisch", 3, 1000, 0.8, "knn")) == (22.2, 23.5)


def test_kernel_and_knn_differ_by_at_most_four_points_in_most_cells():
    """Zählung für „Kern ungefähr kNN“: in wie vielen der 48 Zellen liegen die Mediane höchstens 4 Prozentpunkte auseinander, und wo nicht."""
    far = sorted((k for k, c in F.items() if abs(c["knn"]["median"] - c["kern"]["median"]) > 4.0))
    assert len(far) == 9 and {k[0] for k in far} == {"quadratisch", "streuung_linear"}
    assert Counter(k[0] for k in far) == {"quadratisch": 7, "streuung_linear": 2}


def test_p10_n100_weights_lose_between_31_and_75_percent_in_three_worlds():
    """README: p = 10, n = 100, Welten linear, Streuung linear und |x₁|, τ = 0,5 / 0,8 / 0,95: kNN- und Wald-Gewichte liegen bei 31,4 bis 75,0 %."""
    vals = [_med(w, 10, 100, t, m) for w in ("linear", "streuung_linear", "betrag") for t in C.TAUS for m in ("knn", "wald")]
    assert len(vals) == 18 and (min(vals), max(vals)) == (31.4, 75.0)


def test_quadratic_p3_n100_tau95_and_p10_n100_tau95():
    assert (_med("quadratisch", 3, 100, 0.95, "kq"), _med("quadratisch", 3, 100, 0.95, "knn"), _med("quadratisch", 3, 100, 0.95, "wald")) == (146.2, 85.7, 89.4)
    assert (_med("quadratisch", 10, 100, 0.95, "kq"), _med("quadratisch", 10, 100, 0.95, "knn"), _med("quadratisch", 10, 100, 0.95, "wald")) == (149.8, 150.9, 95.5)


def test_count_of_cells_where_each_side_wins_by_the_rule_of_the_verdict():
    """README: In 36 von 48 Zellen liegt die beste parametrische Regel vorn, in 9 die besten Gewichte (kNN oder Kern; 8 quadratische Welt, 1 nicht monotone Streuung), 3 gleichauf (quadratisch, p = 10).
    Regel: Unterschied größer als max(1 Prozentpunkt, 5 % des größeren Werts) wie im Verdikt der App."""
    kinds = {key: ev.verdict_welt(ev.Settings(*key), c)[0] for key, c in F.items()}
    assert Counter(kinds.values()) == {"warning": 36, "success": 9, "info": 3}
    wins = [k for k, v in kinds.items() if v == "success"]
    assert Counter(k[0] for k in wins) == {"quadratisch": 8, "betrag": 1}
    draws = [k for k, v in kinds.items() if v == "info"]
    assert {(k[0], k[1]) for k in draws} == {("quadratisch", 10)}
    assert wins and ("betrag", 3, 1000, 0.95) in wins


def test_forest_is_clearly_best_in_six_cells_all_quadratic_with_p10():
    """README: Der Wald liegt in 6 von 48 Zellen um mehr als die Gleichauf-Grenze unter allen anderen vier Regeln; alle 6 sind quadratisch mit p = 10."""
    cells = []
    for key, c in F.items():
        m = ev.frozen_medians(c)
        best_other = min(m[x] for x in ("kq", "direkt", "knn", "kern"))
        if m["wald"] < best_other - ev.tolerance(m["wald"], best_other):
            cells.append(key)
    assert len(cells) == 6 and {(k[0], k[1]) for k in cells} == {("quadratisch", 10)}
    assert Counter(k[2] for k in cells) == {100: 3, 1000: 3}


def test_lowest_k_median_is_five_and_highest_is_eighty():
    ks = [c["k_median"] for c in F.values()]
    assert min(ks) == 5 and max(ks) == 80


# --- Live (fünf feste Instanzen): Ungleichungen -----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dim_quadratic():
    return ev.sweep_p("quadratisch", 100, 0.95)


@pytest.fixture(scope="module")
def dim_linear_1000():
    return ev.sweep_p("linear", 1000, 0.95)


def test_live_linear_truth_parametric_rule_stays_near_the_oracle_while_the_weights_rise_with_p(dim_linear_1000):
    rows = {r["p"]: r for r in dim_linear_1000}
    assert all(abs(r["kq"]["median"]) < 1.0 for r in rows.values())                      # KQ + empirisch: höchstens ein Prozent von den Mehrkosten 0 entfernt
    knn = [rows[p]["knn"]["median"] for p in C.P_SWEEP]
    assert knn[0] < 6.0 and knn[-1] > 15.0 and all(a < b for a, b in zip(knn, knn[1:]))   # p = 1 niedrig, p = 10 hoch, dazwischen steigend
    assert rows[10]["kern"]["median"] > 15.0


def test_live_quadratic_truth_n100_weights_win_at_p_up_to_3_and_lose_that_edge_at_p10(dim_quadratic):
    rows = {r["p"]: r for r in dim_quadratic}
    for p in (1, 2, 3):
        assert rows[p]["knn"]["median"] < 0.6 * rows[p]["kq"]["median"], p
        assert rows[p]["kern"]["median"] < 0.6 * rows[p]["kq"]["median"], p
    assert rows[10]["knn"]["median"] > 0.8 * rows[10]["kq"]["median"]


def test_live_quadratic_truth_n1000_p3_weights_cost_less_than_half_of_the_linear_rule():
    s = ev.Settings("quadratisch", 3, 1000, 0.95)
    rows = {r["p"]: r for r in ev.sweep_p(s.welt, s.n, s.tau, ps=(3,))}
    assert rows[3]["knn"]["median"] < 0.5 * rows[3]["kq"]["median"] and rows[3]["kern"]["median"] < 0.5 * rows[3]["kq"]["median"]
    assert rows[3]["direkt"]["median"] > 0.9 * rows[3]["kq"]["median"]                 # direkt in der falschen Klasse hilft nicht


def test_live_choice_experiment_leave_one_out_is_never_below_the_hindsight_best_and_close_to_it():
    res = ev.choice_experiment("quadratisch", 3, 1000, 0.95)
    for name in ("knn", "kern"):
        r = res[name]
        assert r["loo"] >= r["best"] - 1e-9
        assert r["loo"] - r["best"] < 6.0                                                # beobachtet 1,3 und 0,0 Prozentpunkte
        assert max(r["fixed"]) > 2.5 * min(r["fixed"])                                   # feste Wahl am Rand des Rasters ist deutlich schlechter
        assert min(r["fixed"]) >= r["best"] - 1e-9
        assert len(r["chosen_grid_index"]) == len(C.SWEEP_SEEDS)
    assert len(res["knn"]["grid"]) == len(C.KGRID) and len(res["kern"]["grid"]) == len(C.HGRID)


def test_live_instance_of_the_two_headline_presets():
    quad = ev.analyse(ev.Settings(**{k: C.PRESETS["Quadratische Wahrheit: Gewichte gewinnen"][k] for k in ("welt", "p", "n", "tau", "seed")}))
    assert quad.live["knn"] < 0.4 * quad.live["kq"] and quad.live["kern"] < 0.4 * quad.live["kq"]
    lin = ev.analyse(ev.Settings(**{k: C.PRESETS["Lineare Wahrheit: Gewichte zahlen"][k] for k in ("welt", "p", "n", "tau", "seed")}))
    assert lin.live["kq"] < 2.0 < 5.0 < lin.live["knn"] and lin.live["kern"] > 5.0


def test_no_scikit_learn_at_runtime():
    """Kein scikit-learn zur Laufzeit: weder in den Modulen noch in requirements.txt."""
    root = Path(__file__).resolve().parent.parent
    for path in list(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "import sklearn" not in text and "from sklearn" not in text, path.name
    assert "scikit" not in (root / "requirements.txt").read_text(encoding="utf-8").lower()
