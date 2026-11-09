"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider an seine Grenzen und stellt jede Auswahl
(Selectbox, Select-Slider) auf jeden Wert. Prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei Plotly-Diagrammen ohne key= und
StreamlitAPIException bei Slidern mit berechneten Grenzen), dass die Verdikte erscheinen und dass der Permalink die Regler setzt."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import gws_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception_and_shows_all_four_verdicts():
    at = _fresh_app()
    texts = [e.value for e in list(at.success) + list(at.info) + list(at.warning)]
    assert any(t.startswith("Mit k = 3 Nachbarn bestellt") for t in texts)                     # Experiment 1
    assert any(t.startswith("In der Vormessung (Median über 30 Seeds") for t in texts)         # Kernfrage / Experiment 2
    assert any(t.startswith("Von p = 1 auf p = 10") for t in texts)                            # Experiment 3
    assert any(t.startswith("Die Wahl per Leave-one-out kostet") for t in texts)               # Experiment 4


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("welt", C.WELTEN)
@pytest.mark.parametrize("n", C.N_OPTIONS)
def test_every_world_and_data_size_runs(welt, n):
    at = _fresh_app()
    at.sidebar.selectbox(key="welt_sel").set_value(welt)
    at.sidebar.select_slider(key="n_sel").set_value(n)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (welt, n, [str(e) for e in at.exception])


@pytest.mark.parametrize("p", C.P_OPTIONS)
@pytest.mark.parametrize("tau", C.TAUS)
def test_every_dimension_and_tau_runs(p, tau):
    at = _fresh_app()
    at.sidebar.select_slider(key="p_sel").set_value(p)
    at.sidebar.select_slider(key="tau_sel").set_value(tau)
    at.sidebar.select_slider(key="n_sel").set_value(100)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (p, tau, [str(e) for e in at.exception])


def test_permalink_sets_the_controls_and_is_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["welt"] = "linear"
    at.query_params["n"] = "100"
    at.query_params["p"] = "7"            # keine gültige Auswahl: Standard
    at.query_params["hk"] = "99"          # über der Grenze: begrenzt auf 5
    at.query_params["hx"] = "3.26"        # auf das Raster 0,1 gelegt
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["welt_sel"] == "linear" and at.session_state["n_sel"] == 100
    assert at.session_state["p_sel"] == C.DEFAULT_P
    assert at.session_state["hand_k"] == 5 and at.session_state["hand_x"] == pytest.approx(3.3)


def test_hand_example_reset_button_restores_the_defaults():
    at = _fresh_app()
    at.slider(key="hand_k").set_value(5).run()
    at.slider(key="hand_x").set_value(1.0).run()
    reset = next(b for b in at.main.button if b.label == "Zurücksetzen")
    reset.click().run()
    assert at.session_state["hand_k"] == 3 and at.session_state["hand_x"] == pytest.approx(3.2)
