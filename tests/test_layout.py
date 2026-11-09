"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, Instanz-Einstellungen darunter, Regler des Handbeispiels im Hauptbereich beim Experiment, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler
der App entweder in SETTING_SPECS (Instanz) oder in HAND_SPECS (Handbeispiel, gehört zu keinem Preset) steht."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gws_constants as C
import gws_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key, bereich="sidebar"):
    for art in REGLER_ARTEN:
        try:
            return getattr(getattr(at, bereich), art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in {bereich}")


def test_presets_and_instance_controls_are_in_the_sidebar_and_hand_controls_in_the_main_area():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)       # Schnellstart oben
    assert not [b for b in at.main.button if b.label in C.PRESETS]
    side = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    main = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert side == set(P.SETTING_SPECS) and main == set(P.HAND_SPECS)
    assert any(h.value == "⚙️ Einstellungen" for h in at.sidebar.header)


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_setzt_die_regler(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == C.PRESETS[name][key]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs_oder_hand_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key} | {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.ALL_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS/HAND_SPECS: ihre Änderung hebt das aktive Preset nicht auf (oder sie gehen im Permalink verloren)"
    assert set(P.ALL_SPECS) <= schluessel


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_instanzreglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    if spec.choices is not None:
        # ein anderer Wert, mit dem die Einstellungen kein Preset treffen (sonst wäre das Preset zu Recht wieder hervorgehoben)
        aktuell = {k: at.session_state[sk] for k, sk in P.PRESET_KEYS.items()}
        key = next(k for k, sk in P.PRESET_KEYS.items() if sk == state_key)
        andere = next(c for c in spec.choices if c != regler.value and not any({**aktuell, key: c} == {k: v for k, v in pr.items()} for pr in C.PRESETS.values()))
        regler.set_value(andere).run()
    else:
        regler.set_value(spec.hi if regler.value != spec.hi else spec.lo).run()
    assert not at.exception
    assert "primary" not in _typen(at).values()
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()           # erneutes Anklicken markiert wieder
    assert _typen(at)[name] == "primary"


@pytest.mark.parametrize("state_key", list(P.HAND_SPECS))
def test_die_regler_des_handbeispiels_aendern_die_instanz_nicht(state_key):
    at = _app()
    spec = P.HAND_SPECS[state_key]
    regler = _regler(at, state_key, "main")
    regler.set_value(spec.hi if regler.value != spec.hi else spec.lo).run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [next(iter(C.PRESETS))]
