"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Zwei Reglergruppen:
  SETTING_SPECS  (Seitenleiste) die Instanz-Einstellungen (Welt, p, n, tau, Zufalls-Seed) für die Experimente 2 bis 4; ein Preset legt sie fest, ihre Änderung hebt die Hervorhebung auf
  HAND_SPECS     (Hauptbereich beim Experiment) die Regler des Handbeispiels (Experiment 1: Anfrage x, tau, k, Bandbreite); sie gehören zu keinem Preset und hängen nicht an der Instanz
Der Permalink in der Adresszeile umfasst beide Gruppen."""

from dataclasses import dataclass

import streamlit as st

import gws_constants as C
from gws_rng import uniform_block


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None       # Einrasten: Werte aus der Adresszeile werden auf das Raster des Reglers gelegt
    choices: tuple | None = None    # Auswahlregler: nur diese Werte sind gültig


SETTING_SPECS = {
    "welt_sel": SettingSpec("welt", str, C.DEFAULT_WELT, choices=C.WELTEN),
    "p_sel": SettingSpec("p", int, C.DEFAULT_P, choices=C.P_OPTIONS),
    "n_sel": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_OPTIONS),
    "tau_sel": SettingSpec("tau", float, C.DEFAULT_TAU, choices=C.TAUS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
HAND_SPECS = {
    "hand_x": SettingSpec("hx", float, C.HAND_DEFAULT["x"], C.HAND_X_MIN, C.HAND_X_MAX, 0.1),
    "hand_tau": SettingSpec("ht", float, C.HAND_DEFAULT["tau"], C.HAND_TAU_MIN, C.HAND_TAU_MAX, 0.05),
    "hand_k": SettingSpec("hk", int, C.HAND_DEFAULT["k"], C.HAND_K_MIN, C.HAND_K_MAX, 1),
    "hand_h": SettingSpec("hh", float, C.HAND_DEFAULT["h"], C.HAND_H_MIN, C.HAND_H_MAX, 0.1),
}
ALL_SPECS = {**SETTING_SPECS, **HAND_SPECS}
PRESET_KEYS = {"welt": "welt_sel", "p": "p_sel", "n": "n_sel", "tau": "tau_sel", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in ALL_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = ALL_SPECS[state_key]
    return spec.lo, spec.hi


def snap(spec: SettingSpec, value):
    """Wert auf den Regler legen: Auswahl nur aus den Optionen (sonst None), Zahl auf die Grenzen begrenzen und auf das Raster legen."""
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.caster is float:
        value = round(round((value - spec.lo) / spec.step) * spec.step + spec.lo, 10)
        value = min(spec.hi, max(spec.lo, value))
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet (ungültige Auswahl: Standard)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in ALL_SPECS.items():
        if spec.url_param in qp:
            try:
                value = snap(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[ALL_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Instanz-Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Die Regler des Handbeispiels (HAND_SPECS) gehören nicht dazu: sie hängen nicht an der Instanz."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def next_seed(seed: int) -> int:
    """Der nächste Zufalls-Seed aus SplitMix64 (erstes Element des Stroms zum aktuellen Zufalls-Seed), nie gleich dem aktuellen; kein Python-`random`."""
    new = int(uniform_block(int(seed), 1)[0] * (C.SEED_MAX + 1))
    return new if new != seed else (new + 1) % (C.SEED_MAX + 1)


def randomize_seed():
    st.session_state["seed_input"] = next_seed(int(st.session_state.get("seed_input", C.DEFAULT_SEED)))


def reset_hand():
    for state_key, spec in HAND_SPECS.items():
        st.session_state[state_key] = spec.default
