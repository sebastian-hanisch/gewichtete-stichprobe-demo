"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink (Begrenzen, Einrasten) und Handbeispiel-Regler."""

import pytest

import gws_constants as C
import gws_evaluation as ev
import gws_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_choices_and_inside_the_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["welt"] in C.WELTEN and preset["p"] in C.P_OPTIONS and preset["n"] in C.N_OPTIONS and preset["tau"] in C.TAUS, name
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(**first) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_hand_controls_are_separate():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert not set(P.HAND_SPECS) & set(P.SETTING_SPECS)
    assert set(P.ALL_SPECS) == set(P.SETTING_SPECS) | set(P.HAND_SPECS)


def test_preset_names_match_their_content():
    """Namen gegen die tatsächlichen Werte (DEMO-PLAYBOOK §11 Punkt 5)."""
    quad, lin, dim, forest = (C.PRESETS[n] for n in C.PRESETS)
    assert quad["welt"] == "quadratisch" and quad["p"] == 3 and quad["n"] == 1000
    assert lin["welt"] == "linear" and lin["p"] == 3 and lin["n"] == 1000
    assert dim["p"] == 10 and dim["n"] == 100
    assert forest["welt"] == "quadratisch" and forest["p"] == 10 and forest["n"] == 100


def test_every_preset_has_a_frozen_cell():
    for preset in C.PRESETS.values():
        assert ev.frozen_cell(ev.Settings(**preset))


def test_hand_defaults_are_the_hand_example():
    assert C.HAND_DEFAULT == {"x": 3.2, "tau": 0.8, "k": 3, "h": 1.0}
    assert tuple(C.HAND_X) == (1.0, 2.0, 3.0, 4.0, 5.0) and tuple(C.HAND_D) == (10.0, 12.0, 15.0, 30.0, 32.0)


@pytest.mark.parametrize(
    "key, raw, expected",
    [
        ("hand_x", 9.0, 5.0), ("hand_x", -2.0, 1.0), ("hand_x", 3.26, 3.3), ("hand_tau", 0.83, 0.85), ("hand_tau", 0.1, 0.5),
        ("hand_k", 99, 5), ("hand_k", 0, 1), ("hand_h", 0.34, 0.3), ("seed_input", 10**9, C.SEED_MAX), ("seed_input", -5, 0),
        ("welt_sel", "betrag", "betrag"), ("welt_sel", "gibt-es-nicht", None), ("p_sel", 4, None), ("p_sel", 10, 10), ("n_sel", 1000, 1000), ("tau_sel", 0.9, None), ("tau_sel", 0.8, 0.8),
    ],
)
def test_permalink_values_are_clamped_snapped_and_invalid_choices_dropped(key, raw, expected):
    got = P.snap(P.ALL_SPECS[key], raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_every_default_lies_on_its_own_grid():
    for key, spec in P.ALL_SPECS.items():
        assert P.snap(spec, spec.default) == spec.default, key


def test_next_seed_comes_from_splitmix64_and_stays_in_range():
    """Der Knopf „Neue Instanz“ wählt den Seed aus SplitMix64 (kein Python-`random`): im Bereich, nie gleich dem alten, bei gleichem Ausgangswert derselbe, und die Kette wiederholt sich nicht gleich."""
    assert not hasattr(P, "random")
    seed, chain = C.DEFAULT_SEED, []
    for _ in range(30):
        new = P.next_seed(seed)
        assert 0 <= new <= C.SEED_MAX and new != seed and new == P.next_seed(seed)
        chain.append(new)
        seed = new
    assert len(set(chain)) == len(chain)
