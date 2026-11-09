"""SplitMix64: die vektorisierte Fassung gegen eine Schleife mit Python-Ganzzahlen (unabhängig, kein numpy)."""

import numpy as np
import pytest

from gws_rng import normal_block, uniform_block

MASK = (1 << 64) - 1


def splitmix_reference(seed: int, count: int) -> list[int]:
    """Der Strom von SplitMix64 mit Python-Ganzzahlen: state += gamma; Ausgabe = Mischung des neuen Zustands."""
    state, out = seed & MASK, []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


def test_known_first_outputs_of_the_reference_generator():
    """Bekannte Ausgaben von SplitMix64 mit Seed 1234567 (häufig zitierter Prüfvektor)."""
    assert splitmix_reference(1234567, 3) == [6457827717110365317, 3203168211198807973, 9817491932198370423]


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**40 + 7])
def test_uniform_block_matches_the_integer_loop(seed):
    expected = [(z >> 11) / float(1 << 53) for z in splitmix_reference(seed, 50)]
    assert np.array_equal(uniform_block(seed, 50), np.array(expected))


def test_uniform_values_are_in_the_half_open_unit_interval_and_prefix_stable():
    u = uniform_block(5, 1000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert np.array_equal(uniform_block(5, 10), u[:10])


def test_normal_block_has_roughly_standard_moments_and_is_reproducible():
    z = normal_block(11, 20000)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03
    assert np.array_equal(z, normal_block(11, 20000))
