"""Gemeinsame Fixtures: das Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Fünf vergangene Fälle x = 1..5 mit Nachfragen 10, 12, 15, 30, 32 (Spalte x, Vektor D).

    Handrechnung für die Anfrage x = 3,2 und tau = 0,8:
      k = 3: Nachbarn x = 3, 4, 2 (Abstände 0,2 / 0,8 / 1,2), Nachfragen 15, 30, 12, Gewichte je 1/3; sortiert 12, 15, 30 mit kumuliert 1/3, 2/3, 1 -> Quantil 30.
      Gauß-Kern, Bandbreite 1: Gewichte 0,0359 / 0,1963 / 0,3953 / 0,2928 / 0,0798, nach D kumuliert 0,0359 / 0,2321 / 0,6274 / 0,9202 / 1 -> Quantil 30."""
    return np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([10.0, 12.0, 15.0, 30.0, 32.0])
