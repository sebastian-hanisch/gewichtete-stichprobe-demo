"""Kennzahlen, Verdikt-Logik und Wiederholungen über feste Instanzen der Demo „Gewichtete Stichprobe“.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Zahlen im Text erscheinen mit Dezimalkomma (`de`).
Zahlen der Vormessung (30 Seeds, 5000 Testfälle, Wald mit scikit-learn) kommen nur aus `gws_vormessung.json`, nie aus der Live-Rechnung."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np

import gws_algorithm as A
import gws_constants as C
import gws_weights as W
from gws_scenario import make_data

METHODEN = ("kq", "direkt", "knn", "kern", "wald")
METHODEN_NAMEN = {
    "kq": "KQ + empirisches Quantil",
    "direkt": "direkt (Quantilregression)",
    "knn": "kNN-Gewichte",
    "kern": "Gauß-Kerngewichte",
    "wald": "Wald-Gewichte (Vormessung)",
}
LIVE_METHODEN = ("kq", "direkt", "knn", "kern")


def de(x: float, nd: int = 1) -> str:
    """Zahl mit Dezimalkomma; ein auf null gerundeter Wert erscheint ohne Minuszeichen."""
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s.replace(".", ",")


def de_signed(x: float, nd: int = 1) -> str:
    """Zahl mit Dezimalkomma und Vorzeichen (+ oder -); gerundet null ohne Vorzeichen."""
    s = de(x, nd)
    return s if float(s.replace(",", ".")) == 0.0 else (s if s.startswith("-") else "+" + s)


GLEICHAUF_ABS = 1.0      # Unterschiede bis zu dieser Zahl von Prozentpunkten gelten als gleichauf ...
GLEICHAUF_REL = 0.05     # ... oder bis zu diesem Anteil des größeren der beiden Werte


@dataclass(frozen=True)
class Settings:
    welt: str = C.DEFAULT_WELT
    p: int = C.DEFAULT_P
    n: int = C.DEFAULT_N
    tau: float = C.DEFAULT_TAU
    seed: int = C.DEFAULT_SEED


# --- eingefrorene Vormessung -----------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def frozen() -> dict:
    """Zellen der Vormessung, Schlüssel (Welt der App, p, n, tau) -> {kq, direkt, knn, kern, wald: {median, min, max}, k_median}."""
    raw = json.loads((Path(__file__).resolve().parent / C.VORMESSUNG_DATEI).read_text(encoding="utf-8"))
    back = {v: k for k, v in C.WELT_IN_VORMESSUNG.items()}
    return {(back[c["welt"]], c["p"], c["n"], c["tau"]): c for c in raw["zellen"]}


def frozen_cell(s: Settings) -> dict:
    return frozen()[(s.welt, s.p, s.n, s.tau)]


def frozen_medians(cell: dict) -> dict:
    return {m: cell[m]["median"] for m in METHODEN}


# --- Experiment 1: Handbeispiel ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HandResult:
    x: float
    tau: float
    k: int
    h: float
    w_knn: tuple
    w_kern: tuple
    q_knn: float
    q_kern: float
    mean_knn: float      # gewichtetes Mittel der Nachfragen mit kNN-Gewichten
    cover_mean_knn: float  # Gewicht der Nachfragen D <= Mittel (mit kNN-Gewichten): Anteil der gedeckten Fälle
    mean_all: float      # Mittel aller fünf Nachfragen


def hand_example(x: float, tau: float, k: int, h: float) -> HandResult:
    xs = np.array(C.HAND_X).reshape(-1, 1)
    d = np.array(C.HAND_D)
    d2 = W.sqdist(np.array([[x]]), xs)
    w_knn = W.knn_weights(W.neighbour_order(d2), k)
    w_kern = W.kernel_weights(d2, h)
    mean_knn = float((w_knn[0] * d).sum())
    return HandResult(
        x, tau, k, h, tuple(w_knn[0]), tuple(w_kern[0]),
        float(W.weighted_quantile(w_knn, d, tau)[0]), float(W.weighted_quantile(w_kern, d, tau)[0]),
        mean_knn, float(w_knn[0][d <= mean_knn].sum()), float(d.mean()),
    )


def hand_cdf(w: tuple) -> list[tuple[float, float]]:
    """Gewichtete Verteilungsfunktion als Sprungstellen (Nachfrage, kumuliertes Gewicht), nach Nachfrage sortiert; gleiche Nachfragen werden zusammengefasst."""
    d = np.array(C.HAND_D)
    order = np.argsort(d, kind="stable")
    out: dict[float, float] = {}
    cum = 0.0
    for i in order:
        cum += w[i]
        out[float(d[i])] = cum
    return list(out.items())


# --- Experiment 2: Welt wählen (eine Live-Instanz neben den Bändern der Vormessung) -------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    live: dict        # Methode -> Mehrkosten in % auf dieser Instanz (kq, direkt, knn, kern)
    k: int            # per Leave-one-out gewähltes k
    h_rel: float      # per Leave-one-out gewählte Bandbreite, auf p = 3 bezogen (Raster HGRID)


def _h_rel(h: float, p: int) -> float:
    return h / float(np.sqrt(p / 3.0))


def analyse(s: Settings) -> Analysis:
    data = make_data(s.welt, s.p, s.n, s.seed)
    dist = A.distances(data)
    knn = A.knn_rule(data, s.tau, dist=dist)
    kern = A.kernel_rule(data, s.tau, dist=dist)
    live = {
        "kq": A.extra_cost_percent(A.kq_empirical_rule(data, s.tau), data, s.tau),
        "direkt": A.extra_cost_percent(A.direct_rule(data, s.tau), data, s.tau),
        "knn": A.extra_cost_percent(knn.q, data, s.tau),
        "kern": A.extra_cost_percent(kern.q, data, s.tau),
    }
    return Analysis(s, live, int(knn.param), _h_rel(kern.param, s.p))


# --- Experiment 3: Fluch der Dimension (Wiederholungen über feste Instanzen) ---------------------------------------------------------------------

def sweep_p(welt: str, n: int, tau: float, ps=C.P_SWEEP, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Mehrkosten je Zahl der Merkmale p bei fester Datenmenge n: Median, Minimum, Maximum über die festen Instanzen `seeds` (nie ein einzelner Seed)."""
    rows = []
    for p in ps:
        vals = {m: [] for m in LIVE_METHODEN}
        for seed in seeds:
            data = make_data(welt, p, n, seed)
            dist = A.distances(data)
            vals["kq"].append(A.extra_cost_percent(A.kq_empirical_rule(data, tau), data, tau))
            vals["direkt"].append(A.extra_cost_percent(A.direct_rule(data, tau), data, tau))
            vals["knn"].append(A.extra_cost_percent(A.knn_rule(data, tau, dist=dist).q, data, tau))
            vals["kern"].append(A.extra_cost_percent(A.kernel_rule(data, tau, dist=dist).q, data, tau))
        rows.append({"p": p, **{m: {"median": float(median(v)), "min": min(v), "max": max(v)} for m, v in vals.items()}})
    return rows


# --- Experiment 4: k und Bandbreite (Leave-one-out gegen feste Werte) -----------------------------------------------------------------------------

def choice_experiment(welt: str, p: int, n: int, tau: float, seeds=C.SWEEP_SEEDS) -> dict:
    """Je Instanz: Mehrkosten jedes festen Rasterwerts, der Leave-one-out-Wahl und der im Nachhinein besten festen Wahl; Zusammenfassung als Median über die Instanzen."""
    out = {}
    for name, fn in (("knn", A.knn_grid), ("kern", A.kernel_grid)):
        results = [fn(make_data(welt, p, n, seed), tau) for seed in seeds]
        grid = results[0].grid
        out[name] = {
            "grid": list(grid) if name == "kern" else [int(g) for g in grid],
            "fixed": [float(median(r.extras[i] for r in results)) for i in range(len(grid))],
            "loo": float(median(r.chosen_extra for r in results)),
            "best": float(median(r.best_extra for r in results)),
            "chosen_grid_index": [r.chosen for r in results],
        }
    return out


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------------

def verdict_hand(r: HandResult) -> tuple[str, str]:
    """Verdikt des Handbeispiels: Bestellmengen beider Gewichtungen gegen das gewichtete Mittel."""
    cover = 100.0 * r.cover_mean_knn
    return "info", (
        f"Mit k = {r.k} Nachbarn bestellt das gewichtete {de(r.tau, 2)}-Quantil **{de(r.q_knn, 0)}**, mit Gauß-Kern (Bandbreite {de(r.h, 1)}) **{de(r.q_kern, 0)}**. "
        f"Das gewichtete Mittel der Nachbarn ({de(r.mean_knn, 1)}) deckt nur {de(cover, 1)} % des Gewichts; verlangt sind mindestens {de(100 * r.tau, 0)} %."
    )


def tolerance(a: float, b: float) -> float:
    """Gleichauf-Grenze zweier Mehrkosten-Werte in Prozentpunkten: das Größere aus 1 und 5 % des größeren Werts (bei Werten um 150 % sind 3 Punkte kein Unterschied)."""
    return max(GLEICHAUF_ABS, GLEICHAUF_REL * max(abs(a), abs(b)))


def verdict_welt(s: Settings, cell: dict) -> tuple[str, str]:
    """Verdikt des Experiments „Welt wählen“ aus den Medianen der Vormessung (30 Seeds, 5000 Testfälle): Hauptverfahren kNN und Kern gegen KQ und direkt; der Wald nur als Vergleichszahl."""
    med = frozen_medians(cell)
    param = min(("kq", "direkt"), key=lambda m: med[m])
    weighted = min(("knn", "kern"), key=lambda m: med[m])
    diff = med[param] - med[weighted]
    head = (
        f"In der Vormessung (Median über {C.VORMESSUNG_SEEDS} Seeds, {C.WELT_NAMEN[s.welt]}, p = {s.p}, n = {s.n}, τ = {de(s.tau, 2)}) kosten die besten Gewichte "
        f"({METHODEN_NAMEN[weighted]}) **{de(med[weighted])} %** Mehrkosten gegen das Orakel, die beste parametrische Regel ({METHODEN_NAMEN[param]}) **{de(med[param])} %**. "
    )
    wald_unten = med["wald"] < min(med[param], med[weighted]) - tolerance(med[param], med[weighted])
    tail = f"Wald-Gewichte (Vormessung mit scikit-learn, nicht live): {de(med['wald'])} %" + (", deutlich darunter." if wald_unten else ".")
    if diff > tolerance(med[param], med[weighted]):
        return "success", head + f"Die Gewichte liegen {de(diff)} Prozentpunkte darunter. " + tail
    if -diff > tolerance(med[param], med[weighted]):
        return "warning", head + f"Die parametrische Regel liegt {de(-diff)} Prozentpunkte darunter (bei dieser Datenmenge und Dimension). " + tail
    return "info", head + f"Der Unterschied von {de(abs(diff))} Prozentpunkten liegt innerhalb der Gleichauf-Grenze (höchstens {de(GLEICHAUF_ABS, 0)} Prozentpunkt oder {de(100 * GLEICHAUF_REL, 0)} % des größeren Werts). " + tail


def verdict_dim(rows: list[dict]) -> tuple[str, str]:
    """Verdikt des Experiments „Fluch der Dimension“: Anstieg der Mehrkosten vom kleinsten zum größten p (Median über die festen Instanzen)."""
    lo, hi = rows[0], rows[-1]
    rise = {m: hi[m]["median"] - lo[m]["median"] for m in ("kq", "knn", "kern")}
    text = (
        f"Von p = {lo['p']} auf p = {hi['p']} (gleiche Datenmenge, Median über {len(C.SWEEP_SEEDS)} feste Instanzen) steigen die Mehrkosten der kNN-Gewichte von {de(lo['knn']['median'])} % "
        f"auf {de(hi['knn']['median'])} % (Änderung {de_signed(rise['knn'])} Prozentpunkte), die der Gauß-Kerngewichte um {de_signed(rise['kern'])} Prozentpunkte, "
        f"die von KQ + empirischem Quantil von {de(lo['kq']['median'])} % auf {de(hi['kq']['median'])} % (Änderung {de_signed(rise['kq'])} Prozentpunkte)."
    )
    gap = hi["knn"]["median"] - hi["kq"]["median"]
    end = (f" Bei p = {hi['p']} liegen die kNN-Gewichte {de(abs(gap))} Prozentpunkte " + ("über" if gap > 0 else "unter") + " KQ + empirischem Quantil.")
    if rise["knn"] > rise["kq"] + GLEICHAUF_ABS:
        return "warning", text + f" Die Gewichte verlieren zwischen p = {lo['p']} und p = {hi['p']} stärker als die parametrische Regel." + end
    return "info", text + " Hier steigt die parametrische Regel mindestens so stark wie die Gewichte." + end


def verdict_wahl(res: dict) -> tuple[str, str]:
    """Verdikt des Experiments „k und Bandbreite“: Leave-one-out-Wahl gegen die im Nachhinein beste feste Wahl (Median über die festen Instanzen)."""
    k, h = res["knn"], res["kern"]
    return "info", (
        f"Die Wahl per Leave-one-out kostet im Median über {len(C.SWEEP_SEEDS)} feste Instanzen {de(k['loo'])} % Mehrkosten (kNN) beziehungsweise {de(h['loo'])} % (Kern); "
        f"die im Nachhinein beste feste Wahl je Instanz käme auf {de(k['best'])} % beziehungsweise {de(h['best'])} %. "
        f"Der Abstand von {de(k['loo'] - k['best'])} beziehungsweise {de(h['loo'] - h['best'])} Prozentpunkten ist der Preis dafür, dass k und Bandbreite selbst aus den Daten kommen (der Vergleich bevorzugt die Nachhinein-Wahl, sie ist nicht ausführbar)."
    )
