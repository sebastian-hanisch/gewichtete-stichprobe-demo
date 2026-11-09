"""Gewichte aus Ähnlichkeit und das gewichtete Quantil (SAA mit Gewichten), reine numpy-Einheiten.

Form nach Bertsimas/Kallus (nicht im Volltext gelesen, Websuche): Zu einem neuen Fall x bekommt jede vergangene Beobachtung i ein Gewicht w_i(x) >= 0 mit
Summe 1; die Bestellmenge ist das kleinste q, dessen kumuliertes Gewicht der Nachfragen D_i <= q mindestens tau erreicht (gewichtetes tau-Quantil =
Minimum der gewichteten Newsvendor-Kosten).

Einheiten:
  sqdist           quadratische Abstände (Zeilen = Anfragen, Spalten = Beobachtungen), elementweise gerechnet: Gleichstände bleiben exakt
  neighbour_order  Reihenfolge der Beobachtungen je Anfrage nach (Abstand, Nummer): Gleichstand im Abstand entscheidet die kleinere Nummer
  knn_weights      Gewicht 1/k für die k nächsten
  kernel_weights   Gauß-Kern exp(-d²/(2h²)), je Zeile auf Summe 1 normiert (stabil: nächster Nachbar hat vor der Normierung Gewicht 1)
  weighted_quantile gewichtetes tau-Quantil je Zeile
  pinball          mittlerer Pinball-Verlust
  loo_choice       k beziehungsweise Bandbreite per Leave-one-out auf dem Pinball-Verlust des Trainings
Zufall kommt hier nirgends vor."""

import numpy as np

import gws_constants as C


def sqdist(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Quadratische euklidische Abstände (len(a), len(b)); elementweise Differenzen statt der Binomialformel, damit gleiche Punkte exakt gleiche Abstände haben."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    out = np.empty((len(a), len(b)))
    for s in range(0, len(a), chunk):
        diff = a[s:s + chunk, None, :] - b[None, :, :]
        out[s:s + chunk] = (diff * diff).sum(axis=2)
    return out


def neighbour_order(d2: np.ndarray) -> np.ndarray:
    """Spaltenindizes je Zeile, aufsteigend nach Abstand; bei gleichem Abstand kommt die kleinere Nummer zuerst (stabile Sortierung)."""
    return np.argsort(d2, axis=1, kind="stable")


def knn_weights(order: np.ndarray, k: int, n_cols: int | None = None) -> np.ndarray:
    """Gewichte 1/k für die k nächsten Beobachtungen. k wird auf die Zahl der verfügbaren Spalten begrenzt (k > n heißt: alle, gleiche Gewichte)."""
    m, avail = order.shape
    n_cols = avail if n_cols is None else n_cols
    if k < 1:
        raise ValueError("k muss mindestens 1 sein")
    k = min(k, avail)
    w = np.zeros((m, n_cols))
    w[np.arange(m)[:, None], order[:, :k]] = 1.0 / k
    return w


def kernel_weights(d2: np.ndarray, h: float) -> np.ndarray:
    """Gauß-Kerngewichte je Zeile, auf Summe 1 normiert. Der kleinste Abstand der Zeile wird vor dem Exponieren abgezogen (Unterlauf-sicher):
    auch bei winziger Bandbreite behält der nächste Nachbar das Gewicht 1 statt 0 / 0. Ausgeschlossene Spalten stehen als +inf in d2 (Gewicht 0)."""
    if h <= 0:
        raise ValueError("Bandbreite muss positiv sein")
    d2 = np.asarray(d2, dtype=float)
    ref = d2.min(axis=1, keepdims=True)
    w = np.exp(-0.5 * (d2 - ref) / (h * h))
    return w / w.sum(axis=1, keepdims=True)


def weighted_quantile(w: np.ndarray, d: np.ndarray, tau: float) -> np.ndarray:
    """Kleinstes D_(j), dessen kumuliertes Gewicht (nach Nachfrage sortiert, Gleichstände nach Nummer) tau erreicht; je Zeile von `w`.
    tau = 1 liefert das Maximum der Nachfragen mit Gewicht > 0 (bis auf die Toleranz LOO_TOL im Gewicht); tau <= 0 das kleinste D."""
    w = np.atleast_2d(w)
    d = np.asarray(d, dtype=float)
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(w[:, order], axis=1)
    pos = (cum >= tau * cum[:, -1:] - C.LOO_TOL).argmax(axis=1)
    return d[order][pos]


def pinball(q: np.ndarray, d: np.ndarray, tau: float) -> float:
    """Mittlerer Pinball-Verlust max(tau * (D - q), (tau - 1) * (D - q)) = Newsvendor-Kosten bis auf den Faktor b + h."""
    r = np.asarray(d, dtype=float) - q
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def k_grid(n: int) -> list[int]:
    """Nachbarzahlen des Rasters, die im Leave-one-out noch Platz haben (k <= n - 1); mindestens der kleinste Wert."""
    ks = [k for k in C.KGRID if k <= n - 1]
    return ks or [min(C.KGRID)]


def h_grid(p: int) -> list[float]:
    """Bandbreiten des Rasters, auf die Dimension umgerechnet: h * sqrt(p / 3)."""
    return [h * float(np.sqrt(p / 3.0)) for h in C.HGRID]


def loo_losses_knn(d2_train: np.ndarray, d: np.ndarray, tau: float, ks: list[int]) -> list[float]:
    """Leave-one-out-Pinball-Verlust je k: jede Trainingsbeobachtung wird aus ihren eigenen Nachbarn (ohne sich selbst) vorhergesagt."""
    n = len(d)
    if n < 2:
        return [0.0 for _ in ks]
    dd = d2_train.copy()
    np.fill_diagonal(dd, np.inf)
    order = neighbour_order(dd)[:, : n - 1]
    return [pinball(weighted_quantile(knn_weights(order, k, n), d, tau), d, tau) for k in ks]


def loo_losses_kernel(d2_train: np.ndarray, d: np.ndarray, tau: float, hs: list[float]) -> list[float]:
    n = len(d)
    if n < 2:
        return [0.0 for _ in hs]
    dd = d2_train.copy()
    np.fill_diagonal(dd, np.inf)
    return [pinball(weighted_quantile(kernel_weights(dd, h), d, tau), d, tau) for h in hs]


def loo_choice(losses: list[float], grid: list) -> tuple:
    """Wert des Rasters mit dem kleinsten Verlust; bei Gleichstand der kleinere Rasterwert (np.argmin nimmt den ersten)."""
    i = int(np.argmin(losses))
    return grid[i], i
