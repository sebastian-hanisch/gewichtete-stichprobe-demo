"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Voreinstellungen der Demo „Gewichtete Stichprobe“."""

# --- Welten (Newsvendor mit Merkmalen; Nachfrage D = max(0, mu(x) + sigma(x) * z), z standardnormal) ------------------------------------------
WELTEN = ("linear", "streuung_linear", "betrag", "quadratisch")
WELT_NAMEN = {
    "linear": "linear (einfach, glatt)",
    "streuung_linear": "Streuung linear in x₁",
    "betrag": "nicht monotone Streuung |x₁|",
    "quadratisch": "quadratisch (lineare Klasse falsch)",
}
# Namen der Welten in der Vormessung (vormessung/out_eo_v3_gewichtet.txt): linear, streuung-linear, heteroskedastisch, nichtlinear
WELT_IN_VORMESSUNG = {"linear": "linear", "streuung_linear": "streuung-linear", "betrag": "heteroskedastisch", "quadratisch": "nichtlinear"}
MU0, MU_SLOPE, MU_QUAD = 60.0, 10.0, 8.0      # mu(x) = 60 + 10 * (b . x) [+ 8 * (x1^2 - 1) in der quadratischen Welt], b mit Länge 1
SIGMA_CONST = 8.0                             # linear und quadratisch: konstante Streuung
SIGMA_LIN = (6.0, 4.0, 1.0)                   # streuung_linear: max(1, 6 + 4 * x1)
SIGMA_ABS = (4.0, 6.0)                        # betrag: 4 + 6 * |x1|

# --- Kritisches Verhältnis tau = b / (b + h): Kosten b je fehlende, h je überschüssige Einheit --------------------------------------------------
TAUS = (0.5, 0.8, 0.95)
KOSTEN_BH = {0.5: (1.0, 1.0), 0.8: (4.0, 1.0), 0.95: (19.0, 1.0)}

# --- Raster der Verfahren ---------------------------------------------------------------------------------------------------------------------
KGRID = (5, 10, 20, 40, 80, 160)              # Nachbarzahlen k, Wahl per Leave-one-out (nur Werte <= n - 1)
HGRID = (0.3, 0.5, 0.8, 1.2, 2.0, 3.0)        # Bandbreiten, bezogen auf p = 3; gerechnet wird h * sqrt(p / 3)
LOO_TOL = 1e-12                               # Toleranz beim Vergleich der kumulierten Gewichte mit tau

# --- Regler (Hauptbereich) ----------------------------------------------------------------------------------------------------------------------
P_OPTIONS = (3, 10)
N_OPTIONS = (100, 1000)
DEFAULT_WELT, DEFAULT_P, DEFAULT_N, DEFAULT_TAU = "quadratisch", 3, 1000, 0.95
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100005))    # feste Instanzen für alle Wiederholungen der App: nie ein einzelner Seed als Befund
NTEST_LIVE = 1000                             # Testfälle der Live-Rechnung (Vormessung: 5000)
P_SWEEP = (1, 2, 3, 5, 10)                    # Experiment 3: Zahl der Merkmale bei fester Datenmenge

# --- Handbeispiel (Experiment 1) ----------------------------------------------------------------------------------------------------------------
HAND_X = (1.0, 2.0, 3.0, 4.0, 5.0)
HAND_D = (10.0, 12.0, 15.0, 30.0, 32.0)
HAND_DEFAULT = {"x": 3.2, "tau": 0.8, "k": 3, "h": 1.0}
HAND_X_MIN, HAND_X_MAX = 1.0, 5.0
HAND_TAU_MIN, HAND_TAU_MAX = 0.5, 0.95
HAND_K_MIN, HAND_K_MAX = 1, 5
HAND_H_MIN, HAND_H_MAX = 0.3, 3.0

# --- Vormessung (eingefroren) -----------------------------------------------------------------------------------------------------------------
VORMESSUNG_DATEI = "gws_vormessung.json"
VORMESSUNG_SEEDS = 30
VORMESSUNG_NTEST = 5000
WALD_HINWEIS = "Vormessung mit scikit-learn, nicht live"

COLOR_KNN = "#2F6B65"
COLOR_KERN = "#4c78a8"
COLOR_KQ = "#b0622b"
COLOR_DIREKT = "#9b7bb8"
COLOR_WALD = "#8a8a8a"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Quadratische Wahrheit: Gewichte gewinnen": {"welt": "quadratisch", "p": 3, "n": 1000, "tau": 0.95, "seed": DEFAULT_SEED},
    "Lineare Wahrheit: Gewichte zahlen": {"welt": "linear", "p": 3, "n": 1000, "tau": 0.95, "seed": DEFAULT_SEED},
    "Fluch der Dimension: wenig Daten, viele Merkmale": {"welt": "streuung_linear", "p": 10, "n": 100, "tau": 0.95, "seed": DEFAULT_SEED},
    "Quadratisch, p = 10, n = 100: nur der Wald trägt": {"welt": "quadratisch", "p": 10, "n": 100, "tau": 0.95, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Quadratische Wahrheit: Gewichte gewinnen": "Der Mittelwert hängt quadratisch von x₁ ab, die lineare Regel trifft ihn nicht: p = 3, n = 1000, τ = 0,95.",
    "Lineare Wahrheit: Gewichte zahlen": "Die Wahrheit ist linear und die Streuung konstant: p = 3, n = 1000, τ = 0,95. Die parametrische Regel ist hier richtig spezifiziert.",
    "Fluch der Dimension: wenig Daten, viele Merkmale": "Streuung linear in x₁, aber zehn Merkmale und nur 100 Fälle: p = 10, n = 100, τ = 0,95.",
    "Quadratisch, p = 10, n = 100: nur der Wald trägt": "Quadratische Wahrheit mit zehn Merkmalen und 100 Fällen: in der Vormessung gewinnt hier nur der (nicht live gerechnete) Wald.",
}
