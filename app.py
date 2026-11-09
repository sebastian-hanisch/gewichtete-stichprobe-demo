"""Gewichtete Stichprobe: Nachbarn statt Modell - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Demo der Reihe „Entscheidungsorientiertes Lernen“: Statt ein Modell zu schätzen, werden vergangene Fälle nach Ähnlichkeit gewichtet und das gewichtete
Quantil der Nachfragen bestellt (SAA mit Gewichten). kNN- und Gauß-Kerngewichte rechnet die App live in numpy; Wald-Gewichte erscheinen nur als
eingefrorene Vormessung mit scikit-learn.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import gws_constants as C
import gws_evaluation as ev
from gws_layout import dunkles_design, kopf_css
from gws_presets import (
    HAND_SPECS,
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    reset_hand,
    sidebar_presets,
    sync_query_params,
)
from gws_visualization import build_choice, build_dim, build_hand_cdf, build_hand_weights, build_methods

st.set_page_config(page_title="Gewichtete Stichprobe – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_p(welt, n, tau):
    return ev.sweep_p(welt, n, tau)


@st.cache_data(show_spinner=False)
def _choice(welt, p, n, tau):
    return ev.choice_experiment(welt, p, n, tau)


def _tau_label(t):
    return f"{t:.2f}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gws_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Gewichtete Stichprobe: Nachbarn statt Modell")
st.markdown(
    """
Demo der Reihe „Entscheidungsorientiertes Lernen“. Ein Händler bestellt vor der Nachfrage; zu jedem Fall gibt es **Merkmale x**. Statt ein Modell für die Nachfrage zu schätzen, vergibt man an die
**vergangenen Fälle Gewichte nach ihrer Ähnlichkeit** zum neuen Fall und bestellt das **Quantil der gewichteten Nachfragen** (SAA mit Gewichten). Das kommt ohne parametrisches Modell für die Nachfrage aus; es setzt voraus, dass ähnliche Fälle ähnliche Nachfrageverteilungen haben.
Die Demo misst, wann das gewinnt und wann es zahlt: Gewichte aus den **k nächsten Nachbarn** und aus einem **Gauß-Kern** rechnet die App live in numpy;
Gewichte aus einem **Zufallswald** erscheinen nur als eingefrorene Vormessung (mit scikit-learn, nicht live).

Weiter unten: vier Experimente (**Gewichte von Hand**, **Welt wählen**, **Fluch der Dimension**, **k und Bandbreite**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die gewichtete Stichprobe", expanded=True):
    st.markdown(
        r"""
1. **Ähnlichkeit messen:** Zu einem neuen Fall $x$ hat jeder vergangene Fall $i$ den Abstand $\lVert x - x_i \rVert$.
2. **Gewichte vergeben:** beim **kNN** bekommen die $k$ nächsten je $1/k$, alle anderen 0; beim **Gauß-Kern** gilt $w_i \propto \exp(-\lVert x - x_i\rVert^2 / 2h^2)$. Die Gewichte summieren zu 1.
3. **Gewichtetes Quantil bestellen:** die Nachfragen nach Größe sortieren, die Gewichte addieren und die erste Nachfrage bestellen, bei der die Summe $\tau$ erreicht. Dabei ist $\tau = b/(b+h)$ das kritische Verhältnis aus den Kosten $b$ je fehlende und $h$ je überschüssige Einheit.
4. **Warum das funktioniert:** Das $\tau$-Quantil der Nachfrage minimiert die erwarteten Newsvendor-Kosten. Die Gewichte ersetzen die unbekannte Verteilung der Nachfrage zum Fall $x$ durch eine gewichtete Stichprobe. Voraussetzung: ähnliche Fälle haben ähnliche Nachfrageverteilungen, und es gibt genug Fälle in der Nähe.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Instanz** (Experimente 2 bis 4)")
    st.selectbox("Welt (wahre Nachfrage)", C.WELTEN, key="welt_sel", format_func=C.WELT_NAMEN.get, help="Wie Mittelwert und Streuung der Nachfrage von den Merkmalen abhängen.")
    st.select_slider("Merkmale p", C.P_OPTIONS, key="p_sel", help="Zahl der Merkmale. Wirkt auf die Experimente 2 und 4; Experiment 3 variiert p selbst.")
    st.select_slider("Fälle n", C.N_OPTIONS, key="n_sel", help="Zahl der vergangenen Fälle (Trainingsdaten).")
    st.select_slider("Verhältnis τ", C.TAUS, key="tau_sel", format_func=_tau_label, help="Kritisches Verhältnis b / (b + h): bei 0,95 ist eine fehlende Einheit 19-mal so teuer wie eine überschüssige.")
    st.number_input("Zufalls-Seed der Instanz", 0, C.SEED_MAX, key="seed_input", step=1, help="Bestimmt Merkmale, Nachfragen und Testfälle der Live-Instanz.")
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

# --- Experiment 1: Gewichte von Hand (das wachsende Beispiel) ---------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Gewichte von Hand")
st.markdown(
    "Fünf vergangene Fälle mit einem Merkmal: $x = 1, 2, 3, 4, 5$ und Nachfragen $D = 10, 12, 15, 30, 32$. Zu einer neuen Anfrage $x$ vergeben die beiden Verfahren Gewichte; "
    "bestellt wird das gewichtete $\\tau$-Quantil. Bei gleichem Abstand zählt der Fall mit der kleineren Nummer zuerst."
)
c1, c2, c3, c4, c5 = st.columns([3, 3, 3, 3, 2])
with c1:
    st.slider("Anfrage x", C.HAND_X_MIN, C.HAND_X_MAX, key="hand_x", step=0.1, format="%.1f", help="Das neue Merkmal, für das bestellt wird.")
with c2:
    st.slider("Kritisches Verhältnis τ", C.HAND_TAU_MIN, C.HAND_TAU_MAX, key="hand_tau", step=0.05, format="%.2f", help="Anteil der Nachfragen, der mit der Bestellmenge gedeckt sein soll.")
with c3:
    st.slider("kNN: Nachbarn k", C.HAND_K_MIN, C.HAND_K_MAX, key="hand_k", step=1, help="Wie viele nächste Nachbarn je das Gewicht 1/k bekommen (k = 5 gibt allen fünf Fällen Gewicht 0,2).")
with c4:
    st.slider("Gauß-Kern: Bandbreite h", C.HAND_H_MIN, C.HAND_H_MAX, key="hand_h", step=0.1, format="%.1f", help="Wie schnell das Gewicht mit dem Abstand fällt; große Bandbreite: fast gleiche Gewichte.")
with c5:
    st.button("Zurücksetzen", on_click=reset_hand, width="stretch", help="Setzt Anfrage, τ, k und Bandbreite auf das Handbeispiel der README zurück (x = 3,2; τ = 0,8; k = 3; h = 1,0).")

hand = ev.hand_example(
    float(st.session_state["hand_x"]), float(st.session_state["hand_tau"]), int(st.session_state["hand_k"]), float(st.session_state["hand_h"])
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bestellmenge kNN", f"{hand.q_knn:g}")
m2.metric("Bestellmenge Gauß-Kern", f"{hand.q_kern:g}")
m3.metric("Gewichtetes Mittel (kNN)", f"{hand.mean_knn:.1f}".replace(".", ","))
m4.metric("Mittel aller fünf Fälle", f"{hand.mean_all:.1f}".replace(".", ","))
g1, g2 = st.columns(2)
g1.plotly_chart(build_hand_weights(hand), width="stretch", key="hand_weights_chart")
g2.plotly_chart(build_hand_cdf(hand), width="stretch", key="hand_cdf_chart")
kind, text = ev.verdict_hand(hand)
getattr(st, kind)(text)

st.markdown("---")

# --- Kernfrage und Experiment 2: Welt wählen ------------------------------------------------------------------------------------------------------

st.subheader("❓ Kernfrage: Wann lohnen Nachbarn statt eines Modells?")
st.markdown(
    "Experiment 2, **Welt wählen**: Die Newsvendor-Nachfrage hängt von standardnormalen Merkmalen ab (vier Welten). Die Gewichte treten gegen zwei parametrische Regeln an, "
    "**KQ + empirisches Quantil** (Mittelwert linear schätzen, dazu das Quantil der Residuen) und **direkt** (lineare Quantilregression). Die Balken sind die **Vormessung** "
    f"(Median über {C.VORMESSUNG_SEEDS} feste Seeds, {C.VORMESSUNG_NTEST} Testfälle), die Rauten **diese eine Instanz, live gerechnet**. Welt, p, n, τ und Zufalls-Seed stehen in der Seitenleiste, dort auch der Schnellstart."
)
sync_query_params({**{k: st.session_state[k] for k in SETTING_SPECS}, **{k: st.session_state[k] for k in HAND_SPECS}})
settings = ev.Settings(
    st.session_state["welt_sel"], int(st.session_state["p_sel"]), int(st.session_state["n_sel"]), float(st.session_state["tau_sel"]), int(st.session_state["seed_input"])
)
cell = ev.frozen_cell(settings)
analysis = _analysis(settings)

st.markdown("## 🎯 Ergebnis")
mm = st.columns(5)
for col, m in zip(mm, ev.METHODEN):
    if m == "wald":
        col.metric(ev.METHODEN_NAMEN[m], f"{cell[m]['median']:.1f} %".replace(".", ","), delta="nur Vormessung, nicht live", delta_color="off")
    else:
        # Delta immer als „mein Wert minus Referenz“ (live minus Median der Vormessung); die Farbe regelt delta_color="inverse".
        col.metric(ev.METHODEN_NAMEN[m], f"{analysis.live[m]:.1f} %".replace(".", ","), delta=f"{analysis.live[m] - cell[m]['median']:+.1f} gegen Median".replace(".", ","), delta_color="inverse")
st.caption(
    "Große Zahl: Mehrkosten gegen das Orakel (kennt die wahre bedingte Verteilung) auf dieser Live-Instanz; darunter der Abstand zum Median der Vormessung. "
    "Einzelne Instanzen weichen vom Median ab (Spalten Minimum und Maximum der Tabelle); kleine negative Werte sind Rauschen der Testfälle."
)
st.plotly_chart(build_methods(cell, analysis.live), width="stretch", key="methods_chart")
kind, text = ev.verdict_welt(settings, cell)
getattr(st, kind)(text)
rows_table = [
    {
        "Regel": ev.METHODEN_NAMEN[m],
        "diese Instanz (live), %": None if m == "wald" else round(analysis.live[m], 1),
        f"Median {C.VORMESSUNG_SEEDS} Seeds, %": cell[m]["median"],
        "Minimum, %": cell[m]["min"],
        "Maximum, %": cell[m]["max"],
    }
    for m in ev.METHODEN
]
st.dataframe(
    pd.DataFrame(rows_table), hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%.1f") for c in rows_table[0] if c != "Regel"},
)
st.caption(
    f"Leave-one-out wählt auf dieser Instanz k = {analysis.k} und die Bandbreite {ev.de(analysis.h_rel)} (bezogen auf p = 3). Gewähltes k der Vormessung im Median über {C.VORMESSUNG_SEEDS} Seeds: {cell['k_median']}. "
    f"Wald-Gewichte: {C.WALD_HINWEIS} (50 Bäume, mindestens 5 je Blatt, Merkmalsauswahl 0,5, nicht abgestimmt); die Zeile sagt nichts über Zufallswälder im Allgemeinen."
)

st.markdown("---")

# --- Experiment 3: Fluch der Dimension --------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Fluch der Dimension")
st.markdown(
    f"Gleiche Welt, gleiche Datenmenge n = {settings.n} und gleiches τ wie oben; nur die Zahl der Merkmale p wächst. Gerechnet wird live über {len(C.SWEEP_SEEDS)} feste Instanzen "
    f"(Seeds {C.SWEEP_SEEDS[0]} bis {C.SWEEP_SEEDS[-1]}), Fehlerbalken: Minimum bis Maximum, Punkte: Median."
)
with st.spinner(f"Rechnet {len(C.P_SWEEP) * len(C.SWEEP_SEEDS)} Instanzen (bei n = 1000 deutlich länger als bei n = 100) …"):
    sweep_rows = _sweep_p(settings.welt, settings.n, settings.tau)
st.plotly_chart(build_dim(sweep_rows), width="stretch", key="dim_chart")
kind, text = ev.verdict_dim(sweep_rows)
getattr(st, kind)(text)
st.caption("Vergleich mit der Vormessung bei p = 3 und p = 10 in dieser Welt, bei diesem n und τ (Median über 30 Seeds, Mehrkosten in %):")
dim_rows = []
for p in C.P_OPTIONS:
    c = ev.frozen_cell(ev.Settings(settings.welt, p, settings.n, settings.tau, settings.seed))
    dim_rows.append({"p": p, **{ev.METHODEN_NAMEN[m]: c[m]["median"] for m in ev.METHODEN}})
st.dataframe(
    pd.DataFrame(dim_rows), hide_index=True, width="stretch",
    column_config={ev.METHODEN_NAMEN[m]: st.column_config.NumberColumn(format="%.1f") for m in ev.METHODEN},
)

st.markdown("---")

# --- Experiment 4: k und Bandbreite -----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: k und Bandbreite")
st.markdown(
    f"Die Nachbarzahl k und die Bandbreite h sind keine Naturkonstanten: Die App wählt sie per **Leave-one-out** auf dem Pinball-Verlust des Trainings (jeder Fall wird aus den übrigen vorhergesagt). "
    f"Gezeigt sind die Mehrkosten jeder festen Wahl, die Leave-one-out-Wahl und die im Nachhinein beste feste Wahl (Median über {len(C.SWEEP_SEEDS)} feste Instanzen; p = {settings.p}, n = {settings.n}, τ = {_tau_label(settings.tau)})."
)
with st.spinner(f"Rechnet {len(C.SWEEP_SEEDS)} Instanzen mit allen Rasterwerten …"):
    choice = _choice(settings.welt, settings.p, settings.n, settings.tau)
st.plotly_chart(build_choice(choice), width="stretch", key="choice_chart")
kind, text = ev.verdict_wahl(choice)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Wahrheit ist nicht einfach und glatt.** Ist sie linear und die Streuung konstant, braucht man die Gewichte nicht. | Welt linear, n = 1000, τ = 0,95: kNN-Gewichte 9,3 % (p = 3) und 23,2 % (p = 10) Mehrkosten gegen das Orakel, KQ + empirisches Quantil 0,3 % und 0,7 % (Vormessung, Median über 30 Seeds). | passende parametrische Klasse (hier als Vergleich gemessen) |
| **Genug Fälle in der Nähe.** Bei wenig Daten und vielen Merkmalen sind die nächsten Nachbarn vermutlich weit entfernt (Deutung, Abstände sind nicht gemessen). | p = 10, n = 100: kNN- und Wald-Gewichte liegen in den Welten linear, Streuung linear und \\|x₁\\| bei 31,4 bis 75,0 % Mehrkosten (Vormessung, Median über 30 Seeds, τ = 0,5 / 0,8 / 0,95). | Regularisierung, Merkmalsauswahl (hier nicht gemessen) |
| **Nähe im Merkmalsraum heißt ähnliche Nachfrage.** Der Abstand zählt alle p Richtungen gleich. | Die Demo prüft nicht, was eine Gewichtung der Richtungen (Merkmalsauswahl, andere Abstände) ändern würde; der Wald wählt Merkmale, ist aber nicht abgestimmt. | nicht gemessen |
| **k und Bandbreite kommen aus den Daten.** | Experiment 4 misst den Abstand der Leave-one-out-Wahl zur im Nachhinein besten festen Wahl (Median über 5 feste Instanzen); die Nachhinein-Wahl ist nicht ausführbar. | Wahl per Kreuzvalidierung statt eines festen Rasters (hier nur Leave-one-out) |
| **Eine Familie von Instanzen.** Normalverteiltes Rauschen, Stutzung bei 0, standardnormale Merkmale, eine Nachfrage je Fall. | Andere Verteilungen, schiefe Merkmale oder abhängige Fälle sind nicht gemessen; die Zahlen sind keine Bestätigung der Sätze der Literatur. | nicht gemessen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ vergangene Fälle $(x_i, D_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und Nachfragen $D_i \ge 0$ sowie ein neuer Fall $x$. Gesucht ist die Bestellmenge $q(x)$.

| Symbol | Bedeutung |
|---|---|
| $x_i,\ D_i$ | Merkmale und Nachfrage des vergangenen Falls $i$ |
| $b,\ h$ | Kosten je fehlende beziehungsweise überschüssige Einheit |
| $\tau = b/(b+h)$ | kritisches Verhältnis |
| $w_i(x)$ | Gewicht des Falls $i$ für den neuen Fall $x$, $w_i \ge 0$, $\sum_i w_i = 1$ |
| $k,\ h_K$ | Nachbarzahl beziehungsweise Bandbreite des Kerns |

**Kosten und Optimum.** $C(q, D) = b\,(D - q)^+ + h\,(q - D)^+$. Bei bekannter bedingter Verteilung ist das $\tau$-Quantil $Q_\tau(D \mid x)$ optimal (das Orakel).

**Gewichtete Stichprobe (SAA mit Gewichten).**
$$q(x) = \arg\min_q \sum_{i=1}^n w_i(x)\, C(q, D_i) = \min\Big\{ q : \sum_{i:\, D_i \le q} w_i(x) \ge \tau \Big\}.$$

**Gewichte.**
$$w_i^{\mathrm{kNN}}(x) = \tfrac{1}{k}\,\mathbf{1}[\,i \in N_k(x)\,], \qquad w_i^{\mathrm{Kern}}(x) = \frac{\exp\!\big(-\lVert x - x_i\rVert^2 / (2 h_K^2)\big)}{\sum_j \exp\!\big(-\lVert x - x_j\rVert^2 / (2 h_K^2)\big)}.$$

**Wahl von $k$ und $h_K$ (Leave-one-out).** Mit $q^{(-i)}(x_i)$ dem gewichteten Quantil ohne den Fall $i$ und dem Pinball-Verlust $\rho_\tau(u) = \max(\tau u, (\tau - 1) u)$:
$$\hat\theta = \arg\min_{\theta} \frac{1}{n} \sum_{i=1}^n \rho_\tau\big(D_i - q_\theta^{(-i)}(x_i)\big).$$
Das Raster ist $k \in \{5, 10, 20, 40, 80, 160\}$ (nur $k \le n - 1$) und $h_K \in \{0{,}3;\ 0{,}5;\ 0{,}8;\ 1{,}2;\ 2{,}0;\ 3{,}0\} \cdot \sqrt{p/3}$.

**Mehrkosten gegen das Orakel.** $\dfrac{\bar C(\text{Regel}) - \bar C(\text{Orakel})}{\bar C(\text{Orakel})}$, gemittelt über dieselben Testfälle.

**Literatur** (nicht im Volltext gelesen, der Wortlaut der Abstracts nicht geprüft, die Inhaltsangaben stützen sich auf Zusammenfassungen einer Websuche; Satznummern und Seiten sind nicht geprüft): Bertsimas und Kallus, *From Predictive to Prescriptive Analytics*, Management Science, 2020: Entscheidung als Minimum der gewichteten Kosten, Gewichte laut Websuche zum Beispiel aus Nachbarn, Kernen oder Wäldern (im arXiv-Abstract stehen nach einer späteren Prüfung lokale Regression, CART und Random Forests; Nachbarn und Kerne erst im Haupttext). Ban und Rudin, *The Big Data Newsvendor*, Operations Research, 2019: Newsvendor mit Merkmalen; welche Verfahren genau vorkommen (zum Beispiel eine Optimierung mit Kerngewichten), ist hier nicht geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
