# Gewichtete Stichprobe: Nachbarn statt Modell – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gewichtete-stichprobe-demo.streamlit.app/)**

Demo der **Linie „Entscheidungsorientiertes Lernen“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Anders als die Fall-Demos (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt sie **ein** Verfahren an einem wachsenden Beispiel: die **gewichtete Stichprobe** (SAA mit Gewichten).
Das Vehikel ist der **Newsvendor mit Merkmalen**: Ein Händler bestellt vor der Nachfrage, zu jedem Fall gibt es Merkmale x, eine fehlende Einheit kostet b, eine überschüssige h.
Die verwandte Demo [saa-demo](https://github.com/sebastian-hanisch/saa-demo) (Optimierung unter Unsicherheit) rechnet SAA aus einer Stichprobe **ohne** Merkmale; hier haben die Fälle Merkmale, und die Gewichte kommen aus der Ähnlichkeit.

## Warum dieses Problem

Man kann ein Modell für die Nachfrage schätzen und dann bestellen (zweistufig), oder man vergibt an die **vergangenen Fälle Gewichte nach ihrer Ähnlichkeit** zum neuen Fall und bestellt das **Quantil der gewichteten Nachfragen**.
Das kommt ohne parametrisches Modell für die Nachfrage aus; es setzt voraus, dass ähnliche Fälle ähnliche Nachfrageverteilungen haben. Die Kernfrage der Demo: **Wann gewinnt das, und wann zahlt man dafür?** Gemessen wird gegen zwei parametrische Regeln (Kleinste Quadrate plus empirisches Residuenquantil, lineare Quantilregression) und gegen ein Orakel, das die wahre bedingte Verteilung kennt.

## Modell

- **Nachfrage:** D = max(0, μ(x) + σ(x)·z) mit standardnormalem z und standardnormalen Merkmalen x ∈ ℝᵖ. Vier Welten: **linear** (μ = 60 + 10·(b·x), σ = 8), **Streuung linear** (σ = max(1, 6 + 4·x₁)), **nicht monotone Streuung** (σ = 4 + 6·|x₁|) und **quadratisch** (μ = 60 + 10·(b·x) + 8·(x₁² − 1)); b ist ein Zufallsvektor der Länge 1.
- **Kosten:** b je fehlende, h je überschüssige Einheit, kritisches Verhältnis τ = b/(b+h) ∈ {0,5; 0,8; 0,95} (b : h = 1 : 1, 4 : 1, 19 : 1).
- **Gewichtete Stichprobe:** q(x) = kleinstes q mit Σ_{i: Dᵢ ≤ q} wᵢ(x) ≥ τ. Gewichte: **kNN** (die k nächsten je 1/k) oder **Gauß-Kern** (wᵢ ∝ exp(−‖x − xᵢ‖²/2h²)); k und Bandbreite per **Leave-one-out** auf dem Pinball-Verlust des Trainings (Raster k ∈ {5, 10, 20, 40, 80, 160} mit k ≤ n − 1, Bandbreiten {0,3; 0,5; 0,8; 1,2; 2,0; 3,0}·√(p/3)).
- **Orakel und Mehrkosten:** Das Orakel bestellt das wahre bedingte τ-Quantil. Mehrkosten = (Kosten der Regel − Kosten des Orakels) / Kosten des Orakels, in Prozent, auf denselben Testfällen.

Formeln und Symboltabelle stehen im 📐-Expander der App.

## Methodik

Fünf Regeln im Vergleich; vier rechnet die App live in numpy/scipy, eine steht nur als eingefrorene Vormessung da:

| Regel | Quelle in der App |
|---|---|
| KQ + empirisches Quantil (Mittelwert linear schätzen, dazu das Quantil der Residuen; numpy-Standard mit linearer Interpolation) | live |
| direkt: lineare Quantilregression (Pinball-Verlust als lineares Programm, `scipy.optimize.linprog`, HiGHS) | live |
| kNN-Gewichte, k per Leave-one-out | live |
| Gauß-Kerngewichte, Bandbreite per Leave-one-out | live |
| **Wald-Gewichte** (50 Bäume, mindestens 5 je Blatt, Merkmalsauswahl 0,5, nicht abgestimmt) | **Vormessung mit scikit-learn, nicht live**; die Zahlen stehen in `gws_vormessung.json` |

Die App enthält **kein scikit-learn zur Laufzeit**. Vier Experimente:

1. **Gewichte von Hand:** fünf Fälle x = 1, …, 5 mit den Nachfragen 10, 12, 15, 30, 32; Regler für Anfrage x, τ, k und Bandbreite; Balken der Gewichte, gewichtete Verteilungsfunktion, Bestellmenge.
2. **Welt wählen (Kernfrage):** Welt, p, n, τ und Zufalls-Seed einstellen; Balken = Vormessung (Median über 30 Seeds, Fehlerbalken Minimum bis Maximum), Rauten = die eine Live-Instanz.
3. **Fluch der Dimension:** p ∈ {1, 2, 3, 5, 10} bei fester Datenmenge, live über fünf feste Instanzen (Seeds 100000 bis 100004, 1000 Testfälle je Instanz).
4. **k und Bandbreite:** Mehrkosten jedes festen Rasterwerts gegen die Leave-one-out-Wahl und die im Nachhinein beste feste Wahl (gleiche fünf Instanzen).

Zufall kommt nur aus SplitMix64 (`gws_rng.py`, aus `saa-demo` kopiert), nicht aus Python-`hash()` oder `random`; auch der Zufalls-Seed des Knopfes „Neue Instanz generieren“ ist das erste Element des SplitMix64-Stroms zum aktuellen Zufalls-Seed. Die Rechenzeit hängt vor allem an n (der erste Aufruf bei n = 1000 rechnet Experiment 3 über 25 Instanzen und braucht deutlich länger als bei n = 100; die App zeigt einen Spinner, Ergebnisse werden zwischengespeichert).

## Befunde (gemessen, keine Behauptungen)

**Vormessung** heißt: 30 feste Seeds, 5000 Testfälle, Median der Mehrkosten gegen das Orakel in %, Rohausgaben `vormessung/out_eo_v3_gewichtet.txt` (KQ, direkt, kNN, Wald, gewähltes k) und `vormessung/out_eo_v3b_kern.txt` (Kern); eingefroren in `gws_vormessung.json`, im Test exakt geprüft. Alle Vormessungs-Zahlen unten: n = 1000, τ = 0,95, p = 3 / p = 10, wenn nichts anderes steht.

| Frage | Befund | Test |
|---|---|---|
| Wahrheit **quadratisch**: Was kosten die Regeln? | KQ + empirisch 134,9 / 140,0; direkt 136,5 / 148,1; **kNN 30,4 / 81,1; Kern 29,6 / 80,8; Wald 29,7 / 32,9** | `test_claims.py::test_quadratic_truth_n1000_tau95_p3_and_p10` |
| Wahrheit **linear**: Was kostet der Verzicht auf das Modell? | KQ + empirisch **0,3 / 0,7**; direkt 0,6 / 2,2; kNN 9,3 / 23,2; Kern 9,1 / 24,8; Wald 12,4 / 19,3 | `test_linear_truth_n1000_tau95_p3_and_p10` |
| Streuung linear und nicht monotone Streuung | Streuung linear: KQ + empirisch 32,5 / 33,1, direkt **7,5 / 9,7**, kNN 18,6 / 44,4, Kern 22,4 / 45,5, Wald 23,1 / 41,4. Nicht monoton (\|x₁\|): KQ + empirisch 19,3 / 19,0, direkt 20,7 / 21,8, **kNN 12,1 / 29,5**, Kern 14,3 / 32,8, Wald 17,0 / 28,1 | `test_other_two_worlds_n1000_tau95` |
| In wie vielen der 48 Zellen (4 Welten × p ∈ {3, 10} × n ∈ {100, 1000} × τ ∈ {0,5; 0,8; 0,95}) gewinnen die Gewichte? | Beste Gewichte (kNN oder Kern) gegen beste parametrische Regel (KQ + empirisch oder direkt), Unterschied größer als max(1 Prozentpunkt, 5 % des größeren Werts): **parametrisch vorn in 36 Zellen, Gewichte vorn in 9** (8 quadratische Welt, 1 nicht monotone Streuung bei p = 3, n = 1000, τ = 0,95), **gleichauf in 3** (quadratisch, p = 10) | `test_count_of_cells_where_each_side_wins_by_the_rule_of_the_verdict` |
| Wo liegt der Wald vorn? | In 6 von 48 Zellen um mehr als diese Grenze unter allen anderen vier Regeln; alle 6 quadratisch mit p = 10 (je 3 bei n = 100 und n = 1000, τ = 0,5 / 0,8 / 0,95) | `test_forest_is_clearly_best_in_six_cells_all_quadratic_with_p10` |
| **Wenig Daten, viele Merkmale** | p = 10, n = 100, Welten linear, Streuung linear, \|x₁\|, τ = 0,5 / 0,8 / 0,95: kNN- und Wald-Gewichte kosten **31,4 bis 75,0 %** Mehrkosten (18 Werte) | `test_p10_n100_weights_lose_between_31_and_75_percent_in_three_worlds` |
| Quadratisch, n = 100, τ = 0,95 | p = 3: KQ + empirisch 146,2, kNN 85,7, Wald 89,4. p = 10: KQ + empirisch 149,8, kNN 150,9, **Wald 95,5** | `test_quadratic_p3_n100_tau95_and_p10_n100_tau95` |
| Gauß-Kern gegen kNN | In **39 von 48 Zellen** liegen die Mediane höchstens 4 Prozentpunkte auseinander. Die übrigen 9 (7 quadratisch, 2 Streuung linear) gehen in beide Richtungen: quadratisch, p = 3, n = 1000, τ = 0,5 / 0,8: Kern 14,6 / 16,5 gegen kNN 22,2 / 23,5; quadratisch, p = 3, n = 100, τ = 0,95: Kern 94,1 gegen kNN 85,7 | `test_kernel_beats_knn_at_quadratic_p3_n1000_for_tau_05_and_08`, `test_kernel_and_knn_differ_by_at_most_four_points_in_most_cells` |
| Der Nachbau bestätigt die Aussagen der Vormessung (live, quadratisch, p = 3, n = 1000, τ = 0,95) | Über fünf feste Instanzen: kNN und Kern kosten weniger als die Hälfte der Mehrkosten von KQ + empirisch; direkt in der falschen Klasse bleibt über 90 % von KQ + empirisch | `test_live_quadratic_truth_n1000_p3_weights_cost_less_than_half_of_the_linear_rule` |

**Experiment 3 (live, fünf feste Instanzen, Seeds 100000 bis 100004, 1000 Testfälle, τ = 0,95; Median der Mehrkosten in %, Messung dieser README, ein Lauf unter Windows).** Im Test stehen nur die Ungleichungen daneben, nicht die Ziffern:

| Welt, n | Regel | p = 1 | p = 2 | p = 3 | p = 5 | p = 10 |
|---|---|---|---|---|---|---|
| linear, n = 1000 | KQ + empirisch | 0,0 | −0,1 | 0,0 | −0,1 | −0,1 |
| linear, n = 1000 | kNN | 3,4 | 6,5 | 8,7 | 14,4 | 25,5 |
| quadratisch, n = 100 | KQ + empirisch | 139,0 | 150,0 | 145,3 | 139,9 | 150,1 |
| quadratisch, n = 100 | kNN | 68,7 | 61,8 | 63,5 | 123,5 | 165,2 |

Tests: `test_live_linear_truth_parametric_rule_stays_near_the_oracle_while_the_weights_rise_with_p` (KQ + empirisch bleibt in jedem p unter 1 Prozentpunkt vom Orakel, kNN steigt von unter 6 auf über 15 % und monoton) und `test_live_quadratic_truth_n100_weights_win_at_p_up_to_3_and_lose_that_edge_at_p10` (bei p = 1, 2, 3 kosten kNN und Kern weniger als 60 % der Mehrkosten von KQ + empirisch, bei p = 10 liegt kNN über 80 % davon).
Kleine negative Werte sind Rauschen der Testfälle (die Mehrkosten sind auf einer endlichen Testmenge gemessen).

**Experiment 4 (live, quadratisch, p = 3, n = 1000, τ = 0,95, fünf feste Instanzen, Messung dieser README):** kNN, Median der Mehrkosten je fester Wahl k = 5 / 10 / 20 / 40 / 80 / 160: 68,8 / 32,6 / 25,0 / 25,9 / 32,0 / 46,9 %; die Leave-one-out-Wahl kostet 26,3 %, die im Nachhinein beste feste Wahl je Instanz 25,0 %. Kern: Leave-one-out-Wahl 22,5 %, im Nachhinein beste 22,5 %.
Der Test `test_live_choice_experiment_leave_one_out_is_never_below_the_hindsight_best_and_close_to_it` prüft nur: Leave-one-out nie unter der Nachhinein-Wahl (nach Konstruktion), Abstand unter 6 Prozentpunkten, ein Rand des Rasters mehr als 2,5-mal so teuer wie das beste feste.

**Zahl → Test, Gesamtbild:** Die Vormessungs-Zahlen stehen exakt in `tests/test_claims.py`; die Live-Zahlen (Experimente 3 und 4) stehen nicht als Ziffern im Test; die Zahlen in „Befunde und Korrekturen gegenüber dem Plan“ zum Nachbau (30 Seeds, 2000 Testfälle) sind **Messung dieser README, nicht im Test**.

## Ehrliche Grenzen

- **Eine Familie von Instanzen:** Normalverteiltes Rauschen, Stutzung bei 0, standardnormale Merkmale, vier Welten. Andere Verteilungen, schiefe oder korrelierte Merkmale, abhängige Fälle sind nicht gemessen.
- **Der Wald ist nicht abgestimmt** (50 Bäume, Blattgröße 5, Merkmalsauswahl 0,5) und **nicht live**: Die Wald-Zahlen sind eine Vormessung mit scikit-learn. Sie sagen nichts über Zufallswälder im Allgemeinen. Die Gewichte sind nicht regularisiert; der Kern ist ein Gauß-Kern mit einer Bandbreite für alle Richtungen, der Abstand zählt alle p Merkmale gleich.
- **Live-Zahlen sind klein gerechnet:** fünf feste Instanzen und 1000 Testfälle gegen 30 Seeds und 5000 in der Vormessung; eine einzelne Live-Instanz weicht vom Median ab (siehe Spalten Minimum und Maximum). Der Nachbau nutzt einen anderen Zufallsstrom (SplitMix64) als die Vormessung (numpy-Standardgenerator), die Ziffern stimmen deshalb nicht überein.
- **Die „Gleichauf-Grenze“** (Unterschied höchstens max(1 Prozentpunkt, 5 % des größeren Werts)) ist eine Festlegung dieser Demo für das Verdikt, keine statistische Schwelle; die Zählung 36 / 9 / 3 hängt an ihr.
- **Die Leave-one-out-Wahl auf einem Raster** ist eine von vielen Wahlregeln; die „im Nachhinein beste feste Wahl“ nutzt die Testfälle und ist nicht ausführbar, sie ist nur ein Maßstab.
- **Quellen:** Bertsimas/Kallus und Ban/Rudin sind nicht im Volltext gelesen; die Inhaltsangaben stützen sich auf Zusammenfassungen einer Websuche, der Wortlaut der Abstracts ist nicht geprüft (im arXiv-Abstract von Bertsimas/Kallus stehen lokale Regression, CART und Random Forests; Nachbarn und Kerne erst im Haupttext; bei Ban/Rudin ist nicht geprüft, welche Verfahren genau vorkommen). Satznummern, Voraussetzungen und Experimente der Quellen sind nicht geprüft. Die Zahlen dieser Demo sind weder Literaturwerte noch Bestätigungen von Sätzen. Elmachtoub/Lam/Zhang/Zhao (Volltext gelesen) zeigen asymptotisch für eine parametrische Verteilungsfamilie, dass SAA ohne Modell bei Fehlspezifikation gewinnt; ihre kontextuellen Sätze vergleichen nur zweistufiges und integriertes Lernen. Die gewichtete Stichprobe mit Merkmalen fällt in keinen dieser Fälle, die endlichen Messungen hier ersetzen keinen Satz.
- **Nicht gemessen:** Rechenzeit (nur qualitativ: wächst mit n), Merkmalsauswahl oder -skalierung, andere Kerne, Regularisierung der Gewichte, mehr als drei Werte von τ.
- **Browser:** Die Plotly-Diagramme haben keine Auswahlfunktion; geprüft ist die App im AppTest und mit einem echten Browser-Durchlauf (siehe Verifikation).

## Befunde und Korrekturen gegenüber dem Plan

- **Kern ≈ kNN stimmt in 39 von 48 Zellen, nicht überall mit einer einzigen Ausnahme.** Der Plan nannte als Ausnahme quadratisch, p = 3, n = 1000, τ = 0,5 / 0,8. Die Zählung über alle 48 Zellen zeigt 9 Zellen mit mehr als 4 Prozentpunkten Unterschied, in beide Richtungen (Tabelle oben).
- **Gewichte gewinnen seltener, als die Kernbotschaft nahelegt:** Parametrisch vorn in 36 von 48 Zellen, Gewichte vorn in 9, gleichauf in 3. Der Plan formulierte „gewinnt, wenn die Form unbekannt ist“; belegt ist das nur für die quadratische Welt (8 von 12 Zellen dort) und eine Zelle der nicht monotonen Streuung.
- **Wald nur eingefroren:** Auftragsgemäß kein scikit-learn zur Laufzeit; der im Plan genannte Orakel-Test „Waldgewichte gegen Zählung der Blattgemeinschaft“ entfällt, weil die Demo keine Waldgewichte rechnet. Alle anderen Orakel des Plans stehen (siehe Verifikation).
- **Experiment 3 live statt nur Tabelle:** Der Plan sah p = 3 → 10 aus der Vormessung vor. Die App rechnet zusätzlich p ∈ {1, 2, 3, 5, 10} live über fünf Instanzen; die Vormessung bei p = 3 und p = 10 steht darunter als Tabelle.
- **Verdikt in Mehrkosten und Prozentpunkten:** Der Plan formulierte „sparen x %“; die Basis solcher Prozentangaben ist uneindeutig (Mehrkosten gegen das Orakel sind selbst Prozent). Die App nennt Mehrkosten in % und Unterschiede in Prozentpunkten.
- **Nachbau gegen Vormessung (30 Seeds 0 bis 29, 2000 Testfälle, τ = 0,95, Median in %, Messung dieser README, nicht im Test):**

| Welt, p, n | KQ + empirisch | direkt | kNN | Kern |
|---|---|---|---|---|
| quadratisch, 3, 1000 (Vormessung 134,9 / 136,5 / 30,4 / 29,6) | 140,0 | 142,5 | 28,6 | 29,1 |
| linear, 3, 1000 (Vormessung 0,3 / 0,6 / 9,3 / 9,1) | 0,3 | 1,1 | 10,0 | 9,1 |
| linear, 10, 100 (Vormessung 9,9 / 33,0 / 42,1 / 42,2) | 8,5 | 33,2 | 43,6 | 44,1 |
| Streuung linear, 3, 1000 (Vormessung 32,5 / 7,5 / 18,6 / 22,4) | 34,5 | 8,4 | 18,0 | 22,1 |
| nicht monoton, 3, 1000 (Vormessung 19,3 / 20,7 / 12,1 / 14,3) | 18,1 | 19,4 | 12,2 | 13,6 |

Der Nachbau trifft in diesen fünf Zellen die Größenordnung und denselben Sieger (parametrische Regel gegen Gewichte), nicht die Ziffern und nicht jede Rangfolge innerhalb der beiden Gewichtsverfahren (anderer Zufallsstrom, 2000 statt 5000 Testfälle).
- **Footer:** Die Linie hat noch keine Website-Spezifikation; der Footer verlinkt vorläufig `demos.html` und ist vor der Website-Einbindung auf die Linie umzustellen.

## Verifikation

- **Kern:** gewichtetes Quantil und Gewichte gegen das **Handbeispiel von Hand** (k = 3: 30; Gauß-Kern mit Bandbreite 1: Gewichte 0,0359 / 0,1963 / 0,3953 / 0,2928 / 0,0798, Quantil 30; k = n; k = 1; τ = 1), gegen Eigenschaften (Monotonie in τ, Skalierung der Nachfragen, Vertauschen der Beobachtungen, Gewichte summieren zu 1, große Bandbreite gibt gleiche Gewichte) und Randfälle (n = 1, n = 2, k > n, alle x gleich, Gleichstände, winzige Bandbreite ohne 0/0).
- **Orakel** (`tests/test_oracle_gewichtete_stichprobe.py`, je ein anderer Rechenweg): gewichtetes Quantil gegen Aufzählung mit exakten Brüchen (kNN, 40 Instanzen mit Gleichständen) und mit `math`/`fsum` (Kern, 40 Instanzen); Leave-one-out-Verluste gegen eine Python-Schleife; Quantilregression gegen die Aufzählung aller Geraden durch zwei Punkte (25 Instanzen); Kleinste Quadrate gegen `scipy.stats.linregress`; wahres Quantil gegen numerische Minimierung der erwarteten Kosten in allen vier Welten.
- **Zufall:** SplitMix64 gegen eine Schleife mit Python-Ganzzahlen und den bekannten Prüfvektor.
- **Verdikte** als reine Funktionen an handgesetzten Werten (`tests/test_verdicts.py`); **Zahlen** der Befunde (`tests/test_claims.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Slider am Minimum und Maximum, jede Welt, jedes p, jedes τ, Permalink, Zurücksetzen), Layout-Test (Schnellstart und Instanz-Einstellungen in der Seitenleiste, Regler des Handbeispiels im Hauptbereich, aktives Preset hervorgehoben, jeder Regler in `SETTING_SPECS` oder `HAND_SPECS`), Footer-Test, Laufzeit-Importe (`tests/test_runtime_imports.py`).
- **Testzahl:** 156 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Grenzen, Formeln |
| `gws_algorithm.py` | Bestellregeln: Kleinste Quadrate, Quantilregression, kNN- und Kernregel, Kosten, Mehrkosten |
| `gws_constants.py` | Welten, τ, Raster, Reglergrenzen, Handbeispiel, Presets, Farben |
| `gws_evaluation.py` | Handbeispiel, Live-Analyse, Wiederholungen über feste Instanzen, Vormessung laden, Verdikte |
| `gws_layout.py` | Fester Kopf (Stylesheet) |
| `gws_presets.py` | Permalink (Begrenzen, Einrasten), Schnellstart in der Seitenleiste (aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `gws_rng.py` | SplitMix64 (aus `saa-demo` kopiert) |
| `gws_scenario.py` | Welten, Orakel-Quantil, Daten |
| `gws_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `gws_weights.py` | Abstände, Nachbarn, kNN- und Kerngewichte, gewichtetes Quantil, Leave-one-out |
| `gws_vormessung.json` | eingefrorene Bänder der Vormessung (48 Zellen, inklusive Wald) |
| `tools/freeze_vormessung.py` | erzeugt die JSON-Datei aus den Rohausgaben der Vormessung (einmal gelaufen) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Handbeispiel, Orakel, Eigenschaften, Randfälle, Verdikte, Zahlen, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Zufallswald-Gewichte live** (kein scikit-learn zur Laufzeit; ein numpy-Wald ist nicht Teil der Demo).
- **PDF-Export** der Ergebnisse.
- **Regularisierte oder abgestimmte Gewichte**, andere Kerne, gewichtete Abstände, Merkmalsauswahl.
- **Weitere Welten und Verteilungsfamilien**, Merkmale mit Korrelation, Zeitreihen.
- **Konfidenzaussagen** für einzelne Live-Instanzen (nur Median und Bereich über feste Instanzen).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html).
