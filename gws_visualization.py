"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gws_constants as C
import gws_evaluation as ev

METHOD_COLORS = {"kq": C.COLOR_KQ, "direkt": C.COLOR_DIREKT, "knn": C.COLOR_KNN, "kern": C.COLOR_KERN, "wald": C.COLOR_WALD}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_hand_weights(r: ev.HandResult):
    """Gewichte der fünf vergangenen Fälle für die Anfrage x: kNN (gleiche Gewichte für die k nächsten) neben dem Gauß-Kern (je näher, desto schwerer)."""
    labels = [f"x = {x:g}<br>D = {d:g}" for x, d in zip(C.HAND_X, C.HAND_D)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=list(r.w_knn), name=f"kNN, k = {r.k}", marker_color=C.COLOR_KNN, hovertemplate="%{x}: Gewicht %{y:.4f}<extra>kNN</extra>"))
    fig.add_trace(go.Bar(x=labels, y=list(r.w_kern), name=f"Gauß-Kern, h = {ev.de(r.h)}", marker_color=C.COLOR_KERN, hovertemplate="%{x}: Gewicht %{y:.4f}<extra>Kern</extra>"))
    fig.update_layout(barmode="group", title=dict(text=f"Gewichte der fünf Fälle für die Anfrage x = {ev.de(r.x)}", font=dict(size=14)))
    fig.update_yaxes(title_text="Gewicht", range=[0, 1.05])
    return _base(fig)


def build_hand_cdf(r: ev.HandResult):
    """Gewichtete Verteilungsfunktion der Nachfrage: das tau-Quantil ist der erste Wert, an dem die Kurve tau erreicht."""
    fig = go.Figure()
    for name, w, q, color in (("kNN", r.w_knn, r.q_knn, C.COLOR_KNN), ("Gauß-Kern", r.w_kern, r.q_kern, C.COLOR_KERN)):
        pts = ev.hand_cdf(w)
        xs = [p[0] for p in pts]
        ys = [min(1.0, p[1]) for p in pts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=3, shape="hv"), marker=dict(size=8), name=f"{name}: Quantil {q:g}",
                                 hovertemplate="D = %{x:g}: kumuliertes Gewicht %{y:.3f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=[q], y=[r.tau], mode="markers", marker=dict(size=14, color=color, symbol="diamond", line=dict(color="white", width=1)),
                                 showlegend=False, hovertemplate=f"{name}: bestelle %{{x:g}}<extra></extra>"))
    fig.add_hline(y=r.tau, line=dict(color="#888", dash="dash"), annotation_text=f"τ = {ev.de(r.tau, 2)}", annotation_position="top left")
    fig.update_xaxes(title_text="Nachfrage D (Bestellmenge)", range=[8, 34])
    fig.update_yaxes(title_text="kumuliertes Gewicht", range=[0, 1.08])
    return _base(fig)


def build_methods(cell: dict, live: dict | None):
    """Mehrkosten gegen das Orakel je Regel: Balken = Median der Vormessung (30 Seeds), Fehlerbalken = Minimum bis Maximum; Rauten = diese Live-Instanz.
    Der Wald ist schraffiert und nur eingefroren."""
    names = [ev.METHODEN_NAMEN[m] for m in ev.METHODEN]
    med = [cell[m]["median"] for m in ev.METHODEN]
    up = [cell[m]["max"] - cell[m]["median"] for m in ev.METHODEN]
    dn = [cell[m]["median"] - cell[m]["min"] for m in ev.METHODEN]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=med, name=f"Vormessung: Median ({C.VORMESSUNG_SEEDS} Seeds), Fehlerbalken Minimum bis Maximum", marker_color=[METHOD_COLORS[m] for m in ev.METHODEN],
        marker_pattern_shape=["", "", "", "", "/"], error_y=dict(type="data", symmetric=False, array=up, arrayminus=dn, thickness=1.5, width=6),
        hovertemplate="%{x}<br>Median %{y:.1f} %<extra></extra>",
    ))
    if live is not None:
        fig.add_trace(go.Scatter(
            x=[ev.METHODEN_NAMEN[m] for m in ev.LIVE_METHODEN], y=[live[m] for m in ev.LIVE_METHODEN], mode="markers", name="diese Instanz (live)",
            marker=dict(size=13, symbol="diamond", color="#111", line=dict(color="white", width=1)), hovertemplate="%{x}<br>live %{y:.1f} %<extra></extra>",
        ))
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)")
    fig.update_xaxes(tickangle=0)
    return _base(fig, height=400, legend_y=-0.3)


def build_dim(rows: list[dict]):
    """Mehrkosten gegen die Zahl der Merkmale p bei fester Datenmenge: Median über die festen Instanzen, Fehlerbalken Minimum bis Maximum."""
    xs = [str(r["p"]) for r in rows]
    fig = go.Figure()
    for m in ev.LIVE_METHODEN:
        med = [r[m]["median"] for r in rows]
        fig.add_trace(go.Scatter(
            x=xs, y=med, mode="lines+markers", name=ev.METHODEN_NAMEN[m], line=dict(color=METHOD_COLORS[m], width=3, dash="dash" if m == "direkt" else "solid"), marker=dict(size=8),
            error_y=dict(type="data", symmetric=False, array=[r[m]["max"] - r[m]["median"] for r in rows], arrayminus=[r[m]["median"] - r[m]["min"] for r in rows],
                         thickness=1, width=4, color=METHOD_COLORS[m]),
            hovertemplate="p = %{x}: %{y:.1f} %<extra>" + ev.METHODEN_NAMEN[m] + "</extra>",
        ))
    fig.update_xaxes(title_text="Zahl der Merkmale p", type="category")
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)")
    return _base(fig, height=400, legend_y=-0.3)


def build_choice(res: dict):
    """Mehrkosten je fester Wahl von k (links) und Bandbreite (rechts), Median über die festen Instanzen; waagerecht: Leave-one-out-Wahl und im Nachhinein beste feste Wahl."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("kNN: Nachbarzahl k", "Gauß-Kern: Bandbreite (auf p = 3 bezogen)"), horizontal_spacing=0.12)
    for col, (name, color) in enumerate((("knn", C.COLOR_KNN), ("kern", C.COLOR_KERN)), start=1):
        r = res[name]
        # Kern: Anzeige im Raster HGRID (bezogen auf p = 3), nicht die auf p umgerechneten Bandbreiten
        xs = [f"{h:g}" for h in C.HGRID] if name == "kern" else [str(g) for g in r["grid"]]
        fig.add_trace(go.Scatter(x=xs, y=r["fixed"], mode="lines+markers", line=dict(color=color, width=3), marker=dict(size=8), name="feste Wahl (Median über die Instanzen)",
                                 showlegend=col == 1, hovertemplate="%{x}: %{y:.1f} %<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=xs, y=[r["loo"]] * len(xs), mode="lines", line=dict(color="#c0392b", width=2, dash="dash"), name="Wahl per Leave-one-out (Median)",
                                 showlegend=col == 1, hovertemplate="Leave-one-out: %{y:.1f} %<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=xs, y=[r["best"]] * len(xs), mode="lines", line=dict(color="#555", width=2, dash="dot"), name="im Nachhinein beste feste Wahl (Median)",
                                 showlegend=col == 1, hovertemplate="im Nachhinein beste: %{y:.1f} %<extra></extra>"), row=1, col=col)
        fig.update_xaxes(type="category", row=1, col=col)
    fig.update_yaxes(title_text="Mehrkosten gegen das Orakel (%)", row=1, col=1)
    return _base(fig, height=400, legend_y=-0.3)
