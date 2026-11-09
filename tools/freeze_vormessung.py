"""Friert die Rohausgaben der Vormessung als Bänder in `gws_vormessung.json` ein (einmal gelaufen, Ergebnis im Repo).

Aufruf:  python tools/freeze_vormessung.py <out_eo_v3_gewichtet.txt> <out_eo_v3b_kern.txt> [--out gws_vormessung.json]

Eingaben (nur gelesen, Wegwerf-Messungen mit festen Seeds, 30 Seeds, 5000 Testfälle, scikit-learn für den Wald):
  out_eo_v3_gewichtet.txt   KQ + empirisch, direkt (Pinball), kNN-Gewichte, Wald-Gewichte, gewähltes k (Median); Zeilen "p n tau | 4 Bänder | k"
  out_eo_v3b_kern.txt       kNN gegen Gauß-Kern (Bandbreite per Leave-one-out); Zeilen "p = P, n = N, tau = T: kNN <Band> | Kern <Band>"
Ein Band steht als "Median [Min, Max]" in Prozent Mehrkosten gegen das Orakel. Die kNN-Spalten beider Dateien sind dieselbe Messung; das Skript bricht ab,
wenn sie sich unterscheiden. Die Wald-Zahlen sind eine Vormessung mit scikit-learn und werden in der Demo nie live gerechnet."""

import argparse
import json
import re
import sys

BAND = r"\s*(-?[\d.]+)\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+)\]"
ROW_V3 = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s*\|" + (BAND + r"\s*\|") * 3 + BAND + r"\s*\|\s*(\d+)\s*$")
ROW_V3B = re.compile(r"^\s*p\s*=\s*(\d+),\s*n\s*=\s*(\d+),\s*tau\s*=\s*([\d.]+):\s*kNN" + BAND + r"\s*\|\s*Kern" + BAND + r"\s*$")
WELT = re.compile(r"^--- Welt (\S+)\s*$")


def band(med, lo, hi):
    return {"median": float(med), "min": float(lo), "max": float(hi)}


def parse_v3(path):
    welt, cells = None, {}
    for line in open(path, encoding="utf-8"):
        m = WELT.match(line)
        if m:
            welt = m.group(1)
            continue
        m = ROW_V3.match(line)
        if m and welt:
            g = m.groups()
            p, n, tau = int(g[0]), int(g[1]), float(g[2])
            vals = [band(*g[3 + 3 * i: 6 + 3 * i]) for i in range(4)]
            cells[(welt, p, n, tau)] = {"kq": vals[0], "direkt": vals[1], "knn": vals[2], "wald": vals[3], "k_median": int(g[15])}
    return cells


def parse_v3b(path):
    welt, cells = None, {}
    for line in open(path, encoding="utf-8"):
        m = WELT.match(line)
        if m:
            welt = m.group(1)
            continue
        m = ROW_V3B.match(line)
        if m and welt:
            g = m.groups()
            cells[(welt, int(g[0]), int(g[1]), float(g[2]))] = {"knn": band(*g[3:6]), "kern": band(*g[6:9])}
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("v3")
    ap.add_argument("v3b")
    ap.add_argument("--out", default="gws_vormessung.json")
    a = ap.parse_args()
    v3 = parse_v3(a.v3)
    v3b = parse_v3b(a.v3b)
    if len(v3) != 48 or len(v3b) != 48:
        sys.exit(f"erwartet 48 Zellen je Datei, gefunden {len(v3)} und {len(v3b)}")
    cells = []
    for key in sorted(v3):
        if v3[key]["knn"] != v3b[key]["knn"]:
            sys.exit(f"kNN-Spalten weichen ab bei {key}: {v3[key]['knn']} gegen {v3b[key]['knn']}")
        welt, p, n, tau = key
        cells.append({"welt": welt, "p": p, "n": n, "tau": tau, **v3[key], "kern": v3b[key]["kern"]})
    out = {
        "quelle": "vormessung/out_eo_v3_gewichtet.txt (kq, direkt, knn, wald, k_median) und vormessung/out_eo_v3b_kern.txt (kern); Wald: Vormessung mit scikit-learn, nicht live",
        "seeds": 30, "testfaelle": 5000, "einheit": "Mehrkosten gegen das Orakel in Prozent (Median, Minimum, Maximum über die Seeds)",
        "zellen": cells,
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(cells)} Zellen nach {a.out}")


if __name__ == "__main__":
    main()
