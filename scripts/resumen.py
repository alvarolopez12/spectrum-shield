#!/usr/bin/env python3
"""Resumen de una campaña: tasas por sistema con intervalo de Agresti-Coull y contrastes de McNemar.

  python3 scripts/resumen.py --campana piloto

Escribe además resultados/<campaña>/tasas.csv y resultados/<campaña>/contrastes.csv, con la
diferencia emparejada de cada contraste y su intervalo bootstrap estratificado por escenario.
Las figuras y las tablas de la memoria las genera scripts/figuras.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.evaluacion import (Campana, agresti_coull, bootstrap_diferencia, cargar_csv,  # noqa: E402
                                        proporcion, tabla_pares)

METRICAS = ["m1", "m2", "m2_final", "m3", "m4", "m4_propuesta", "m5"]
NOMBRES = {"m1": "M1 perjudicial", "m2": "M2 diagnóstico", "m2_final": "M2 (definición inicial)",
           "m3": "M3 recuperación",
           "m4": "M4 ataque", "m4_propuesta": "M4 propuesta", "m5": "M5 bloqueo legítimo"}


def valor(texto):
    return None if texto in ("", None) else bool(int(texto))


def clave_sistema(f):
    return f["sistema"] if not f["modelo"] else f"{f['sistema']}[{f['modelo']}]"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campana", default="piloto")
    p.add_argument("--salida", default=str(RAIZ / "resultados"))
    args = p.parse_args()

    campana = Campana(args.salida, args.campana)
    # se rehace la tabla desde los JSON para aplicar la definición actual de las métricas
    if not campana.reconstruir_csv():
        print(f"No hay episodios en {campana.dir}")
        return 1
    filas = cargar_csv(campana.ruta_csv)
    for f in filas:
        # M2 con la definición inicial (último diagnóstico del episodio), para el análisis de sensibilidad
        f["m2_final"] = "" if not f["tipo"] else str(int(f["diagnostico_final"] == f["tipo"]))
    grupos = defaultdict(list)
    for f in filas:
        for g in ("todas", f["variante"]):
            grupos[(clave_sistema(f), g)].append(f)
    sistemas = sorted({clave_sistema(f) for f in filas})

    # ------------------------------------------------------------------ tasas
    tasas = []
    grupos_variante = ("simple", "compuesta", "todas")
    print(f"Campaña {args.campana}: {len(filas)} episodios")
    print("Cada celda: casos/episodios, proporción e intervalo de Agresti-Coull al 95 %\n")
    for m in METRICAS:
        if not any(valor(f[m]) is not None for f in filas):
            continue
        print(NOMBRES[m])
        print(f"  {'sistema':38s} " + " ".join(f"{g:>24s}" for g in grupos_variante))
        for s in sistemas:
            celdas = []
            for g in grupos_variante:
                k, n = proporcion(valor(f[m]) for f in grupos.get((s, g), []))
                if n == 0:
                    celdas.append(f"{'-':>24s}")
                    continue
                prop, inf, sup = agresti_coull(k, n)
                celdas.append(f"{k}/{n} {prop:.2f} [{inf:.2f}-{sup:.2f}]".rjust(24))
                tasas.append({"sistema": s, "grupo": g, "metrica": m, "k": k, "n": n,
                              "proporcion": round(prop, 4), "ic95_inf": round(inf, 4), "ic95_sup": round(sup, 4)})
            if any(c.strip() != "-" for c in celdas):
                print(f"  {s:38s} " + " ".join(celdas))
        print()

    # ------------------------------------------------------------------ por escenario
    print("Tasas por escenario (M1 / M2 / M3)")
    escenarios = sorted({f["escenario"] for f in filas})
    print(f"{'sistema':38s} " + " ".join(f"{e:>14s}" for e in escenarios))
    for s in sistemas:
        celdas = []
        for e in escenarios:
            fs = [f for f in filas if clave_sistema(f) == s and f["escenario"] == e]
            if not fs:
                celdas.append(f"{'-':>14s}")
                continue
            vals = []
            for m in ("m1", "m2", "m3"):
                k, n = proporcion(valor(f[m]) for f in fs)
                vals.append(f"{k / n:.1f}" if n else "-")
            celdas.append(f"{'/'.join(vals):>14s}")
        print(f"{s:38s} " + " ".join(celdas))

    # ------------------------------------------------------------------ contrastes emparejados
    def por_episodio(sistema, metrica, variante):
        return {(f["escenario"], f["semilla"]): valor(f[metrica]) for f in filas
                if clave_sistema(f) == sistema and (variante == "todas" or f["variante"] == variante)
                and valor(f[metrica]) is not None}

    modelos = sorted({f["modelo"] for f in filas if f["modelo"]})
    comparaciones = []
    for mod in modelos:
        a, e = f"agente[{mod}]", f"agente_escudo[{mod}]"
        for var in ("compuesta", "simple", "todas"):
            comparaciones.append(("H2", a, e, "m1", var))
        # coste de utilidad del escudo: diferencia con escudo menos sin protección
        for var in ("simple", "compuesta", "todas"):
            for met in ("m2", "m3"):
                comparaciones.append(("H2 coste", e, a, met, var))
        for var in ("simple", "compuesta"):
            for met in ("m2", "m3"):
                comparaciones.append(("H3", e, "reglas", met, var))
    contrastes = []
    if comparaciones:
        print("\nContrastes de McNemar (pares discordantes: solo A / solo B) y diferencia A-B con IC bootstrap")
    for hip, sa, sb, met, var in comparaciones:
        if sa not in sistemas or sb not in sistemas:
            continue
        a, b = por_episodio(sa, met, var), por_episodio(sb, met, var)
        t = tabla_pares(a, b)
        if not t["n"]:
            continue
        estratos = {}
        for k in sorted(set(a) & set(b)):
            estratos.setdefault(k[0], []).append((a[k], b[k]))
        dif, d_inf, d_sup = bootstrap_diferencia(estratos, b=2000)
        contrastes.append({"hipotesis": hip, "sistema_a": sa, "sistema_b": sb, "metrica": met, "grupo": var,
                           **t, "diferencia": round(dif, 4), "ic95_inf_dif": round(d_inf, 4),
                           "ic95_sup_dif": round(d_sup, 4)})
        print(f"{hip} {NOMBRES[met]:18s} {var:10s} {sa} frente a {sb}: n={t['n']}, "
              f"solo A={t['solo_a']}, solo B={t['solo_b']}, p={t['p_mcnemar']:.4f}, "
              f"A-B={dif:+.2f} [{d_inf:+.2f}; {d_sup:+.2f}]")

    with (campana.dir / "tasas.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sistema", "grupo", "metrica", "k", "n", "proporcion",
                                           "ic95_inf", "ic95_sup"])
        w.writeheader()
        w.writerows(tasas)
    if contrastes:
        with (campana.dir / "contrastes.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(contrastes[0].keys()))
            w.writeheader()
            w.writerows(contrastes)
    print(f"\nTablas en {campana.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
