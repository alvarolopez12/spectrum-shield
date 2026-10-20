#!/usr/bin/env python3
"""Compara campañas que comparten semillas: repeticiones de un mismo sistema o sistemas distintos.

  repeticiones   Varias ejecuciones del mismo sistema y modelo sobre las mismas semillas (por ejemplo,
                 la campaña y dos repeticiones) para medir cuánto varían las métricas de un episodio
                 entre ejecuciones: tasa de cada métrica en cada ejecución, proporción de episodios con
                 el mismo valor en todas, y pass^k (métrica cierta en todas) frente a "en alguna".
  sistemas       Dos sistemas sobre las mismas semillas (por ejemplo, agente_endurecido frente a
                 agente, o frente a agente_escudo): contraste emparejado de McNemar y diferencia con
                 intervalo bootstrap estratificado por escenario.

  python3 scripts/comparar_campanas.py repeticiones --modelo granite4.1:8b --sistema agente \\
      --campanas campana,repeticion_2,repeticion_3
  python3 scripts/comparar_campanas.py sistemas --modelo granite4.1:8b \\
      --a endurecido:agente_endurecido --b campana:agente --escenarios JAM_C,PUE_C,SSDF_C,FALLO_C
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.evaluacion import (Campana, agresti_coull, bootstrap_diferencia, calcular_metricas,  # noqa: E402
                                        id_sistema, tabla_pares)

METRICAS = ("m1", "m2", "m3", "m4", "m4_propuesta", "m5")


def cargar(raiz, campana: str, sistema: str, modelo: str, escenarios):
    camp = Campana(raiz, campana)
    cfg = camp.config()
    out = {}
    for ruta in sorted((camp.dir_episodios / id_sistema(sistema, modelo)).glob("*/s*.json")):
        ep = json.loads(ruta.read_text(encoding="utf-8"))
        if escenarios and ep["escenario"] not in escenarios:
            continue
        out[(ep["escenario"], ep["semilla"])] = calcular_metricas(ep, cfg)
    return out


def fmt(k, n):
    p, lo, hi = agresti_coull(k, n)
    return f"{k}/{n} ({100 * p:.1f} % [{100 * lo:.1f}; {100 * hi:.1f}])"


def repeticiones(args) -> int:
    nombres = [c.strip() for c in args.campanas.split(",")]
    datos = [cargar(args.salida, c, args.sistema, args.modelo, args.escenarios) for c in nombres]
    comunes = sorted(set.intersection(*(set(d) for d in datos)))
    print(f"{args.sistema} [{args.modelo}] en {len(nombres)} ejecuciones; {len(comunes)} episodios comunes\n")
    if not comunes:
        return 1
    for met in METRICAS:
        claves = [k for k in comunes if datos[0][k][met] is not None]
        if not claves:
            continue
        tasas = [fmt(sum(bool(d[k][met]) for k in claves), len(claves)) for d in datos]
        iguales = sum(1 for k in claves if len({bool(d[k][met]) for d in datos}) == 1)
        todas = sum(1 for k in claves if all(d[k][met] for d in datos))
        alguna = sum(1 for k in claves if any(d[k][met] for d in datos))
        print(f"{met:12s} " + " | ".join(tasas))
        print(f"{'':12s} episodios con el mismo valor en todas: {iguales}/{len(claves)} ({100 * iguales / len(claves):.1f} %); "
              f"cierta en todas: {todas}; cierta en alguna: {alguna}")
    return 0


def sistemas(args) -> int:
    ca, sa = args.a.split(":")
    cb, sb = args.b.split(":")
    a = cargar(args.salida, ca, sa, args.modelo, args.escenarios)
    b = cargar(args.salida, cb, sb, args.modelo, args.escenarios)
    comunes = sorted(set(a) & set(b))
    print(f"{sa} ({ca}) frente a {sb} ({cb}) [{args.modelo}]; {len(comunes)} episodios emparejados\n")
    if not comunes:
        return 1
    for met in METRICAS:
        claves = [k for k in comunes if a[k][met] is not None and b[k][met] is not None]
        if not claves:
            continue
        da = {k: bool(a[k][met]) for k in claves}
        db = {k: bool(b[k][met]) for k in claves}
        t = tabla_pares(da, db)
        estr = defaultdict(list)
        for k in claves:
            estr[k[0]].append((da[k], db[k]))
        d, lo, hi = bootstrap_diferencia(estr, b=10000)
        print(f"{met:12s} {sa}: {fmt(sum(da.values()), len(claves))}   {sb}: {fmt(sum(db.values()), len(claves))}")
        print(f"{'':12s} discordantes (solo {sa} / solo {sb}): {t['solo_a']} / {t['solo_b']}; McNemar p = {t['p_mcnemar']:.3g}; "
              f"diferencia {100 * d:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}] puntos")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="modo", required=True)
    r = sub.add_parser("repeticiones")
    r.add_argument("--campanas", required=True, help="campañas separadas por comas (la primera es la referencia)")
    r.add_argument("--sistema", default="agente")
    s = sub.add_parser("sistemas")
    s.add_argument("--a", required=True, help="campaña:sistema, por ejemplo endurecido:agente_endurecido")
    s.add_argument("--b", required=True, help="campaña:sistema, por ejemplo campana:agente")
    for q in (r, s):
        q.add_argument("--modelo", required=True)
        q.add_argument("--escenarios", default=None, help="lista separada por comas; por defecto, todos")
        q.add_argument("--salida", default=str(RAIZ / "resultados"))
    args = p.parse_args()
    args.escenarios = set(x.strip() for x in args.escenarios.split(",")) if args.escenarios else None
    return repeticiones(args) if args.modo == "repeticiones" else sistemas(args)


if __name__ == "__main__":
    sys.exit(main())
