#!/usr/bin/env python3
"""Acción perjudicial (M1) separada por tipo de daño, para el agente sin protección y con escudo.

  python3 scripts/m1_por_tipo.py --campana campana
  python3 scripts/m1_por_tipo.py --campana campana --salida resultados/m1_por_tipo.json

Solo lee las trayectorias de la campaña. M1 agrupa dos tipos de daño de naturaleza distinta:
  - transmitir sobre un usuario primario activo, que el escudo solo evita si la evidencia lo
    detecta (invariante I1);
  - salirse de los límites autorizados de potencia, banda o duración, que el escudo bloquea
    siempre por construcción (invariantes I2 a I4), porque solo dependen de los campos de la acción.
Para cada modelo y variante da la tasa de cada tipo con intervalo de Agresti-Coull al 95 %, el
contraste exacto de McNemar entre el agente sin protección y con escudo y la diferencia emparejada
con intervalo bootstrap estratificado por escenario (10.000 réplicas, semilla 2026), igual que
scripts/datos_cap5.py. Un episodio cuenta en los dos tipos si tiene acciones de los dos.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.evaluacion import (Campana, agresti_coull, bootstrap_diferencia, id_sistema,  # noqa: E402
                                        tabla_pares)

PRIMARIO = "transmision_sobre_primario"
LIMITES = {"banda_no_autorizada", "potencia_excesiva", "duracion_excesiva"}
TIPOS = ("m1", "primario", "limites")
VARIANTES = ("simple", "compuesta", "todas")


def cargar(campana: Campana, sistema: str, modelo: str):
    eps = {}
    carpeta = campana.dir_episodios / id_sistema(sistema, modelo)
    for r in sorted(carpeta.glob("*/s*.json")):
        ep = json.loads(r.read_text(encoding="utf-8"))
        motivos = [set(p["ejecutada"].get("motivos") or []) for p in ep["pasos"] if not p["automatico"]]
        primario = any(PRIMARIO in m for m in motivos)
        limites = any(m & LIMITES for m in motivos)
        m1 = bool(ep["metricas"]["m1"])
        if m1 != (primario or limites):
            raise SystemExit("M1 no cuadra con los motivos en %s" % r)
        eps[(ep["escenario"], ep["semilla"])] = {"variante": ep["variante"], "m1": m1, "primario": primario,
                                                 "limites": limites}
    return eps


def tasa(valores):
    k, n = sum(valores), len(valores)
    p, inf, sup = agresti_coull(k, n)
    return {"k": k, "n": n, "p": p, "inf": inf, "sup": sup}


def resumen(campana: Campana, modelo: str):
    a, b = cargar(campana, "agente", modelo), cargar(campana, "agente_escudo", modelo)
    out = {}
    for tipo in TIPOS:
        for var in VARIANTES:
            claves = sorted(k for k in set(a) & set(b) if var == "todas" or a[k]["variante"] == var)
            da = {k: a[k][tipo] for k in claves}
            db = {k: b[k][tipo] for k in claves}
            t = tabla_pares(da, db)
            estratos = defaultdict(list)
            for k in claves:
                estratos[k[0]].append((da[k], db[k]))
            dif, inf, sup = bootstrap_diferencia(estratos, b=10000)
            out["%s_%s" % (tipo, var)] = {
                "sin_proteccion": tasa(list(da.values())), "con_escudo": tasa(list(db.values())),
                "solo_sin_proteccion": t["solo_a"], "solo_con_escudo": t["solo_b"], "p_mcnemar": t["p_mcnemar"],
                "dif": dif, "inf": inf, "sup": sup}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campana", default="campana")
    ap.add_argument("--modelos", nargs="+", default=["granite4.1:8b", "qwen3.8:27b"])
    ap.add_argument("--salida", help="fichero JSON con las cifras (opcional)")
    args = ap.parse_args()
    campana = Campana(RAIZ / "resultados", args.campana)
    datos = {m: resumen(campana, m) for m in args.modelos}
    for modelo, d in datos.items():
        print("==", modelo)
        for clave, v in d.items():
            sp, ce = v["sin_proteccion"], v["con_escudo"]
            print("  %-18s %3d/%d (%5.1f %%) -> %3d/%d (%5.1f %%)  pares %d / %d  p = %.2g  dif %+.1f [%+.1f; %+.1f]" % (
                clave, sp["k"], sp["n"], 100 * sp["p"], ce["k"], ce["n"], 100 * ce["p"], v["solo_sin_proteccion"],
                v["solo_con_escudo"], v["p_mcnemar"], 100 * v["dif"], 100 * v["inf"], 100 * v["sup"]))
    if args.salida:
        ruta = Path(args.salida)
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        print("escrito", ruta)


if __name__ == "__main__":
    main()
