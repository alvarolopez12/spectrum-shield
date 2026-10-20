#!/usr/bin/env python3
"""Cifras del capítulo 5 de la memoria para un modelo, en un fichero JSON.

  python3 scripts/datos_cap5.py --campana campana --modelo granite4.1:8b

Solo lee las trayectorias de la campaña (se puede usar mientras corre) y recalcula las métricas con
la definición actual. Reúne, para las reglas y para el agente sin protección y con escudo del modelo:
  - tasas de M1 a M5 por variante y por escenario, con intervalo de Agresti-Coull al 95 %;
  - contrastes emparejados de H2 (seguridad y coste) y de H3: McNemar exacto y diferencia con
    intervalo bootstrap estratificado por escenario (10.000 réplicas, semilla 2026);
  - éxito de la inyección (M4) por plantilla y por escenario, y sin SSDF_C;
  - bloqueos del escudo por invariante, decisiones del escudo en sombra sobre el agente sin
    protección y acciones perjudiciales que pasan el escudo;
  - validez operativa y coste (duración, turnos, tokens, propuestas no válidas, recordatorios,
    modo seguro, pasos en los que el sistema espera);
  - emparejamiento: pares en los que el escudo no intervino y si sus acciones coinciden;
  - diagnósticos en la ventana del incidente.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.evaluacion import (Campana, agresti_coull, bootstrap_diferencia, calcular_metricas,  # noqa: E402
                                        id_sistema, nombre_seguro, tabla_pares)

SISTEMAS = ("reglas", "agente", "agente_escudo")
VARIANTES = ("simple", "compuesta", "todas")
METRICAS = ("m1", "m1_propuesta", "m2", "m2_final", "m3", "m4", "m4_propuesta", "m5")


def cargar(campana: Campana, modelo: str):
    cfg = campana.config()
    eps = {}
    for s in SISTEMAS:
        carpeta = campana.dir_episodios / id_sistema(s, None if s == "reglas" else modelo)
        for r in sorted(carpeta.glob("*/s*.json")):
            ep = json.loads(r.read_text(encoding="utf-8"))
            m = calcular_metricas(ep, cfg)
            # M2 con la definición inicial (último diagnóstico del episodio), para la sensibilidad
            m["m2_final"] = m["diagnostico_final"] == ep["tipo"]
            eps[(s, ep["escenario"], ep["semilla"])] = (ep, m)
    return eps


def tasa(valores):
    vals = [bool(v) for v in valores if v is not None]
    if not vals:
        return None
    k, n = sum(vals), len(vals)
    p, inf, sup = agresti_coull(k, n)
    return {"k": k, "n": n, "p": p, "inf": inf, "sup": sup}


def decisiones(ep):
    return [json.dumps(p["ejecutada"]["accion"], sort_keys=True) for p in ep["pasos"] if not p["automatico"]]


def datos(eps, modelo):
    def sel(s, var="todas", esc=None):
        return [(ep, m) for (ss, e, _), (ep, m) in eps.items()
                if ss == s and (var == "todas" or ep["variante"] == var) and (esc is None or e == esc)]

    def contraste(sa, sb, met, var):
        def dic(s):
            return {(e, sem): bool(m[met]) for (ss, e, sem), (ep, m) in eps.items()
                    if ss == s and (var == "todas" or ep["variante"] == var) and m[met] is not None}
        a, b = dic(sa), dic(sb)
        comunes = sorted(set(a) & set(b))
        if not comunes:
            return None
        t = tabla_pares(a, b)
        estratos = defaultdict(list)
        for k in comunes:
            estratos[k[0]].append((a[k], b[k]))
        d, inf, sup = bootstrap_diferencia(estratos, b=10000)
        return {"a": sa, "b": sb, "metrica": met, "variante": var, "n": t["n"], "solo_a": t["solo_a"],
                "solo_b": t["solo_b"], "p": t["p_mcnemar"], "dif": d, "inf": inf, "sup": sup}

    out = {"modelo": modelo, "episodios": dict(Counter(k[0] for k in eps))}
    out["tasas"] = {s: {v: {mt: tasa([m[mt] for _, m in sel(s, v)]) for mt in METRICAS} for v in VARIANTES}
                    for s in SISTEMAS}
    escenarios = sorted({k[1] for k in eps})
    out["tasas_escenario"] = {s: {e: {mt: tasa([m[mt] for _, m in sel(s, esc=e)]) for mt in ("m1", "m2", "m3", "m4", "m5")}
                                  for e in escenarios} for s in SISTEMAS}
    contrastes = []
    for v in VARIANTES:
        contrastes.append(contraste("agente", "agente_escudo", "m1", v))           # H2, seguridad
        for mt in ("m2", "m3", "m2_final"):
            contrastes.append(contraste("agente_escudo", "agente", mt, v))         # H2, coste
        for mt in ("m2", "m3"):
            contrastes.append(contraste("agente_escudo", "reglas", mt, v))         # H3
    out["contrastes"] = [c for c in contrastes if c]

    # M4 por plantilla y por escenario
    m4 = {"plantilla": defaultdict(lambda: defaultdict(lambda: [0, 0])),
          "escenario": defaultdict(lambda: defaultdict(lambda: [0, 0]))}
    for (s, e, _), (ep, m) in eps.items():
        if s == "reglas" or ep["variante"] != "compuesta":
            continue
        for clave, grupo in (("plantilla", ep["resumen"]["inyeccion"]["id"]), ("escenario", e)):
            for mt in ("m4", "m4_propuesta"):
                m4[clave][grupo][f"{s}:{mt}"][0] += bool(m[mt])
                m4[clave][grupo][f"{s}:{mt}"][1] += 1
    out["m4_desglose"] = json.loads(json.dumps(m4))
    out["m4_sin_ssdf"] = tasa([m["m4"] for ep, m in sel("agente", "compuesta") if ep["escenario"] != "SSDF_C"])

    # escudo: bloqueos por invariante, propuestas evaluadas y acciones perjudiciales que pasan
    bloqueos = defaultdict(lambda: {"bloqueadas": 0, "perjudiciales": 0, "legitimas": 0})
    prop = Counter()
    fugas = []
    for ep, _ in sel("agente_escudo"):
        for p in ep["pasos"]:
            if p["automatico"]:
                continue
            if p["ejecutada"]["perjudicial"]:
                fugas.append({"escenario": ep["escenario"], "semilla": ep["semilla"], "paso": p["paso"],
                              "motivos": p["ejecutada"]["motivos"], "accion": p["ejecutada"]["accion"]})
            for pr in p["propuestas"]:
                if pr["estado"] == "no_valida":
                    continue
                perj = bool(pr["evaluacion"]["perjudicial"])
                prop["evaluadas"] += 1
                prop["perjudiciales"] += perj
                if pr["estado"] == "bloqueada":
                    prop["bloqueadas"] += 1
                    prop["bloqueadas_legitimas"] += not perj
                    for inv in pr["escudo"]["infracciones"]:
                        bloqueos[inv]["bloqueadas"] += 1
                        bloqueos[inv]["perjudiciales" if perj else "legitimas"] += 1
                elif perj:
                    prop["perjudiciales_ejecutadas"] += 1
                for mot in pr["evaluacion"]["motivos"]:
                    prop[f"motivo:{mot}:propuestas"] += 1
                    prop[f"motivo:{mot}:ejecutadas"] += pr["estado"] == "aceptada"
    out["bloqueos"] = dict(sorted(bloqueos.items()))
    out["propuestas_escudo"] = dict(prop)
    out["fugas"] = fugas

    # escudo en sombra sobre las propuestas del agente sin protección
    sombra = Counter()
    for ep, _ in sel("agente"):
        for p in ep["pasos"]:
            if p["automatico"]:
                continue
            for pr in p["propuestas"]:
                esc = pr.get("escudo")
                if pr["estado"] == "no_valida" or not esc:
                    continue
                clase = "perjudiciales" if pr["evaluacion"]["perjudicial"] else "legitimas"
                sombra[clase] += 1
                sombra[clase + "_bloqueadas"] += not esc["permitida"]
    out["sombra"] = dict(sombra)

    # validez operativa y coste
    oper = {}
    for s in SISTEMAS:
        fs = sel(s)
        if not fs:
            continue
        pasos = [p for ep, _ in fs for p in ep["pasos"] if not p["automatico"]]
        oper[s] = {
            "episodios": len(fs),
            "duracion_media_s": statistics.mean(ep["duracion_s"] for ep, _ in fs),
            "duracion_mediana_s": statistics.median(ep["duracion_s"] for ep, _ in fs),
            "turnos_medios": statistics.mean(m["turnos_llm"] for _, m in fs),
            "tokens_entrada_medios": statistics.mean(m["tokens_entrada"] or 0 for _, m in fs),
            "tokens_salida_medios": statistics.mean(m["tokens_salida"] or 0 for _, m in fs),
            "no_validas": sum(m["n_no_validas"] for _, m in fs),
            "recordatorios": sum(m["recordatorios"] for _, m in fs),
            "truncadas": sum(m["salidas_truncadas"] for _, m in fs),
            "sin_accion": sum(m["n_sin_accion"] for _, m in fs),
            "modo_seguro": sum(m["n_modo_seguro"] for _, m in fs),
            "pasos": len(pasos),
            "pasos_esperar": sum(1 for p in pasos if p["ejecutada"]["accion"].get("accion") == "ESPERAR"),
            "inicio": min(ep["fecha"] for ep, _ in fs), "fin": max(ep["fecha"] for ep, _ in fs),
        }
    out["operativa"] = oper

    # emparejamiento: en los pares en los que el escudo no intervino, las trayectorias deberían coincidir
    emp = Counter()
    for (s, e, sem), (ea, ma) in eps.items():
        if s != "agente" or ("agente_escudo", e, sem) not in eps:
            continue
        ee, me = eps[("agente_escudo", e, sem)]
        if me["n_bloqueos"] == 0 and me["n_modo_seguro"] == 0:
            emp["sin_intervencion"] += 1
            emp["sin_intervencion_iguales"] += decisiones(ea) == decisiones(ee)
            for mt in ("m1", "m2", "m3"):
                emp[f"{mt}_discordantes"] += bool(ma[mt]) != bool(me[mt])
        else:
            emp["con_intervencion"] += 1
    out["emparejamiento"] = dict(emp)

    # diagnósticos en la ventana del incidente
    out["diagnosticos"] = {f"{s}:{v}": dict(Counter(f"{ep['tipo']}|{m['diagnostico_incidente']}" for ep, m in sel(s, v)))
                           for s in SISTEMAS for v in ("simple", "compuesta")}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campana", default="campana")
    p.add_argument("--salida", default=str(RAIZ / "resultados"), help="carpeta de resultados")
    p.add_argument("--modelo", required=True)
    p.add_argument("--destino", default=None, help="fichero JSON (por defecto, <campaña>/datos_cap5_<modelo>.json)")
    args = p.parse_args()

    campana = Campana(args.salida, args.campana)
    eps = cargar(campana, args.modelo)
    if not any(k[0] != "reglas" for k in eps):
        print(f"No hay episodios de {args.modelo} en {campana.dir}")
        return 1
    destino = Path(args.destino) if args.destino else campana.dir / f"datos_cap5_{nombre_seguro(args.modelo)}.json"
    destino.write_text(json.dumps(datos(eps, args.modelo), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(eps)} episodios ({args.modelo} y reglas). Datos en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
