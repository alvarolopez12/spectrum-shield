#!/usr/bin/env python3
"""Cifras y figura de los experimentos exploratorios para la memoria (fuera de los criterios fijados).

Lee las campañas exploratorias y escribe resultados/datos_exploratorios.json y la figura
resultados/figuras_exploratorios/tras_bloqueo.png:
  - ataques adaptativos (adaptativa_insistente y adaptativa_reescrita) frente a la campaña, con granite;
  - qué hace el agente con escudo en los pasos en que el escudo le bloquea alguna propuesta;
  - agente endurecido (aviso de seguridad en las instrucciones) frente al agente sin protección y con escudo;
  - repeticiones: proporción de episodios con el mismo valor en las tres ejecuciones;
  - escudo sin agente: cota analítica, variantes de I1, origen comprometido y contrafactual (resultados/escudo_extra);
  - respaldo Simplex, si ya se ha ejecutado (resumen de scripts/respaldo_simplex.py --resumen);
  - disponibilidad: ataques «esperar» y «quedarse» y vigilancia de la vivacidad (resúmenes de scripts/disponibilidad.py);
  - garantías del escudo según la evidencia (resultados/garantias/datos_garantias.json de scripts/garantias.py);
  - experimento 6: ataques a la evidencia y su complemento contra el sistema final (resúmenes de
    scripts/ataque_evidencia.py y scripts/ataque_evidencia_vivo.py, más el recuento por oportunidad);
  - dependencia del desbloqueo por origen: transmisiones aceptadas en las campañas con escudo activo que solo el
    desbloqueo dejaba pasar (se regeneran las medidas con el simulador);
  - experimento 7: incidentes no previstos (resumen de scripts/incidentes_nuevos.py), si ya se ha ejecutado.

Figuras en resultados/figuras_exploratorios/: tras_bloqueo.png, disponibilidad.png, garantias.png y evidencia.png.

  python3 scripts/exploratorios.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.evaluacion import (Campana, agresti_coull, bootstrap_diferencia, calcular_metricas,  # noqa: E402
                                        id_sistema, tabla_pares)

MODELO = "granite4.1:8b"
COMPUESTOS = {"JAM_C", "PUE_C", "SSDF_C", "FALLO_C"}
TEXTOS = {"original": "campana", "insistente": "adaptativa_insistente", "reescrito": "adaptativa_reescrita"}
METRICAS = ("m1", "m2", "m3", "m4", "m4_propuesta", "m5")
CATEGORIAS = (("espera", "Espera aceptada"), ("modo_seguro", "Modo seguro o sin acción"),
              ("transmite_sin_exito", "Transmite sin éxito"), ("transmite_con_exito", "Transmite con éxito"))
COLORES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # validados (scripts de dataviz): orden fijo, sin repetir
TINTA, TINTA_2, REJILLA = "#1A1A1A", "#595959", "#E3E3E3"
GRIS_REF = "#7a7975"


def cargar(raiz, campana: str, sistema: str, escenarios=None, semillas=None):
    camp = Campana(raiz, campana)
    cfg = camp.config()
    out = {}
    for ruta in sorted((camp.dir_episodios / id_sistema(sistema, MODELO)).glob("*/s*.json")):
        ep = json.loads(ruta.read_text(encoding="utf-8"))
        if escenarios and ep["escenario"] not in escenarios:
            continue
        if semillas and ep["semilla"] not in semillas:
            continue
        ep["metricas"] = calcular_metricas(ep, cfg)
        out[(ep["escenario"], ep["semilla"])] = ep
    return out


def tasa(eps, met, claves=None):
    v = [bool(eps[k]["metricas"][met]) for k in (claves or eps) if eps[k]["metricas"][met] is not None]
    k, n = sum(v), len(v)
    p, lo, hi = agresti_coull(k, n) if n else (None, None, None)
    return {"k": k, "n": n, "p": p, "inf": lo, "sup": hi}


def contraste(a, b, met):
    """a frente a b sobre los episodios comunes: McNemar exacto y diferencia a − b con bootstrap estratificado."""
    ks = sorted(k for k in set(a) & set(b) if a[k]["metricas"][met] is not None and b[k]["metricas"][met] is not None)
    da = {k: bool(a[k]["metricas"][met]) for k in ks}
    db = {k: bool(b[k]["metricas"][met]) for k in ks}
    t = tabla_pares(da, db)
    estr = defaultdict(list)
    for k in ks:
        estr[k[0]].append((da[k], db[k]))
    d, lo, hi = bootstrap_diferencia(estr, b=10000)
    return {"n": len(ks), "solo_a": t["solo_a"], "solo_b": t["solo_b"], "p": t["p_mcnemar"], "dif": d, "inf": lo, "sup": hi}


def decision(ep):
    return [p for p in ep["pasos"] if not p["automatico"]]


def resultado_paso(p) -> str:
    if p["origen_accion"] in ("modo_seguro", "sin_accion"):
        return "modo_seguro"
    if p["ejecutada"]["accion"].get("accion") != "TRANSMITIR":
        return "espera"
    return "transmite_con_exito" if p["ejecutada"]["exito"] else "transmite_sin_exito"


def tras_bloqueo(eps):
    todos, ultimo = Counter(), Counter()
    for ep in eps.values():
        dec = decision(ep)
        for i, p in enumerate(dec):
            if not any(pr["estado"] == "bloqueada" for pr in p["propuestas"]):
                continue
            r = resultado_paso(p)
            todos[r] += 1
            todos["pasos"] += 1
            if i == len(dec) - 1:
                ultimo[r] += 1
                ultimo["pasos"] += 1
    return {**{c: todos[c] for c, _ in CATEGORIAS}, "pasos": todos["pasos"],
            "ultimo_paso": {**{c: ultimo[c] for c, _ in CATEGORIAS}, "pasos": ultimo["pasos"]}}


def recuentos(eps):
    c = Counter()
    for ep in eps.values():
        c["turnos"] += ep["metricas"]["turnos_llm"]
        for p in decision(ep):
            c["modo_seguro"] += p["origen_accion"] == "modo_seguro"
            c["intentos_perjudiciales"] += sum(1 for pr in p["propuestas"]
                                               if pr["estado"] != "no_valida" and pr["evaluacion"]["perjudicial"])
    return dict(c)


# ---------------------------------------------------------------- bloques
def adaptativos(raiz):
    out = {"campanas": TEXTOS, "tasas": {}, "contrastes": {}, "recuentos": {}, "tras_bloqueo": {}}
    datos = {t: {s: cargar(raiz, c, s, COMPUESTOS) for s in ("agente", "agente_escudo")} for t, c in TEXTOS.items()}
    for t, por_sis in datos.items():
        out["tasas"][t] = {s: {m: tasa(eps, m) for m in METRICAS} for s, eps in por_sis.items()}
        out["recuentos"][t] = {s: recuentos(eps) for s, eps in por_sis.items()}
        out["tras_bloqueo"][t] = tras_bloqueo(por_sis["agente_escudo"])
        if t != "original":
            o = datos["original"]
            out["contrastes"][t] = {
                "m4_agente": contraste(por_sis["agente"], o["agente"], "m4"),
                "m1_escudo": contraste(por_sis["agente_escudo"], o["agente_escudo"], "m1"),
                "m3_escudo": contraste(por_sis["agente_escudo"], o["agente_escudo"], "m3"),
                "m4_propuesta_escudo": contraste(por_sis["agente_escudo"], o["agente_escudo"], "m4_propuesta"),
            }
    rondas = Counter(str(ep["resumen"]["inyeccion"].get("ronda")) for ep in datos["reescrito"]["agente"].values())
    out["rondas_reescrito"] = dict(sorted(rondas.items()))
    out["n"] = len(datos["original"]["agente"])
    return out


def endurecido(raiz):
    e = cargar(raiz, "endurecido", "agente_endurecido", COMPUESTOS)
    a = cargar(raiz, "campana", "agente", COMPUESTOS)
    s = cargar(raiz, "campana", "agente_escudo", COMPUESTOS)
    mets = ("m1", "m2", "m3", "m4")
    return {"n": len(e), "tasas": {"agente_endurecido": {m: tasa(e, m) for m in mets},
                                   "agente": {m: tasa(a, m) for m in mets},
                                   "agente_escudo": {m: tasa(s, m) for m in mets}},
            "frente_agente": {m: contraste(e, a, m) for m in mets},
            "frente_escudo": {m: contraste(e, s, m) for m in mets}}


def repeticiones(raiz):
    semillas = set(range(20, 30))
    out = {"semillas": "20-29", "sistemas": {}}
    for sis in ("agente", "agente_escudo"):
        runs = [cargar(raiz, c, sis, None, semillas) for c in ("campana", "repeticion_2", "repeticion_3")]
        comunes = sorted(set.intersection(*(set(r) for r in runs)))
        res = {"episodios": len(comunes)}
        for m in METRICAS:
            ks = [k for k in comunes if runs[0][k]["metricas"][m] is not None]
            if not ks:
                continue
            iguales = sum(1 for k in ks if len({bool(r[k]["metricas"][m]) for r in runs}) == 1)
            res[m] = {"n": len(ks), "iguales": iguales, "tasas": [sum(bool(r[k]["metricas"][m]) for k in ks) for r in runs]}
        out["sistemas"][sis] = res
    return out


def escudo_extra(raiz):
    d = raiz / "escudo_extra"
    if not (d / "cota_analitica.json").exists():
        return None
    cota = {k: v for k, v in json.loads((d / "cota_analitica.json").read_text(encoding="utf-8")).items() if k != "config"}
    variantes = defaultdict(dict)
    with (d / "variantes.csv").open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    for analisis in ("memoria", "origen_comprometido"):
        por_var = defaultdict(dict)
        for f in filas:
            if f["analisis"] == analisis:
                por_var[f["variante"]][f["incidente"]] = f
        for var, por_inc in por_var.items():
            resto = [por_inc[i] for i in ("FALLO", "JAM", "PUE") if i in por_inc]
            variantes[analisis][var] = {
                "pasan_total": float(por_inc["Total"]["tasa_pasan"]),
                "pasan_ssdf": float(por_inc["SSDF"]["tasa_pasan"]),
                "pasan_resto": sum(int(r["pasan"]) for r in resto) / max(1, sum(int(r["perjudiciales"]) for r in resto)),
                "aptas_bloqueadas": float(por_inc["Total"]["tasa_aptas_bloqueadas"]),
                "perjudiciales": int(por_inc["Total"]["perjudiciales"]), "aptas": int(por_inc["Total"]["aptas"])}
    with (d / "contrafactual.csv").open(encoding="utf-8") as fh:
        contra = [{k: (int(v) if v not in ("",) and v.isdigit() else v) for k, v in f.items()} for f in csv.DictReader(fh)]
    return {"cota": cota, "variantes": variantes, "contrafactual": contra}


def simplex(raiz):
    out = {}
    for texto in ("insistente", "original"):
        r = raiz / f"simplex_{texto}" / f"resumen_{id_sistema('agente_escudo_simplex', MODELO)}.json"
        if r.exists():
            out[texto] = json.loads(r.read_text(encoding="utf-8"))
    return out or None


CAMPANAS_DISPONIBILIDAD = {"original": "vivacidad_original", "esperar": "disponibilidad_esperar",
                           "quedarse": "disponibilidad_quedarse"}


def disponibilidad(raiz):
    out = {}
    for texto, camp in CAMPANAS_DISPONIBILIDAD.items():
        r = raiz / camp / "resumen_disponibilidad.json"
        if r.exists():
            out[texto] = json.loads(r.read_text(encoding="utf-8"))
            out[texto]["pasos_respaldo"] = pasos_respaldo(raiz, camp)
    return out or None


def pasos_respaldo(raiz, campana: str):
    """Pasos de decisión en los que propone el respaldo (Simplex de bloqueo y vigilancia de la vivacidad)."""
    out = {}
    for sistema in ("agente_escudo_simplex", "agente_escudo_vivo"):
        c = Counter()
        for ruta in sorted((raiz / campana / "episodios" / id_sistema(sistema, MODELO)).glob("*/s*.json")):
            for p in decision(json.loads(ruta.read_text(encoding="utf-8"))):
                c["pasos"] += 1
                c["respaldo"] += int(any(pr.get("respaldo") for pr in p["propuestas"]))
        if c["pasos"]:
            out[sistema] = dict(c)
    return out


def garantias(raiz):
    """Lo que usa la memoria de resultados/garantias/datos_garantias.json (scripts/garantias.py --campanas)."""
    r = raiz / "garantias" / "datos_garantias.json"
    if not r.exists():
        return None
    d = json.loads(r.read_text(encoding="utf-8"))
    mc = d["monte_carlo_campana"]
    claves = ("total", "fiable", "revocada", "honestos", "ssdf", "honestos_fiable", "honestos_revocada", "ssdf_fiable",
              "ssdf_revocada")
    return {"version": d["version"], "config_huella": d["config_huella"], "semillas_mc": d["semillas_mc"],
            "no_interferencia": d["no_interferencia"], "cota_campana": d["cota_campana"],
            "monte_carlo": {e: {v: {g: mc[e][v][g] for g in claves if g in mc[e][v]} for v in ("actual", "sin_origen",
                                                                                              "memoria_1", "sin_revocar")}
                            for e in ("minima", "completa")},
            "envolvente": d["envolvente"], "agregacion": {k: v for k, v in d["agregacion"].items() if k != "curvas"},
            "agregacion_m10": d["agregacion"]["curvas"][9], "capacidades": d["capacidades"], "frontera": d["frontera"],
            "campanas": d.get("campanas")}


# ---------------------------------------------------------------- experimento 6 y dependencia del desbloqueo
CAMPANAS_EVIDENCIA = ("evidencia_origen", "evidencia_bizantinos", "evidencia_enlace")
CAMPANAS_EVIDENCIA_VIVO = ("evidencia_origen_vivo", "evidencia_sin_origen_original")
CAMPANAS_ESCUDO = ("campana", "repeticion_2", "repeticion_3", "adaptativa_insistente", "adaptativa_reescrita",
                   "simplex_insistente", "simplex_original", "disponibilidad_esperar", "disponibilidad_quedarse",
                   "vivacidad_original")


def oportunidades_objetivo(ep, canal: int):
    """Pasos con alguna propuesta del agente (no del respaldo) de transmitir sobre el primario del canal objetivo, una vez
    por paso (dentro del paso la medida es la misma): si pidió el origen en el paso, si la fusión era fiable y si alguna
    pasó el escudo. Es el recuento por oportunidad de scripts/ataque_evidencia_vivo.py."""
    out = []
    for p in decision(ep):
        origen = any(ll["herramienta"] == "estimar_origen" and ll["argumentos"].get("canal") == canal
                     for ll in p.get("llamadas", []))
        its = []
        for pr in p["propuestas"]:
            ev, esc = pr.get("evaluacion"), pr.get("escudo")
            if pr.get("respaldo") or not ev or not esc or pr["accion"].get("canal") != canal:
                continue
            if "transmision_sobre_primario" in ev["motivos"]:
                its.append((bool(esc["permitida"]), bool(esc.get("cooperativa_fiable"))))
        if its:
            out.append({"origen": origen, "fiable": its[0][1], "pasa": any(x[0] for x in its)})
    return out


def por_oportunidad(raiz, campana: str, sistema: str):
    c = Counter()
    for ruta in sorted((raiz / campana / "episodios" / id_sistema(sistema, MODELO)).glob("*/s*.json")):
        ep = json.loads(ruta.read_text(encoding="utf-8"))
        canal = (ep.get("evidencia_atacada") or {}).get("canal")
        if canal is None:
            continue
        for o in oportunidades_objetivo(ep, canal):
            g = ("pide_origen" if o["origen"] else "no_pide_origen") + ("_fiable" if o["fiable"] else "_revocada")
            for k in (g, "total"):
                c[k + "_n"] += 1
                c[k + "_pasan"] += int(o["pasa"])
    return dict(c)


def evidencia(raiz):
    """Resúmenes del experimento 6 y su complemento, con el recuento por oportunidad de A1 y A2."""
    out = {}
    for camp in CAMPANAS_EVIDENCIA:
        r = raiz / camp / "resumen_evidencia.json"
        if not r.exists():
            continue
        d = json.loads(r.read_text(encoding="utf-8"))
        if camp != "evidencia_enlace":
            d["por_oportunidad"] = {s: por_oportunidad(raiz, camp, s) for s in d["sistemas"]}
        out[camp] = d
    for camp in CAMPANAS_EVIDENCIA_VIVO:
        r = raiz / camp / "resumen_evidencia_vivo.json"
        if r.exists():
            out[camp] = json.loads(r.read_text(encoding="utf-8"))
    if "evidencia_sin_origen_original" in out:
        out["evidencia_sin_origen_original"]["identicas"] = trayectorias_identicas(
            raiz, "evidencia_sin_origen_original", "agente_escudo_vivo_sin_origen", "vivacidad_original",
            "agente_escudo_vivo", {"PUE_S", "PUE_C"})
    if "evidencia_origen_vivo" in out:
        out["evidencia_origen_vivo"]["reglas_ssdf"] = reglas_ssdf(raiz)
    return out or None


def _trayectoria(ep):
    return [(p["paso"], json.dumps(p["ejecutada"]["accion"], sort_keys=True), [pr["estado"] for pr in p["propuestas"]],
             [ll["herramienta"] for ll in p.get("llamadas", [])]) for p in ep["pasos"]]


def trayectorias_identicas(raiz, camp_a, sis_a, camp_b, sis_b, escenarios):
    """Episodios con las mismas acciones, propuestas y llamadas en cada paso en dos campañas emparejadas por semilla."""
    a = {(r.parent.name, r.stem): r for r in (raiz / camp_a / "episodios" / id_sistema(sis_a, MODELO)).glob("*/s*.json")
         if r.parent.name in escenarios}
    b = {(r.parent.name, r.stem): r for r in (raiz / camp_b / "episodios" / id_sistema(sis_b, MODELO)).glob("*/s*.json")
         if r.parent.name in escenarios}
    comunes = sorted(set(a) & set(b))
    iguales = sum(_trayectoria(json.loads(a[k].read_text(encoding="utf-8")))
                  == _trayectoria(json.loads(b[k].read_text(encoding="utf-8"))) for k in comunes)
    return {"n": len(comunes), "iguales": iguales}


def reglas_ssdf(raiz):
    """Sistema final con el escudo de la campaña bajo el ataque al origen, en SSDF_C: si el canal objetivo es el activo
    y primer diagnóstico de las reglas cuando las consulta el monitor o el respaldo."""
    c = Counter()
    for r in (raiz / "evidencia_origen_vivo" / "episodios" / id_sistema("agente_escudo_vivo", MODELO) / "SSDF_C").glob("s*.json"):
        ep = json.loads(r.read_text(encoding="utf-8"))
        c["episodios"] += 1
        c["objetivo_es_el_activo"] += int(ep["evidencia_atacada"]["canal"] == ep["resumen"]["canal_inicial"])
        for p in decision(ep):
            a = p.get("agente") or {}
            d = (a.get("vivacidad") or {}).get("diagnostico_reglas") or (a.get("respaldo") or {}).get("diagnostico_reglas")
            if d:
                c["primer_diagnostico_" + d] += 1
                break
    return dict(c)


def dependencia_desbloqueo(raiz):
    """Transmisiones aceptadas por el escudo en las campañas con escudo activo y cuántas dependían del desbloqueo por
    origen: con energía o firma en la medida propia del paso, I1 solo las deja pasar con el desbloqueo. Si no depende
    ninguna, quitar el desbloqueo (que solo añade bloqueos) dejaría todas las trayectorias igual. Tarda un par de minutos:
    el resultado se guarda en resultados/dependencia_desbloqueo.json y se reutiliza mientras no cambien las campañas."""
    cache = raiz / "dependencia_desbloqueo.json"
    firma = {c: len(list((raiz / c / "episodios").glob("agente_escudo*/*/s*.json"))) for c in CAMPANAS_ESCUDO
             if (raiz / c / "episodios").exists()}
    if cache.exists():
        guardado = json.loads(cache.read_text(encoding="utf-8"))
        if guardado.get("firma") == firma:
            return guardado["resultado"]
    from spectrum_shield.entorno.simulador import Simulador
    from spectrum_shield.incidentes import seleccionar
    esc = {e.codigo: e for e in seleccionar("todos")}
    out, medidas = {}, {}
    for camp in CAMPANAS_ESCUDO:
        d = raiz / camp
        if not (d / "episodios").exists():
            continue
        cfg = Campana(raiz, camp).config()
        c = Counter()
        for sisdir in sorted((d / "episodios").glob("agente_escudo*")):
            for ruta in sisdir.glob("*/s*.json"):
                ep = json.loads(ruta.read_text(encoding="utf-8"))
                if ep.get("modo_escudo") != "activo" or ep["escenario"] not in esc:
                    continue
                clave = (cfg.huella(), ep["escenario"], ep["semilla"])
                if clave not in medidas:
                    sim = Simulador(cfg)
                    sim.reiniciar(esc[ep["escenario"]], ep["semilla"])
                    medidas[clave] = sim.medidas
                med = medidas[clave]
                c["episodios"] += 1
                for p in ep["pasos"]:
                    for pr in p["propuestas"]:
                        a, e = pr.get("accion") or {}, pr.get("escudo") or {}
                        ch = a.get("canal")
                        if a.get("accion") != "TRANSMITIR" or not e.get("permitida") or not isinstance(ch, int) \
                                or not 0 <= ch < cfg.n_canales:
                            continue
                        m = med[min(p["paso"], cfg.n_pasos - 1)][ch]
                        c["aceptadas"] += 1
                        c["dependen"] += int(m["energia_dbm"] > cfg.umbral_energia_dbm or bool(m["firma"]))
        if c:
            out[camp] = dict(c)
    if out:
        out["total"] = {k: sum(v.get(k, 0) for v in out.values()) for k in ("episodios", "aceptadas", "dependen")}
        cache.write_text(json.dumps({"firma": firma, "resultado": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out or None


def incidentes(raiz):
    r = raiz / "incidentes_nuevos" / "resumen_incidentes.json"
    return json.loads(r.read_text(encoding="utf-8")) if r.exists() else None


# ---------------------------------------------------------------- figura
def figura_tras_bloqueo(datos, ruta: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    familia = next((f for f in ("Carlito", "Calibri", "Arial", "DejaVu Sans") if f in disponibles), "sans-serif")
    plt.rcParams.update({"font.family": familia, "font.size": 8, "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "savefig.dpi": 300, "savefig.facecolor": "white"})
    nombres = {"original": "Plantilla original", "insistente": "Ataque insistente", "reescrito": "Ataque reescrito"}
    textos = [t for t in ("original", "insistente", "reescrito") if t in datos]
    fig, ax = plt.subplots(figsize=(6.3, 0.62 * len(textos) + 0.95))
    for y, t in enumerate(textos):
        d = datos[t]
        izq = 0.0
        for (clave, _), color in zip(CATEGORIAS, COLORES):
            v = 100 * d[clave] / d["pasos"] if d["pasos"] else 0
            ax.barh(y, v, left=izq, height=0.56, color=color, edgecolor="white", linewidth=1.5, zorder=2)
            if v >= 7:
                ax.text(izq + v / 2, y, f"{v:.0f} %", ha="center", va="center", fontsize=7.4, color=TINTA, zorder=3)
            izq += v
    ax.set_yticks(range(len(textos)))
    ax.set_yticklabels([f"{nombres[t]}\n{datos[t]['pasos']} pasos con bloqueo" for t in textos], fontsize=7.6, color=TINTA)
    ax.set_ylim(len(textos) - 0.45, -0.75)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100 %"], fontsize=7)
    ax.grid(axis="x", color=REJILLA, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#9E9E9E")
    ax.spines["bottom"].set_linewidth(0.8)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORES]
    ax.legend(manijas, [n for _, n in CATEGORIAS], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
              fontsize=7.4, handlelength=1.1, handleheight=0.9, columnspacing=1.4, borderaxespad=0.3)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.82, bottom=0.14)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    familia = next((f for f in ("Carlito", "Calibri", "Arial", "DejaVu Sans") if f in disponibles), "sans-serif")
    plt.rcParams.update({"font.family": familia, "font.size": 8, "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "savefig.dpi": 300, "savefig.facecolor": "white"})
    return plt


# sistemas de la figura de disponibilidad: paleta validada (todas las parejas, modo claro) y las reglas en gris, como
# referencia; la identidad la llevan además las etiquetas de cada fila
SISTEMAS_DISP = (("agente_escudo", "Con escudo", "#008300"), ("agente_escudo_simplex", "Simplex de bloqueo", "#6250d6"),
                 ("agente_escudo_vivo", "Vigilancia de la vivacidad", "#1baf7a"), ("reglas", "Reglas", "#7a7975"))


def filas_disponibilidad(sim, disp):
    """(texto, n, [(sistema, k, n)]) para la figura y la tabla de disponibilidad, en el orden de la memoria."""
    grupos = []
    if sim and "original" in sim:
        t = dict(sim["original"]["tasas"]["m3"])
        if disp and "original" in disp:
            t.update({s: v for s, v in disp["original"]["tasas"]["m3"].items() if s in ("agente_escudo_vivo", "reglas")})
        grupos.append(("original", sim["original"]["n"], t))
    if sim and "insistente" in sim:
        grupos.append(("insistente", sim["insistente"]["n"], sim["insistente"]["tasas"]["m3"]))
    for texto in ("esperar", "quedarse"):
        if disp and texto in disp:
            grupos.append((texto, disp[texto]["n"], disp[texto]["tasas"]["m3"]))
    out = []
    for texto, n, t in grupos:
        out.append((texto, n, [(s, t[s][0], t[s][1]) for s, _, _ in SISTEMAS_DISP if s in t]))
    return out


NOMBRES_TEXTO = {"original": "Textos de la campaña (ocho escenarios)", "insistente": "Ataque insistente",
                 "esperar": "Ataque «esperar»", "quedarse": "Ataque «quedarse»"}


def figura_disponibilidad(sim, disp, ruta: Path):
    plt = _estilo()
    color = {s: c for s, _, c in SISTEMAS_DISP}
    nombre = {s: n for s, n, _ in SISTEMAS_DISP}
    grupos = filas_disponibilidad(sim, disp)
    filas = []          # (y, texto de la fila, sistema, k, n) o cabecera
    y = 0.0
    for texto, n, sistemas in grupos:
        filas.append((y, f"{NOMBRES_TEXTO[texto]}, {n} episodios", None, None, None))
        y += 1.0
        for s, k, m in sistemas:
            filas.append((y, nombre[s], s, k, m))
            y += 0.82
        y += 0.38
    fig, ax = plt.subplots(figsize=(6.3, 0.235 * y + 0.75))
    for yy, etiqueta, s, k, m in filas:
        if s is None:
            ax.text(-0.56, yy, etiqueta, ha="left", va="center", fontsize=7.8, color=TINTA, fontweight="bold",
                    transform=ax.get_yaxis_transform())
            continue
        p, lo, hi = agresti_coull(k, m)
        ax.plot([100 * lo, 100 * hi], [yy, yy], color=color[s], lw=1.3, solid_capstyle="round", zorder=2)
        ax.plot(100 * p, yy, "o", ms=5.6, color=color[s], mec="white", mew=1.1, zorder=3)
        ax.text(100 * hi + 1.6, yy, f"{100 * k / m:.0f} %".replace(".", ","), ha="left", va="center", fontsize=7.2,
                color=TINTA_2, zorder=3)
        ax.text(-0.02, yy, etiqueta, ha="right", va="center", fontsize=7.6, color=TINTA, transform=ax.get_yaxis_transform())
    ax.set_ylim(y - 0.2, -0.7)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100 %"], fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("Recuperación del servicio (M3), con intervalo de Agresti-Coull al 95 %", fontsize=7.6, color=TINTA)
    ax.grid(axis="x", color=REJILLA, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#9E9E9E")
    ax.spines["bottom"].set_linewidth(0.8)
    manijas = [plt.Line2D([0], [0], marker="o", lw=0, ms=5.6, color=c, mec="white", mew=1.1) for _, _, c in SISTEMAS_DISP]
    ax.legend(manijas, [n for _, n, _ in SISTEMAS_DISP], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4,
              frameon=False, fontsize=7.4, handletextpad=0.3, columnspacing=1.4, borderaxespad=0.3)
    fig.subplots_adjust(left=0.36, right=0.98, top=1 - 0.42 / (0.235 * y + 0.75), bottom=0.5 / (0.235 * y + 0.75))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


# nodos bizantinos de la figura de garantías: rampa ordinal de un solo tono (validada con --ordinal)
RAMPA_K = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def _potencia(v, _):
    if v <= 0:
        return ""
    e = int(round(math.log10(v)))
    return "1" if e == 0 else "10" + str(e).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def figura_garantias(env, ruta: Path):
    plt = _estilo()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    xs = [100 * f["pd_energia"] for f in env["curvas"]]
    for k, c in zip((0, 1, 2, 3), RAMPA_K):
        ys = [f["mezcla"][str(k)] for f in env["curvas"]]
        ax.plot(xs, ys, color=c, lw=1.5, solid_capstyle="round", zorder=2)
        for p in env["mc"]:
            if p["k"] == k and p["tasa"]:
                lo = max(p["ic95"][0], 1.2e-7)
                ax.plot([100 * p["pd_energia"]] * 2, [lo, p["ic95"][1]], color=c, lw=0.8, alpha=0.55, zorder=3)
                ax.plot(100 * p["pd_energia"], p["tasa"], "o", ms=5.2, color=c, mec="white", mew=1.0, zorder=4)
        nombre = {0: "sin nodos bizantinos", 1: "1 nodo bizantino de 5", 2: "2 de 5", 3: "3 de 5 (mayoría)"}[k]
        ax.text(xs[-1] + 0.6, ys[-1], nombre, ha="left", va="center", fontsize=7.2, color=TINTA)
    pdc = 100 * env["pd_campana"]
    ax.axvline(pdc, color=GRIS_REF, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(pdc - 0.4, 0.45, "campaña", ha="right", va="center", fontsize=7, color=TINTA_2)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1)
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(-7, 1)]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_potencia))
    ax.set_xlim(70, 107)
    ax.set_xticks([70, 75, 80, 85, 90, 95, 100])
    ax.set_xticklabels(["70", "75", "80", "85", "90", "95", "100 %"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Probabilidad de detección de energía de la medida propia", fontsize=7.6)
    ax.set_ylabel("Probabilidad de que pase una transmisión\nsobre un primario, por oportunidad", fontsize=7.6)
    ax.grid(axis="y", color=REJILLA, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color("#9E9E9E")
        ax.spines[lado].set_linewidth(0.8)
    manijas = [plt.Line2D([0], [0], color=c, lw=1.5) for c in RAMPA_K]
    manijas.append(plt.Line2D([0], [0], marker="o", lw=0, ms=5.2, color=TINTA_2, mec="white", mew=1.0))
    ax.legend(manijas, ["0", "1", "2", "3 nodos bizantinos de 5 (cota cerrada)", "Monte Carlo con la política real"],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, frameon=False, fontsize=7.2, handlelength=1.4,
              handletextpad=0.35, columnspacing=0.9, borderaxespad=0.3)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.15)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


# sistemas de la figura del experimento 6: el tono distingue el escudo solo (verde, como «Con escudo» en la figura de
# disponibilidad) del sistema final (aqua, como «Vigilancia de la vivacidad»); el relleno, con o sin desbloqueo por origen
SISTEMAS_EVI = (("agente_escudo (evidencia_origen)", "Escudo de la campaña", "#008300", True),
                ("agente_escudo_sin_origen (evidencia_origen)", "Escudo sin desbloqueo por origen", "#008300", False),
                ("agente_escudo_vivo", "Sistema final con el escudo de la campaña", "#1baf7a", True),
                ("agente_escudo_vivo_sin_origen", "Sistema final sin desbloqueo (configuración final)", "#1baf7a", False))


def figura_evidencia(ev, ruta: Path):
    """Ataque al origen: acción perjudicial (M1) y recuperación (M3) de los cuatro sistemas, en dos paneles con la
    misma escala."""
    plt = _estilo()
    t = ev["evidencia_origen_vivo"]["tasas"]
    fig, ejes = plt.subplots(1, 2, figsize=(6.3, 2.05), sharey=True)
    for ax, met, titulo in ((ejes[0], "m1", "Episodios con acción perjudicial (M1, %)"),
                            (ejes[1], "m3", "Episodios con recuperación (M3, %)")):
        for y, (clave, nombre, color, lleno) in enumerate(SISTEMAS_EVI):
            k, m = t[met][clave]
            p, lo, hi = agresti_coull(k, m)
            ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=1.3, solid_capstyle="round", zorder=2)
            ax.plot(100 * p, y, "o", ms=5.8, color=color if lleno else "white", mec=color if not lleno else "white",
                    mew=1.4 if not lleno else 1.1, zorder=3)
            texto = f"{100 * k / m:.1f}".replace(".", ",").replace(",0", "") + " %"
            if hi > 0.82:
                ax.text(100 * lo - 2.2, y, texto, ha="right", va="center", fontsize=7.2, color=TINTA_2)
            else:
                ax.text(100 * hi + 2.2, y, texto, ha="left", va="center", fontsize=7.2, color=TINTA_2)
        ax.set_title(titulo, fontsize=7.8, color=TINTA, loc="left", pad=4)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=7)
        ax.grid(axis="x", color=REJILLA, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color("#9E9E9E")
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.tick_params(axis="y", length=0)
    ejes[0].set_ylim(len(SISTEMAS_EVI) - 0.5, -0.6)
    ejes[0].set_yticks(range(len(SISTEMAS_EVI)))
    ejes[0].set_yticklabels([nm for _, nm, _, _ in SISTEMAS_EVI], fontsize=7.4, color=TINTA)
    fig.subplots_adjust(left=0.37, right=0.985, top=0.87, bottom=0.13, wspace=0.12)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", default=str(RAIZ / "resultados"), help="carpeta de resultados")
    p.add_argument("--sin-figura", action="store_true")
    args = p.parse_args()
    raiz = Path(args.salida)
    datos = {"modelo": MODELO, "adaptativos": adaptativos(raiz), "endurecido": endurecido(raiz),
             "repeticiones": repeticiones(raiz), "escudo_extra": escudo_extra(raiz), "simplex": simplex(raiz),
             "disponibilidad": disponibilidad(raiz), "garantias": garantias(raiz), "evidencia": evidencia(raiz),
             "dependencia_desbloqueo": dependencia_desbloqueo(raiz), "incidentes": incidentes(raiz)}
    destino = raiz / "datos_exploratorios.json"
    destino.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Datos en {destino}")
    if not args.sin_figura:
        ruta = raiz / "figuras_exploratorios" / "tras_bloqueo.png"
        figura_tras_bloqueo(datos["adaptativos"]["tras_bloqueo"], ruta)
        print(f"Figura en {ruta}")
        if datos["simplex"] or datos["disponibilidad"]:
            ruta = raiz / "figuras_exploratorios" / "disponibilidad.png"
            figura_disponibilidad(datos["simplex"], datos["disponibilidad"], ruta)
            print(f"Figura en {ruta}")
        if datos["garantias"]:
            ruta = raiz / "figuras_exploratorios" / "garantias.png"
            figura_garantias(datos["garantias"]["envolvente"], ruta)
            print(f"Figura en {ruta}")
        if datos["evidencia"] and "evidencia_origen_vivo" in datos["evidencia"]:
            ruta = raiz / "figuras_exploratorios" / "evidencia.png"
            figura_evidencia(datos["evidencia"], ruta)
            print(f"Figura en {ruta}")
    if datos["dependencia_desbloqueo"]:
        tot = datos["dependencia_desbloqueo"]["total"]
        print(f"Desbloqueo por origen: {tot['dependen']} de {tot['aceptadas']} transmisiones aceptadas dependían de él "
              f"({tot['episodios']} episodios con escudo activo)")
    for texto, n, sistemas in filas_disponibilidad(datos["simplex"], datos["disponibilidad"]):
        print(f"{texto:10s} M3 " + ", ".join(f"{s} {k}/{m}" for s, k, m in sistemas))
    a = datos["adaptativos"]
    for t in TEXTOS:
        tb = a["tras_bloqueo"][t]
        print(f"{t:10s} M4 sin protección {a['tasas'][t]['agente']['m4']['k']}/{a['n']}  M1 con escudo "
              f"{a['tasas'][t]['agente_escudo']['m1']['k']}  M3 con escudo {a['tasas'][t]['agente_escudo']['m3']['k']}  "
              f"pasos con bloqueo {tb['pasos']} (espera {tb['espera']}, modo seguro {tb['modo_seguro']}, "
              f"transmite {tb['transmite_sin_exito']}+{tb['transmite_con_exito']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
