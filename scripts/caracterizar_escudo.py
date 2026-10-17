#!/usr/bin/env python3
"""Caracterización del escudo sin modelo de lenguaje.

Mide cómo decide el escudo con independencia del agente: en cada paso de decisión se
reúne toda la evidencia que se puede pedir (sensado de los 6 canales y estimación de
origen de todos ellos) y se evalúa una transmisión con los valores por defecto en cada
canal autorizado. Cada decisión se compara con la verdad del terreno:

  - perjudicial que pasa: el escudo permite transmitir sobre un primario activo;
  - bloqueo correcto: el escudo bloquea una transmisión perjudicial;
  - bloqueo sin coste: bloquea un canal que no es perjudicial, pero tampoco apto
    (interferencia, emulador o degradación), donde transmitir no tendría éxito;
  - bloqueo con coste: bloquea un canal apto, es decir, una transmisión legítima;
  - permitida: deja transmitir en un canal no perjudicial (apto o no).

Además compara el escudo con cuatro variantes que quitan una pieza cada una, para
medir qué aporta cada pieza: sin revocar la confianza en el centro de fusión, solo la
medida propia, solo el centro de fusión y sin verificación de origen. Las variantes
solo existen en este script; el sistema evaluado usa siempre el escudo completo.

Las semillas por defecto (1000 a 1999) no coinciden con las del piloto ni con las de
la campaña. La banda protegida (canal 5) no se evalúa: la bloquea siempre I3. Se usa
la réplica en Python de la política, que decide igual que la política Rego (lo
comprueban las pruebas sobre 3.840 casos), porque es mucho más rápida.

  python3 scripts/caracterizar_escudo.py
  python3 scripts/caracterizar_escudo.py --semillas 1000-1199 --salida /tmp/escudo
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield import __version__  # noqa: E402
from spectrum_shield.entorno import Config  # noqa: E402
from spectrum_shield.entorno.simulador import TRANSMITIR, Simulador  # noqa: E402
from spectrum_shield.incidentes import CATALOGO  # noqa: E402
from spectrum_shield.sistemas.escudo import Escudo, decidir_python, fuente_cooperativa_comprometida  # noqa: E402
from spectrum_shield.sistemas.herramientas import Herramientas  # noqa: E402

VARIANTES = {
    "completo": dict(),
    "sin_revocar_confianza": dict(confiar_siempre=True),
    "solo_medida_propia": dict(usar_fusion=False),
    "solo_centro_fusion": dict(usar_propia=False),
    "sin_verificacion_origen": dict(usar_origen=False),
}
CATEGORIAS = ("perjudicial_pasa", "bloqueo_correcto", "bloqueo_sin_coste", "bloqueo_con_coste",
              "permitida_apta", "permitida_no_apta")
INCIDENTE = {"JAM": "Interferencia", "PUE": "Emulación de primario", "SSDF": "Falsificación del sensado",
             "FALLO": "Fallo del canal"}


def leer_semillas(texto: str):
    salida = []
    for parte in texto.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            salida.extend(range(int(a), int(b) + 1))
        elif parte.strip():
            salida.append(int(parte))
    return salida


def decidir_i1(entrada, confiar_siempre=False, usar_propia=True, usar_fusion=True, usar_origen=True) -> bool:
    """Decisión de I1 de la política, con piezas desactivables. Con los valores por defecto
    coincide con decidir_python (se comprueba en cada caso)."""
    a, paso, ev = entrada["accion"], entrada["paso"], entrada["evidencias"]
    canal = a["canal"]
    fiable = True if confiar_siempre else not fuente_cooperativa_comprometida(ev)
    sensados = [e for e in ev if e["tipo"] == "sensado" and e["canal"] == canal and e["paso"] == paso]
    propia = usar_propia and any(e["deteccion_local"] == "ocupado" or e["firma_primario"] is True for e in sensados)
    cooperativa = usar_fusion and fiable and any(e["decision_fusion"] == "ocupado" for e in sensados)
    descarta = usar_origen and any(
        e["tipo"] == "origen" and e["canal"] == canal and e["paso"] == paso and e["emisor_detectado"] is True
        and e["distancia_m"] is not None and e["distancia_m"] > entrada["politica"]["umbral_distancia_m"]
        for e in ev)
    return (propia or cooperativa) and not descarta


def categoria(perjudicial: bool, apta: bool, bloqueada: bool) -> str:
    if perjudicial:
        return "bloqueo_correcto" if bloqueada else "perjudicial_pasa"
    if bloqueada:
        return "bloqueo_con_coste" if apta else "bloqueo_sin_coste"
    return "permitida_apta" if apta else "permitida_no_apta"


def estado(c) -> str:
    if c.primario:
        return "primario"
    if c.emulador:
        return "emulador"
    if c.interferencia:
        return "interferencia"
    if c.degradado:
        return "degradado"
    return "libre"


def caracterizar(cfg: Config, semillas):
    """Devuelve una fila por acción candidata evaluada."""
    filas = []
    escudo = Escudo(cfg, motor="python")
    for esc in CATALOGO:
        for s in semillas:
            sim = Simulador(cfg)
            sim.reiniciar(esc, s)
            h = Herramientas(sim)
            escudo.iniciar_episodio()
            resumen = sim.resumen_episodio()
            c0, refugio = resumen["canal_inicial"], resumen["canal_refugio"]
            while not sim.terminado:
                t = sim.paso
                escudo.observar(sim.observacion())
                if t < cfg.paso_incidente:
                    sim.ejecutar({"accion": TRANSMITIR, "canal": sim.canal_activo})
                    continue
                for c in range(cfg.n_canales):
                    h.consultar_sensado(c)
                    h.estimar_origen(c)
                for c in cfg.canales_autorizados:
                    accion = sim.normalizar({"accion": TRANSMITIR, "canal": c})
                    ev = sim.evaluar_accion(accion)
                    decision = escudo.evaluar(accion, t, h.llamadas)
                    entrada = escudo.entrada(accion, t)
                    real = sim.guion[t][c]
                    fila = {"escenario": esc.codigo, "incidente": esc.codigo.split("_")[0], "semilla": s,
                            "paso": t, "canal": c,
                            "rol": "inicial" if c == c0 else ("refugio" if c == refugio else "otro"),
                            "estado_real": estado(real), "perjudicial": ev["perjudicial"], "apta": real.apto(),
                            "cooperativa_fiable": decision["cooperativa_fiable"],
                            "infracciones": "+".join(decision["infracciones"])}
                    # por qué pasa una transmisión perjudicial (solo para el escudo completo)
                    causa = ""
                    if ev["perjudicial"] and decision["permitida"]:
                        propia = decidir_i1(entrada, usar_fusion=False, usar_origen=False)
                        coop = decidir_i1(entrada, usar_propia=False, usar_origen=False)
                        causa = ("origen_descarta" if (propia or coop) else
                                 "sin_deteccion_propia_ni_fusion_fiable")
                    fila["causa_paso"] = causa
                    for nombre, opciones in VARIANTES.items():
                        if nombre == "completo":
                            bloqueada = not decision["permitida"]
                            # la réplica de I1 de este script tiene que decidir igual que la política
                            assert decidir_i1(entrada) == ("I1" in decision["infracciones"]), fila
                            assert decision == {**decidir_python(entrada), "motivo": decision["motivo"],
                                                "motor": "python"}
                        else:
                            otras = [i for i in decision["infracciones"] if i != "I1"]
                            bloqueada = bool(otras) or decidir_i1(entrada, **opciones)
                        fila[nombre] = categoria(ev["perjudicial"], real.apto(), bloqueada)
                    filas.append(fila)
                # se avanza transmitiendo en el refugio, como haría un sistema que se recupera
                sim.ejecutar({"accion": TRANSMITIR, "canal": refugio})
    return filas


def tasas(filas, variante: str):
    """Tasas de error por incidente y en total."""
    salida = {}
    grupos = defaultdict(list)
    for f in filas:
        grupos[f["incidente"]].append(f)
        grupos["Total"].append(f)
    for g, fs in grupos.items():
        n = Counter(f[variante] for f in fs)
        perj = n["perjudicial_pasa"] + n["bloqueo_correcto"]
        aptas = n["bloqueo_con_coste"] + n["permitida_apta"]
        no_aptas = n["bloqueo_sin_coste"] + n["permitida_no_apta"]
        salida[g] = {
            "acciones": len(fs), "perjudiciales": perj, "aptas": aptas, "no_aptas_no_perjudiciales": no_aptas,
            "perjudiciales_que_pasan": n["perjudicial_pasa"],
            "tasa_perjudiciales_que_pasan": n["perjudicial_pasa"] / perj if perj else None,
            "aptas_bloqueadas": n["bloqueo_con_coste"],
            "tasa_aptas_bloqueadas": n["bloqueo_con_coste"] / aptas if aptas else None,
            "tasa_no_aptas_bloqueadas": n["bloqueo_sin_coste"] / no_aptas if no_aptas else None,
        }
    return salida


def pct(x, decimales=1) -> str:
    return "-" if x is None else f"{100 * x:.{decimales}f} %"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--semillas", default="1000-1999")
    p.add_argument("--config", default=str(RAIZ / "config" / "experimento.json"))
    p.add_argument("--salida", default=str(RAIZ / "resultados" / "caracterizacion_escudo"))
    p.add_argument("--detalle", action="store_true", help="guarda también una fila por acción evaluada")
    args = p.parse_args()

    ruta_cfg = Path(args.config)
    cfg = Config.desde_json(ruta_cfg) if ruta_cfg.exists() else Config()
    semillas = leer_semillas(args.semillas)
    filas = caracterizar(cfg, semillas)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    resumen = []
    for variante in VARIANTES:
        for grupo, t in tasas(filas, variante).items():
            resumen.append({"variante": variante, "incidente": grupo, **t})
    with (salida / "resumen.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    if args.detalle:
        with (salida / "acciones.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)

    # bloqueos por estado real del canal destino (todas las variantes)
    por_estado = []
    for variante in VARIANTES:
        grupos = defaultdict(Counter)
        for f in filas:
            g = grupos[(f["incidente"] if f["estado_real"] == "primario" and f["incidente"] == "SSDF"
                        and f["rol"] == "inicial" else "todos", f["estado_real"])]
            g[f[variante].startswith("bloqueo")] += 1
        for (ambito, est), n in sorted(grupos.items()):
            total = n[True] + n[False]
            por_estado.append({"variante": variante, "ambito": ambito, "estado_real": est, "acciones": total,
                               "bloqueadas": n[True], "tasa_bloqueo": round(n[True] / total, 5)})
    with (salida / "por_estado.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(por_estado[0].keys()))
        w.writeheader()
        w.writerows(por_estado)
    motivos = defaultdict(Counter)
    for f in filas:
        motivos[f["estado_real"]][f["infracciones"] or "ninguna"] += 1

    # revocaciones de la confianza en el centro de fusión, por incidente y paso
    revocada = defaultdict(lambda: [0, 0])
    vistos = set()
    for f in filas:
        clave = (f["escenario"], f["semilla"], f["paso"])
        if clave in vistos:
            continue
        vistos.add(clave)
        r = revocada[(f["incidente"], f["paso"])]
        r[0] += (not f["cooperativa_fiable"])
        r[1] += 1
    with (salida / "confianza.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["incidente", "paso", "episodios", "confianza_revocada", "tasa"])
        for (g, paso), (k, n) in sorted(revocada.items()):
            w.writerow([g, paso, n, k, round(k / n, 4)])

    print(f"SPECTRUM-SHIELD {__version__} | caracterización del escudo | {len(semillas)} semillas × "
          f"{len(CATALOGO)} escenarios | {len(filas)} acciones evaluadas\n")
    t = tasas(filas, "completo")
    print(f"{'Incidente':28s} {'perjudiciales que pasan':>26s} {'aptas bloqueadas':>22s} {'no aptas bloqueadas':>22s}")
    for g in ("JAM", "PUE", "SSDF", "FALLO", "Total"):
        x = t[g]
        print(f"{INCIDENTE.get(g, g):28s} {x['perjudiciales_que_pasan']:>6d}/{x['perjudiciales']:<6d}"
              f"{pct(x['tasa_perjudiciales_que_pasan'], 2):>12s}  {x['aptas_bloqueadas']:>6d}/{x['aptas']:<6d}"
              f"{pct(x['tasa_aptas_bloqueadas']):>8s}  {pct(x['tasa_no_aptas_bloqueadas']):>20s}")
    print("\nEscudo completo, por estado real del canal destino:")
    for est in ("primario", "emulador", "interferencia", "degradado", "libre"):
        filas_e = [x for x in por_estado if x["variante"] == "completo" and x["estado_real"] == est]
        n = sum(x["acciones"] for x in filas_e)
        b = sum(x["bloqueadas"] for x in filas_e)
        print(f"  {est:14s} {b:>6d}/{n:<6d} {pct(b / n if n else None, 2):>9s}   motivos: "
              + ", ".join(f"{k} {v}" for k, v in motivos[est].most_common(3)))
    causas = Counter(f["causa_paso"] for f in filas if f["causa_paso"])
    print("  causas de las perjudiciales que pasan: " + ", ".join(f"{k} {v}" for k, v in causas.most_common()))
    print("\nVariantes (total y falsificación del sensado):")
    print(f"{'Variante':26s} {'pasan (total)':>14s} {'pasan (SSDF)':>14s} {'aptas bloq. (total)':>20s} "
          f"{'aptas bloq. (SSDF)':>19s}")
    for variante in VARIANTES:
        t = tasas(filas, variante)
        print(f"{variante:26s} {pct(t['Total']['tasa_perjudiciales_que_pasan'], 2):>14s} "
              f"{pct(t['SSDF']['tasa_perjudiciales_que_pasan'], 2):>14s} "
              f"{pct(t['Total']['tasa_aptas_bloqueadas']):>20s} {pct(t['SSDF']['tasa_aptas_bloqueadas']):>19s}")
    print(f"\nTablas en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
