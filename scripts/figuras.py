#!/usr/bin/env python3
"""Tablas y figuras de resultados de una campaña (capítulo 5 de la memoria).

  python3 scripts/figuras.py --campana campana
  python3 scripts/figuras.py --campana piloto --sin-figuras

Lee las trayectorias guardadas, recalcula las métricas con la definición vigente y
escribe en resultados/<campaña>/figuras/:

  seguridad.png       M1 y M4 por sistema y modelo, con intervalo de Agresti-Coull (H1 y H2)
  intentos.png        acciones perjudiciales propuestas y ejecutadas con y sin escudo (H2)
  utilidad.png        M2 y M3 por sistema, modelo y variante (coste de H2 y H3)
  diagnosticos.png    diagnóstico registrado frente al incidente real
  episodio_*.png      un episodio con y sin escudo (opción --episodio)
  tasas.csv           las cifras de las figuras (también M2 con la definición inicial)
  bloqueos.csv        bloqueos del escudo por invariante, y lo que bloquearía en sombra
  coste.csv           duración, turnos, tokens y validez operativa por sistema y modelo
  diagnosticos.csv    las cifras de la matriz de diagnósticos

Las tablas solo usan la biblioteca estándar; las figuras necesitan matplotlib
(pip install matplotlib). Los contrastes de McNemar están en scripts/resumen.py.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.evaluacion import Campana, agresti_coull, calcular_metricas  # noqa: E402
from spectrum_shield.incidentes import ETIQUETAS_DIAGNOSTICO, TIPOS_INCIDENTE  # noqa: E402

NOMBRE = {"reglas": "Reglas", "agente": "Sin protección", "agente_escudo": "Con escudo"}
# paleta validada para daltonismo (todas las parejas) sobre fondo blanco
COLOR = {"reglas": "#E07B00", "agente": "#2F5496", "agente_escudo": "#2E7D32"}
TINTA, TINTA_2, REJILLA = "#1A1A1A", "#595959", "#E3E3E3"
ETIQUETA_CORTA = {"INTERFERENCIA": "Interferencia", "EMULACION_PRIMARIO": "Emulación",
                  "FALSIFICACION_SENSADO": "Falsificación", "FALLO_CANAL": "Fallo",
                  "PRIMARIO_LEGITIMO": "Primario legítimo", "NINGUNO": "Ninguno", None: "Sin diagnóstico"}
METRICAS = ("m1", "m1_propuesta", "m2", "m2_final", "m3", "m4", "m4_propuesta", "m5")


# ---------------------------------------------------------------------- datos
def cargar(campana: Campana):
    """Una fila por episodio, con las métricas recalculadas y los bloqueos por invariante."""
    cfg = campana.config()
    filas = []
    for ep in campana.episodios():
        m = calcular_metricas(ep, cfg)
        bloqueos, bloqueos_perj, sombra = Counter(), Counter(), Counter()
        for p in ep["pasos"]:
            for pr in p["propuestas"]:
                esc = pr.get("escudo")
                if not esc or esc.get("permitida", True):
                    continue
                for inv in esc["infracciones"]:
                    if pr["estado"] == "bloqueada":
                        bloqueos[inv] += 1
                        bloqueos_perj[inv] += bool(pr["evaluacion"]["perjudicial"])
                    else:
                        sombra[inv] += 1
        filas.append({
            "sistema": ep["sistema"], "modelo": ep.get("modelo"), "escenario": ep["escenario"],
            "tipo": ep["tipo"], "variante": ep["variante"], "semilla": ep["semilla"],
            "duracion_s": ep["duracion_s"], **m,
            # M2 con la definición inicial (último diagnóstico del episodio), para la sensibilidad
            "m2_final": m["diagnostico_final"] == ep["tipo"],
            "bloqueos": bloqueos, "bloqueos_perjudiciales": bloqueos_perj, "sombra": sombra,
        })
    return filas


def configuraciones(filas):
    """Orden fijo de filas en las figuras: reglas y, por modelo, sin protección y con escudo."""
    modelos = sorted({f["modelo"] for f in filas if f["modelo"]}, key=lambda m: (not m.startswith("qwen"), m))
    orden = [("reglas", None)] if any(f["sistema"] == "reglas" for f in filas) else []
    for mod in modelos:
        for s in ("agente", "agente_escudo"):
            if any(f["sistema"] == s and f["modelo"] == mod for f in filas):
                orden.append((s, mod))
    return orden


def seleccionar(filas, sistema, modelo, variante="todas"):
    return [f for f in filas if f["sistema"] == sistema and f["modelo"] == modelo
            and (variante == "todas" or f["variante"] == variante)]


def tasa(fs, metrica):
    vals = [f[metrica] for f in fs if f[metrica] is not None]
    k, n = sum(bool(v) for v in vals), len(vals)
    if not n:
        return None
    p, inf, sup = agresti_coull(k, n)
    return {"k": k, "n": n, "p": p, "inf": inf, "sup": sup}


# ---------------------------------------------------------------------- tablas
def escribir(ruta: Path, filas, campos):
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def tablas(filas, salida: Path):
    orden = configuraciones(filas)
    tasas = []
    for s, mod in orden:
        for var in ("simple", "compuesta", "todas"):
            fs = seleccionar(filas, s, mod, var)
            for met in METRICAS:
                t = tasa(fs, met)
                if t:
                    tasas.append({"sistema": s, "modelo": mod or "", "variante": var, "metrica": met, "k": t["k"],
                                  "n": t["n"], "proporcion": round(t["p"], 4), "ic95_inf": round(t["inf"], 4),
                                  "ic95_sup": round(t["sup"], 4)})
    escribir(salida / "tasas.csv", tasas, ["sistema", "modelo", "variante", "metrica", "k", "n", "proporcion",
                                           "ic95_inf", "ic95_sup"])

    bloqueos = []
    for s, mod in orden:
        fs = seleccionar(filas, s, mod)
        b, bp, sb = Counter(), Counter(), Counter()
        for f in fs:
            b.update(f["bloqueos"])
            bp.update(f["bloqueos_perjudiciales"])
            sb.update(f["sombra"])
        for inv in ("I1", "I2", "I3", "I4", "I5"):
            if b[inv] or sb[inv]:
                bloqueos.append({"sistema": s, "modelo": mod or "", "invariante": inv, "bloqueadas": b[inv],
                                 "bloqueadas_perjudiciales": bp[inv], "bloqueadas_no_perjudiciales": b[inv] - bp[inv],
                                 "en_sombra": sb[inv]})
    escribir(salida / "bloqueos.csv", bloqueos, ["sistema", "modelo", "invariante", "bloqueadas",
                                                 "bloqueadas_perjudiciales", "bloqueadas_no_perjudiciales",
                                                 "en_sombra"])

    coste = []
    for s, mod in orden:
        fs = seleccionar(filas, s, mod)
        pasos = len(fs) * 4
        coste.append({
            "sistema": s, "modelo": mod or "", "episodios": len(fs),
            "duracion_media_s": round(statistics.mean(f["duracion_s"] for f in fs), 1),
            "duracion_mediana_s": round(statistics.median(f["duracion_s"] for f in fs), 1),
            "turnos_medios": round(statistics.mean(f["turnos_llm"] for f in fs), 2),
            "tokens_entrada_medios": round(statistics.mean(f["tokens_entrada"] or 0 for f in fs)),
            "tokens_salida_medios": round(statistics.mean(f["tokens_salida"] or 0 for f in fs)),
            "propuestas_no_validas": sum(f["n_no_validas"] for f in fs),
            "recordatorios": sum(f["recordatorios"] for f in fs),
            "respuestas_truncadas": sum(f["salidas_truncadas"] for f in fs),
            "pasos_sin_accion": sum(f["n_sin_accion"] for f in fs),
            "pasos_en_modo_seguro": sum(f["n_modo_seguro"] for f in fs),
            "pasos_de_decision": pasos,
        })
    escribir(salida / "coste.csv", coste, list(coste[0].keys()) if coste else ["sistema"])

    diags = []
    for s, mod in orden:
        cuenta = Counter((f["tipo"], f["diagnostico_incidente"]) for f in seleccionar(filas, s, mod))
        for (tipo, etiqueta), n in sorted(cuenta.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            diags.append({"sistema": s, "modelo": mod or "", "incidente": tipo, "diagnostico": etiqueta or "",
                          "episodios": n})
    escribir(salida / "diagnosticos.csv", diags, ["sistema", "modelo", "incidente", "diagnostico", "episodios"])
    return tasas, bloqueos, coste


# ---------------------------------------------------------------------- figuras
def preparar_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    familia = next((f for f in ("Carlito", "Calibri", "Arial", "DejaVu Sans") if f in disponibles), "sans-serif")
    plt.rcParams.update({"font.family": familia, "font.size": 8, "axes.edgecolor": TINTA_2,
                         "axes.labelcolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
                         "savefig.dpi": 300, "savefig.facecolor": "white"})
    return plt


def filas_grafico(orden):
    """Posiciones verticales con un hueco entre grupos y la etiqueta de grupo de cada modelo."""
    ys, etiquetas, grupos, y = [], [], [], 0.0
    previo = "inicio"
    for s, mod in orden:
        if mod != previo and previo != "inicio":
            y += 0.6
        if mod and mod != previo:
            grupos.append((y - 0.55, mod))
            y += 0.45
        ys.append(y)
        etiquetas.append(NOMBRE[s])
        previo = mod
        y += 1.0
    return ys, etiquetas, grupos, y


def panel_puntos(ax, filas, orden, metrica, variante, titulo, mostrar_etiquetas):
    ys, etiquetas, grupos, alto = filas_grafico(orden)
    for y, (s, mod) in zip(ys, orden):
        t = tasa(seleccionar(filas, s, mod, variante), metrica)
        if not t:
            ax.text(1, y, "no aplica", fontsize=7, color=TINTA_2, va="center", ha="left")
            continue
        ax.plot([100 * t["inf"], 100 * t["sup"]], [y, y], color=COLOR[s], lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(100 * t["p"], y, "o", ms=6, color=COLOR[s], mec="white", mew=1.2, zorder=3)
    ax.set_xlim(-2, 102)
    ax.set_ylim(alto - 0.4, -1.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100 %"])
    ax.grid(axis="x", color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks(ys)
    ax.set_yticklabels(etiquetas if mostrar_etiquetas else [""] * len(ys))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#9E9E9E")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_title(titulo, fontsize=8.4, fontweight="bold", loc="left", color=TINTA)
    if mostrar_etiquetas:
        for y, mod in grupos:
            ax.text(-0.02, y, mod, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.6,
                    fontweight="bold", color=TINTA)


def figura_puntos(plt, filas, paneles, ruta, alto_extra=0.0):
    orden = configuraciones(filas)
    _, _, _, alto = filas_grafico(orden)
    filas_paneles = max(r for r, _, _, _, _ in paneles) + 1
    cols = max(c for _, c, _, _, _ in paneles) + 1
    fig, axes = plt.subplots(filas_paneles, cols, figsize=(6.3, 0.32 * alto * filas_paneles + 0.75 + alto_extra),
                             squeeze=False)
    for r, c, metrica, variante, titulo in paneles:
        panel_puntos(axes[r][c], filas, orden, metrica, variante, titulo, mostrar_etiquetas=(c == 0))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.12, wspace=0.12, hspace=0.55)
    fig.savefig(ruta)
    plt.close(fig)


def figura_intentos(plt, filas, ruta):
    """Con escudo: propuestas frente a ejecuciones perjudiciales; referencia: sin protección."""
    modelos = [m for s, m in configuraciones(filas) if s == "agente_escudo"]
    if not modelos:
        return False
    casos = [("m1", "todas", "Acción perjudicial (M1), todos los escenarios"),
             ("m4", "compuesta", "Éxito de la inyección (M4), escenarios compuestos")]
    fig, axes = plt.subplots(1, len(casos), figsize=(6.3, 0.5 * len(modelos) + 1.25), squeeze=False)
    for c, (met, var, titulo) in enumerate(casos):
        ax = axes[0][c]
        for i, mod in enumerate(modelos):
            sin = tasa(seleccionar(filas, "agente", mod, var), met)
            prop = tasa(seleccionar(filas, "agente_escudo", mod, var), met + "_propuesta")
            ejec = tasa(seleccionar(filas, "agente_escudo", mod, var), met)
            if prop and ejec:
                ax.plot([100 * ejec["p"], 100 * prop["p"]], [i, i], color=COLOR["agente_escudo"], lw=1.2, zorder=1)
                ax.plot(100 * prop["p"], i, "o", ms=6.5, mfc="white", mec=COLOR["agente_escudo"], mew=1.5, zorder=3,
                        label="Con escudo: propuesta" if i == 0 else None)
                ax.plot(100 * ejec["p"], i, "o", ms=6.5, color=COLOR["agente_escudo"], mec="white", mew=1.2,
                        zorder=4, label="Con escudo: ejecutada" if i == 0 else None)
            if sin:
                ax.plot(100 * sin["p"], i - 0.22, "D", ms=5, color=COLOR["agente"], mec="white", mew=1.0, zorder=4,
                        label="Sin protección: ejecutada" if i == 0 else None)
        ax.set_yticks(range(len(modelos)))
        ax.set_yticklabels(modelos if c == 0 else [""] * len(modelos))
        ax.set_ylim(len(modelos) - 0.5, -0.7)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100 %"])
        ax.grid(axis="x", color=REJILLA, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color("#9E9E9E")
        ax.spines["bottom"].set_linewidth(0.8)
        ax.set_title(titulo, fontsize=8.4, fontweight="bold", loc="left", color=TINTA)
    manejadores, textos = axes[0][0].get_legend_handles_labels()
    fig.legend(manejadores, textos, loc="lower center", ncol=3, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.3, wspace=0.12)
    fig.savefig(ruta)
    plt.close(fig)
    return True


def figura_diagnosticos(plt, filas, ruta):
    orden = [(s, m) for s, m in configuraciones(filas) if s in ("reglas", "agente")]
    if not orden:
        return False
    columnas = list(ETIQUETAS_DIAGNOSTICO) + [None]
    fig, axes = plt.subplots(1, len(orden), figsize=(6.3, 2.35), squeeze=False)
    for c, (s, mod) in enumerate(orden):
        ax = axes[0][c]
        fs = seleccionar(filas, s, mod)
        matriz = [[sum(1 for f in fs if f["tipo"] == t and f["diagnostico_incidente"] == e) for e in columnas]
                  for t in TIPOS_INCIDENTE]
        for i, fila in enumerate(matriz):
            total = sum(fila) or 1
            for j, n in enumerate(fila):
                frac = n / total
                # una sola tonalidad: más oscuro cuanto mayor es la fracción de la fila
                color = (1 - 0.78 * frac, 1 - 0.62 * frac, 1 - 0.38 * frac) if n else (0.97, 0.97, 0.97)
                ax.add_patch(plt.Rectangle((j, i), 0.94, 0.94, color=color, lw=0))
                if n:
                    ax.text(j + 0.47, i + 0.5, str(n), ha="center", va="center", fontsize=6.8,
                            color="white" if frac > 0.55 else TINTA)
        ax.set_xlim(0, len(columnas))
        ax.set_ylim(len(TIPOS_INCIDENTE), 0)
        ax.set_xticks([j + 0.47 for j in range(len(columnas))])
        ax.set_xticklabels([ETIQUETA_CORTA[e] for e in columnas], rotation=55, ha="right", fontsize=6.6)
        ax.set_yticks([i + 0.47 for i in range(len(TIPOS_INCIDENTE))])
        ax.set_yticklabels([ETIQUETA_CORTA[t] for t in TIPOS_INCIDENTE] if c == 0 else [""] * 4, fontsize=7)
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title("Reglas" if s == "reglas" else f"Sin protección ({mod})", fontsize=8, fontweight="bold",
                     loc="left", color=TINTA)
    fig.text(0.01, 0.98, "Incidente real (filas) y diagnóstico registrado en la ventana del incidente (columnas)",
             fontsize=7.4, color=TINTA_2, va="top")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.83, bottom=0.33, wspace=0.08)
    fig.savefig(ruta)
    plt.close(fig)
    return True


def figura_episodio(plt, campana: Campana, escenario: str, semilla: int, modelo: str, ruta) -> bool:
    """Un mismo episodio con y sin escudo: estado real de cada canal, acciones ejecutadas y bloqueadas."""
    from matplotlib.patches import Rectangle
    from spectrum_shield.evaluacion import id_sistema
    import json
    eps = {}
    for s in ("agente", "agente_escudo"):
        r = campana.ruta(id_sistema(s, modelo), escenario, semilla)
        if not r.exists():
            return False
        eps[s] = json.loads(r.read_text(encoding="utf-8"))
    relleno = {"primario": ("#F6DEDB", "primario"), "emulador": ("#FBEBD0", "emulador"),
               "interferencia": ("#FBEBD0", "interferencia"), "degradado": ("#EFEFEF", "degradado")}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.15))
    for ax, (s, ep) in zip(axes, eps.items()):
        pasos = [p for p in ep["pasos"] if not p["automatico"]]
        n_canales = len(pasos[0]["estado_real"])
        for j, p in enumerate(pasos):
            for c in p["estado_real"]:
                estado = next((k for k in ("primario", "emulador", "interferencia", "degradado") if c[k]), None)
                y = c["canal"]
                fc, txt = relleno.get(estado, ("white", ""))
                ax.add_patch(Rectangle((j, y), 0.96, 0.92, fc=fc, ec=REJILLA, lw=0.6, zorder=1))
                if y == n_canales - 1:
                    ax.add_patch(Rectangle((j, y), 0.96, 0.92, fc="none", ec="#BDBDBD", lw=0, hatch="////", zorder=1))
                if txt:
                    ax.text(j + 0.48, y + 0.74, txt, ha="center", va="center", fontsize=5.6, color=TINTA_2, zorder=2)
            for pr in p["propuestas"]:
                if pr["estado"] == "bloqueada":
                    c = pr["accion"].get("canal")
                    ax.plot(j + 0.48, c + 0.36, marker="x", ms=7, mew=1.8, color="#A93226", zorder=4)
                    ax.text(j + 0.48, c + 0.75, "bloqueada (" + "+".join(pr["escudo"]["infracciones"]) + ")",
                            fontsize=5.4, color=TINTA, va="center", ha="center", zorder=4,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
            a = p["ejecutada"]["accion"]
            if a.get("accion") == "TRANSMITIR":
                perj = p["ejecutada"]["perjudicial"]
                ax.plot(j + 0.48, a["canal"] + 0.4, "o", ms=7, color=TINTA, mec="#A93226" if perj else "white",
                        mew=1.8 if perj else 1.0, zorder=5)
                if perj:
                    ax.text(j + 0.48, a["canal"] + 0.75, "perjudicial", fontsize=5.4, color=TINTA, ha="center",
                            va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        ax.set_xlim(-0.05, len(pasos))
        ax.set_ylim(n_canales, -0.1)
        ax.set_xticks([j + 0.48 for j in range(len(pasos))])
        ax.set_xticklabels([f"paso {p['paso']}" for p in pasos], fontsize=7)
        ax.set_yticks([c + 0.46 for c in range(n_canales)])
        ax.set_yticklabels([f"canal {c}" if c < n_canales - 1 else f"canal {c}\n(protegido)" for c in range(n_canales)]
                           if s == "agente" else [""] * n_canales, fontsize=6.8)
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(f"{NOMBRE[s]} ({modelo})", fontsize=8, fontweight="bold", loc="left", color=TINTA)
        diags = ", ".join(f"paso {d['paso']}: {d['etiqueta']}" for d in ep["diagnosticos"] if d["paso"] >= 2)
        ax.text(0, n_canales + 0.75, "Diagnósticos: " + (diags or "ninguno"), fontsize=6.4, color=TINTA_2,
                va="top", ha="left")
    fig.text(0.01, 0.985, f"{escenario}, semilla {semilla}. Círculo: transmisión ejecutada; aspa: propuesta bloqueada "
                          f"por el escudo.", fontsize=6.8, color=TINTA_2, va="top")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.2, wspace=0.06)
    fig.savefig(ruta)
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campana", default="campana")
    p.add_argument("--salida", default=str(RAIZ / "resultados"), help="carpeta de resultados")
    p.add_argument("--destino", default=None, help="dónde escribir (por defecto, <campaña>/figuras)")
    p.add_argument("--sin-figuras", action="store_true", help="solo las tablas (no necesita matplotlib)")
    p.add_argument("--episodio", action="append", default=[], metavar="ESCENARIO:SEMILLA:MODELO",
                   help="figura de un episodio con y sin escudo, por ejemplo JAM_S:0:qwen3.8:27b (se puede repetir)")
    args = p.parse_args()

    campana = Campana(args.salida, args.campana)
    filas = cargar(campana)
    if not filas:
        print(f"No hay episodios en {campana.dir}")
        return 1
    destino = Path(args.destino) if args.destino else campana.dir / "figuras"
    destino.mkdir(parents=True, exist_ok=True)
    tasas, bloqueos, coste = tablas(filas, destino)
    print(f"{len(filas)} episodios de la campaña {args.campana}. Tablas en {destino}")

    if not args.sin_figuras:
        try:
            plt = preparar_matplotlib()
        except ImportError:
            print("Para las figuras hace falta matplotlib (pip install matplotlib).")
            return 0
        figura_puntos(plt, filas, [(0, 0, "m1", "simple", "M1 en escenarios simples"),
                                   (0, 1, "m1", "compuesta", "M1 en escenarios compuestos"),
                                   (0, 2, "m4", "compuesta", "M4 en escenarios compuestos")],
                      destino / "seguridad.png")
        figura_intentos(plt, filas, destino / "intentos.png")
        figura_puntos(plt, filas, [(0, 0, "m2", "simple", "M2 en escenarios simples"),
                                   (0, 1, "m2", "compuesta", "M2 en escenarios compuestos"),
                                   (1, 0, "m3", "simple", "M3 en escenarios simples"),
                                   (1, 1, "m3", "compuesta", "M3 en escenarios compuestos")],
                      destino / "utilidad.png")
        figura_diagnosticos(plt, filas, destino / "diagnosticos.png")
        print("Figuras: seguridad.png, intentos.png, utilidad.png y diagnosticos.png")
        for e in args.episodio:
            esc, sem, mod = e.split(":", 2)
            nombre = f"episodio_{esc}_s{int(sem):03d}_{mod.replace(':', '-')}.png"
            if figura_episodio(plt, campana, esc, int(sem), mod, destino / nombre):
                print(f"Figura del episodio: {nombre}")
            else:
                print(f"No están los dos episodios de {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
