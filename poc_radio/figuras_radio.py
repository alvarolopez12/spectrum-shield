#!/usr/bin/env python3
"""Figuras de la prueba de concepto con radio real para el capítulo 5.

  python3 poc_radio/figuras_radio.py                       # todas las que se puedan hacer con lo que haya
  python3 poc_radio/figuras_radio.py --demo resultados/poc_radio/demo_20261110_183000
  python3 poc_radio/figuras_radio.py --simulado            # con los datos del banco simulado (llevan marca de agua)

Figuras de diseño (no necesitan hardware, salen del código):
  plan_frecuencias.png    banda, seis canales, oscilador local de los Pluto, centro del receptor e imágenes
  formas_onda.png         densidad espectral de las tres señales y ventana de medida de la energía
Figuras de resultados (necesitan las ejecuciones con radio):
  cascada_casos.png       los cuatro casos de la demostración sobre el mismo episodio (de la última demostración
                          o de la que se indique con --demo)
  energia_medida.png      energía medida por el sensado real según lo que había en el canal, frente al modelo del
                          simulador (necesita la minicampaña y su --resumen)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import numpy as np  # noqa: E402

import rf_ejecucion as E  # noqa: E402
import rf_ondas as O  # noqa: E402
import rf_plan as P  # noqa: E402

COLORES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
GRIS = "#666666"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def marca_simulado(fig) -> None:
    fig.text(0.5, 0.5, "SEÑALES SINTÉTICAS (sin hardware)", ha="center", va="center", fontsize=22, color="#d00000",
             alpha=0.25, rotation=20)


# ---------------------------------------------------------------- diseño
def plan_frecuencias(salida: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 2.8))
    lo, hi = (x / 1e6 for x in P.BANDA_HZ)
    ax.axvspan(lo, hi, color=COLORES[0], alpha=0.08, lw=0)
    ax.text((lo + hi) / 2, 1.32, "434,040 a 434,790 MHz: SRD no específicos, 10 mW p.r.a., ciclo de trabajo hasta el "
            "100 % con canales de 25 kHz o menos", ha="center", fontsize=7.5, color=COLORES[0])
    for k, f in enumerate(P.CANALES_HZ):
        c = COLORES[1] if k == P.CANAL_PROTEGIDO else COLORES[0]
        x0 = (f - P.ANCHO_CANAL_HZ / 2) / 1e6
        ax.add_patch(plt.Rectangle((x0, 0.2), P.ANCHO_CANAL_HZ / 1e6, 0.8, color=c, alpha=0.85, lw=0))
        ax.text(f / 1e6, 1.08, f"C{k}", ha="center", fontsize=8, color=c)
        ax.text(f / 1e6, 0.05, f"{f / 1e6:.3f}".replace(".", ","), ha="center", fontsize=6.5, color=GRIS)
        imagen = (2 * P.LO_PLUTO_HZ - f) / 1e6
        ax.plot(imagen, 0.6, marker="v", ms=4, color=COLORES[3])
    ax.axvline(P.LO_PLUTO_HZ / 1e6, color=COLORES[2], lw=1.2, ls="--",
               label="oscilador local de los Pluto (434,425 MHz)")
    ax.axvline(P.CENTRO_RTL_HZ / 1e6, color=GRIS, lw=1.0, ls=":", label="centro del RTL-SDR (434,400 MHz)")
    span = P.FS_RTL / 2 / 1e6
    ax.annotate("", xy=(P.CENTRO_RTL_HZ / 1e6 - span, 1.55), xytext=(P.CENTRO_RTL_HZ / 1e6 + span, 1.55),
                arrowprops=dict(arrowstyle="<->", color=GRIS, lw=0.8))
    ax.text(P.CENTRO_RTL_HZ / 1e6, 1.6, "banda que captura el RTL-SDR (1,024 MHz)", ha="center", fontsize=7, color=GRIS)
    ax.plot([], [], marker="v", ls="", color=COLORES[3], label="imagen de cada canal (cae entre canales)")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=COLORES[0], label="canales 0 a 4 (autorizados)"))
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=COLORES[1], label="canal 5 (banda protegida del experimento)"))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, fontsize=7, frameon=False)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_xlim(P.CENTRO_RTL_HZ / 1e6 - span - 0.02, P.CENTRO_RTL_HZ / 1e6 + span + 0.02)
    ax.set_ylim(-0.1, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("frecuencia (MHz)")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return salida


def formas_onda(salida: Path) -> Path:
    plt = _plt()
    fs, n = P.FS_RTL, P.N_FFT
    t = np.arange(int(fs)) / fs
    ventana = np.blackman(n)
    f = (np.arange(n) - n // 2) * fs / n / 1e3
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    casos = (("primario (y emulador)", O.onda("primario", t, 7)), ("interferencia", O.onda("interferencia", t, 7)),
             ("secundario", O.onda("secundario", t)))
    for ax, (nombre, x) in zip(axs, casos):
        x = x * math.sqrt(O.POTENCIA_ENTORNO / np.mean(np.abs(x) ** 2))      # misma potencia en las tres
        b = x[: (len(x) // n) * n].reshape(-1, n) * ventana
        p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(b, axis=1), axes=1)) ** 2, axis=0)
        p_db = 10 * np.log10(p / p.max())
        ax.axvspan(-P.SEMIANCHO_MEDIDA_HZ / 1e3, P.SEMIANCHO_MEDIDA_HZ / 1e3, color=COLORES[0], alpha=0.08, lw=0)
        ax.plot(f, p_db, color=COLORES[0], lw=0.9)
        if nombre.startswith("primario"):
            for s in (-1, 1):
                ax.annotate("piloto", xy=(s * O.PILOTO_HZ / 1e3, -2), xytext=(s * 8.5, -12), fontsize=7,
                            ha="center", arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.6))
        ax.set_title(nombre, fontsize=9)
        ax.set_xlim(-P.SEMIANCHO_FRANJA_HZ / 1e3, P.SEMIANCHO_FRANJA_HZ / 1e3)
        ax.set_ylim(-70, 3)
        ax.set_xlabel("kHz respecto del centro del canal")
    axs[0].set_ylabel("densidad espectral (dB)")
    axs[1].text(0, -66, "zona sombreada: ventana de medida de la energía (±6 kHz)", ha="center", fontsize=7, color=COLORES[0])
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return salida


# ---------------------------------------------------------------- resultados
def cascada_demo(carpeta: Path, salida: Path, simulado: bool) -> Path:
    from rf_cascada import figura_casos
    res = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    rutas = {c: carpeta / f"{v['archivo']}.npz" for c, v in res["casos"].items()}
    titulos = {c: E.CONFIGURACIONES[c]["titulo"] for c in rutas}
    return figura_casos(rutas, titulos, salida, aviso_simulado=simulado or res.get("simulado", False))


def energia_medida(simulado: bool, salida: Path) -> Path:
    """Histograma de la energía medida por tipo de canal, con las distribuciones que supone el simulador."""
    plt = _plt()
    raiz = E.carpeta_salida(simulado) / "campanas"
    grupos = {"sin emisión (libre o degradado)": [], "con emisión (primario, emulador o interferencia)": []}
    for ruta in raiz.glob("*/episodios/**/s*.json"):
        ep = json.loads(ruta.read_text(encoding="utf-8"))
        for m in ep.get("radio", {}).get("medidas", []):
            clave = list(grupos)[1] if m["verdad"]["emision"] else list(grupos)[0]
            grupos[clave].append(m["medida"]["energia_dbm"])
    if not any(grupos.values()):
        raise FileNotFoundError(f"no hay episodios con radio en {raiz}")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.arange(-120, -20, 1.0)
    for (nombre, v), c in zip(grupos.items(), COLORES[:2]):
        if v:
            ax.hist(v, bins=bins, color=c, alpha=0.75, label=f"radio real, {nombre}\n(n = {len(v)})", density=True)
    x = np.linspace(-120, -20, 600)
    for mu, sd, nombre, c in ((-94, 6, "simulador: canal libre N(-94, 6)", COLORES[2]),
                              (-90, 5, "simulador: canal degradado N(-90, 5)", COLORES[3]),
                              (-62, 10, "simulador: emisión N(-62, 10)", GRIS)):
        ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), color=c, lw=1.0, label=nombre)
    ax.axvline(-78, color="#000000", lw=0.8, ls="--")
    ax.text(-77.3, ax.get_ylim()[1] * 0.04, "umbral -78 dBm", fontsize=7, va="bottom")
    ax.set_xlabel("energía que recibe el sistema (dBm, escala del simulador)")
    ax.set_ylabel("densidad")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if simulado:
        marca_simulado(fig)
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return salida


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", default=None, help="carpeta de una demostración (por defecto, la última)")
    p.add_argument("--simulado", action="store_true", help="usa los resultados del banco simulado")
    args = p.parse_args()
    base = E.carpeta_salida(args.simulado)
    out = base / "figuras"
    out.mkdir(parents=True, exist_ok=True)
    hechas = [plan_frecuencias(out / "plan_frecuencias.png"), formas_onda(out / "formas_onda.png")]
    demos = sorted(base.glob("demo_*/resumen.json"))
    carpeta = Path(args.demo) if args.demo else (demos[-1].parent if demos else None)
    if carpeta is not None and (carpeta / "resumen.json").exists():
        hechas.append(cascada_demo(carpeta, out / "cascada_casos.png", args.simulado))
    else:
        print("   (sin demostración todavía: no hay cascada de los casos)")
    try:
        hechas.append(energia_medida(args.simulado, out / "energia_medida.png"))
    except FileNotFoundError as e:
        print(f"   (sin minicampaña todavía: {e})")
    for h in hechas:
        print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
