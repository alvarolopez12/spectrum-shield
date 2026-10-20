"""Cascada de los seis canales: en directo durante la demostración y como figura a partir de una grabación.

Cada canal se dibuja como una franja de 25 kHz (su canal entero), separada de la siguiente por un hueco; el
color es la potencia sobre el ruido de fondo de cada bin, en dB. El tiempo avanza hacia abajo en las figuras y
hacia arriba en directo (lo más reciente arriba). En las figuras se marcan los cambios de paso, los bloqueos del
escudo y los diagnósticos.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

COLORMAPA = "viridis"
RANGO_DB = (0.0, 45.0)
HUECO = 8                    # columnas vacías entre franjas
COLOR_BLOQUEO = "#eb6834"
COLOR_PASO = "#ffffff"
NOMBRES_TIPO = {"primario": "primario", "emulador": "emulador (PUE)", "interferencia": "interferencia"}
CONF_SOMBRA = ("reglas", "sin_escudo")      # configuraciones con el escudo en sombra: anota, no bloquea


def imagen_franjas(franjas_db: np.ndarray, piso_db: Optional[np.ndarray] = None) -> np.ndarray:
    """(filas, 6, ancho) en dB -> imagen (filas, 6·ancho + huecos) en dB sobre el ruido, con NaN en los huecos."""
    f = franjas_db.astype(np.float32)
    if piso_db is not None:
        f = f - piso_db[None, :, :].astype(np.float32)
    else:
        f = f - np.median(f, axis=(0, 2), keepdims=True)
    n, k, w = f.shape
    out = np.full((n, k * w + (k - 1) * HUECO), np.nan, dtype=np.float32)
    for c in range(k):
        x0 = c * (w + HUECO)
        out[:, x0:x0 + w] = f[:, c, :]
    return out


def centros_franjas(ancho: int, n: int = 6) -> List[float]:
    return [c * (ancho + HUECO) + (ancho - 1) / 2 for c in range(n)]


def etiquetas_canales(meta: Dict, canales_hz) -> List[str]:
    out = []
    for c, f in enumerate(canales_hz):
        papel = []
        if meta.get("canal_inicial") == c:
            papel.append("inicial")
        if meta.get("canal_refugio") == c:
            papel.append("refugio")
        if c == 5:
            papel.append("protegido")
        if meta.get("canal_objetivo") == c and meta.get("inyeccion"):
            papel.append("pide la inyección")
        out.append(f"C{c}\n{f / 1e6:.2f} MHz".replace(".", ",") + ("\n" + ", ".join(papel) if papel else ""))
    return out


# ====================================================================== figura a partir de una grabación
def cargar(ruta: Path) -> Dict:
    d = np.load(ruta, allow_pickle=False)
    out = {"t": d["t"], "franjas": d["franjas"], "eventos": json.loads(str(d["eventos"])),
           "meta": json.loads(str(d["meta"]))}
    out["piso"] = d["piso_franjas_db"] if "piso_franjas_db" in d.files else None
    return out


def dibujar(ax, g: Dict, titulo: str = "", rango=RANGO_DB, eventos: bool = True, barra: bool = False):
    t, fr, meta = g["t"], g["franjas"], g["meta"]
    img = imagen_franjas(fr, g.get("piso"))
    ancho = fr.shape[2]
    dt = float(meta.get("dt_fila", 0.032))
    t_fin = float(t[-1] + dt) if len(t) else 1.0
    im = ax.imshow(img, aspect="auto", origin="upper", cmap=COLORMAPA, vmin=rango[0], vmax=rango[1],
                   extent=[-0.5, img.shape[1] - 0.5, t_fin, float(t[0]) if len(t) else 0.0], interpolation="nearest")
    ax.set_xticks(centros_franjas(ancho))
    ax.set_xticklabels(etiquetas_canales(meta, meta.get("canales_hz", [434.15e6 + 1e5 * k for k in range(6)])),
                       fontsize=7)
    ax.set_ylabel("tiempo desde el inicio del episodio (s)")
    ax.tick_params(axis="x", length=0)
    if titulo:
        ax.set_title(titulo, fontsize=9)
    if eventos:
        centros = centros_franjas(ancho)
        respaldo = {d["paso"] for d in meta.get("pasos", []) if d.get("respaldo")}
        marcas = []
        for e in g["eventos"]:
            if e["tipo"] == "paso":
                ax.axhline(e["t"], color=COLOR_PASO, lw=0.6, ls=(0, (4, 3)), alpha=0.8)
                marcas.append((e["t"], f"paso {e['paso']}" + (" (R)" if e["paso"] in respaldo else "")))
            elif (e["tipo"] == "escudo" and not e.get("permitida", True)
                  and e.get("modo", "sombra" if meta.get("configuracion") in CONF_SOMBRA else "activo") == "activo"):
                a = e.get("accion") or {}
                if a.get("accion") == "TRANSMITIR" and a.get("canal") is not None:
                    ax.plot(centros[int(a["canal"])], e["t"], marker="x", ms=7, mew=2, color=COLOR_BLOQUEO)
        # etiquetas de los pasos en el eje derecho; si dos pasos caen casi a la vez, se juntan
        alto = abs(ax.get_ylim()[0] - ax.get_ylim()[1]) or 1.0
        juntas = []
        for t_m, txt in marcas:
            if juntas and t_m - juntas[-1][0] < 0.035 * alto:
                juntas[-1] = (juntas[-1][0], juntas[-1][1] + ", " + txt.replace("paso ", ""))
            else:
                juntas.append((t_m, txt))
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([t_m for t_m, _ in juntas])
        ax2.set_yticklabels([txt.replace("paso ", "pasos ", 1) if "," in txt else txt for _, txt in juntas],
                            fontsize=6.5)
        ax2.tick_params(axis="y", length=2)
    if barra:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("dB sobre el ruido")
    return im


def describir_episodio(meta: Dict) -> str:
    papeles = {"con_primario": "canal con primario", "no_autorizado": "banda protegida", "activo": "canal activo"}
    txt = f"{meta.get('escenario')}, semilla {meta.get('semilla')}"
    if meta.get("inyeccion"):
        txt += f"; inyección {meta['inyeccion']}, que pide transmitir en C{meta.get('canal_objetivo')}"
        if meta.get("rol_canal") in papeles:
            txt += f" ({papeles[meta['rol_canal']]})"
    return txt + f"; canal inicial C{meta.get('canal_inicial')}, refugio C{meta.get('canal_refugio')}"


def figura_casos(rutas: Dict[str, Path], titulos: Dict[str, str], salida: Path, aviso_simulado: bool = False,
                 suptitulo: Optional[str] = None) -> Path:
    # sin pyplot: se puede llamar desde el hilo de la demostración mientras la ventana de la cascada está abierta
    from matplotlib.figure import Figure
    n = len(rutas)
    cols = 2 if n > 1 else 1
    filas = (n + cols - 1) // cols
    fig = Figure(figsize=(7.2 * cols, 5.2 * filas))
    axs = fig.subplots(filas, cols, squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.9, bottom=0.07, hspace=0.42, wspace=0.28)
    im = None
    meta0 = None
    for ax, (clave, ruta) in zip(axs.flat, rutas.items()):
        g = cargar(ruta)
        meta0 = meta0 or g["meta"]
        m = g["meta"].get("metricas", {})
        sub = f"{titulos.get(clave, clave)}\nM1={_m(m.get('m1'))} M3={_m(m.get('m3'))} M4={_m(m.get('m4'))}"
        im = dibujar(ax, g, sub)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, cax=fig.add_axes([0.925, 0.3, 0.012, 0.4]))
        cb.set_label("dB sobre el ruido de fondo")
    if meta0 is not None:
        fig.suptitle(suptitulo or describir_episodio(meta0), fontsize=10)
    if aviso_simulado:
        fig.text(0.5, 0.5, "SEÑALES SINTÉTICAS (sin hardware)", ha="center", va="center", fontsize=28,
                 color="#d00000", alpha=0.25, rotation=20)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=200)
    return salida


def _m(v) -> str:
    return "-" if v is None else ("1" if v else "0")


# ====================================================================== en directo
def texto_evento(e) -> Optional[str]:
    t, tipo, d = e
    h = time.strftime("%H:%M:%S", time.localtime(t))
    if tipo == "paso":
        em = ", ".join(f"{NOMBRES_TIPO.get(x[1], x[1])} en C{x[0]}" for x in d.get("emisiones", [])) or "nada"
        return f"{h}  paso {d['paso']}: el entorno emite {em}"
    if tipo == "medida":
        return f"{h}  sensado C{d['canal']}: {d['energia_dbm']:.0f} dBm{', firma de primario' if d['firma'] else ''}"
    if tipo == "escudo":
        a = d.get("accion") or {}
        acc = f"{a.get('accion', '?')}" + (f" C{a['canal']}" if a.get("canal") is not None else "")
        if d.get("permitida"):
            return f"{h}  escudo: permite {acc}"
        if d.get("modo") == "sombra":
            return f"{h}  escudo en sombra: habría bloqueado {acc} ({', '.join(d.get('infracciones', []))})"
        return f"{h}  ESCUDO: BLOQUEA {acc} ({', '.join(d.get('infracciones', []))})"
    if tipo == "rafaga":
        return f"{h}  transmite en C{d['canal']} ({d['duracion_s']:.1f} s, {d['potencia_dbm']:g} dBm pedidos)"
    if tipo == "espera":
        return f"{h}  paso {d['paso']}: no transmite"
    if tipo == "diagnostico":
        return f"{h}  diagnóstico: {d['etiqueta']}"
    if tipo == "episodio":
        return f"{h}  episodio {d['escenario']} semilla {d['semilla']}: inicial C{d['c0']}, refugio C{d['refugio']}"
    return None


class VisorCascada:
    """Ventana de matplotlib que muestra la captura en curso. Debe crearse y mostrarse en el hilo principal."""

    def __init__(self, control, segundos: float = 30.0, rango=RANGO_DB):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        self.plt = plt
        self.control, self.segundos, self.rango = control, segundos, rango
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_axes([0.05, 0.12, 0.58, 0.78])
        self.ax_txt = self.fig.add_axes([0.66, 0.12, 0.33, 0.78])
        self.ax_txt.axis("off")
        ancho = control.an.ancho_franja
        self.ancho = ancho
        n = int(segundos / control.captura.dt_fila)
        self.n = n
        vacia = np.full((n, 6 * ancho + 5 * HUECO), np.nan, dtype=np.float32)
        self.im = self.ax.imshow(vacia, aspect="auto", origin="upper", cmap=COLORMAPA, vmin=rango[0], vmax=rango[1],
                                 extent=[-0.5, vacia.shape[1] - 0.5, -segundos, 0], interpolation="nearest")
        self.ax.set_ylabel("segundos (0 = ahora)")
        self.ax.set_xticks(centros_franjas(ancho))
        cb = self.fig.colorbar(self.im, ax=self.ax, fraction=0.03, pad=0.01)
        cb.set_label("dB sobre el ruido")
        self.titulo = self.fig.suptitle("", fontsize=13)
        self.txt = self.ax_txt.text(0, 1, "", va="top", ha="left", fontsize=9, family="monospace")
        self.lineas = []
        self.anim = FuncAnimation(self.fig, self._actualizar, interval=250, cache_frame_data=False)

    def _actualizar(self, _):
        c = self.control
        t, fr = c.captura.ultimas_franjas(self.segundos)
        if t is None:
            return
        piso = None
        if c.piso_db is not None:
            piso = np.stack([c.piso_db[idx] for idx in c.an.franja])
        img = imagen_franjas(fr[::-1], piso)
        if img.shape[0] < self.n:
            img = np.vstack([img, np.full((self.n - img.shape[0], img.shape[1]), np.nan, dtype=np.float32)])
        self.im.set_data(img[: self.n])
        est = c.estado
        meta = {"canal_inicial": est.get("c0"), "canal_refugio": est.get("refugio")}
        from rf_plan import CANALES_HZ
        self.ax.set_xticklabels(etiquetas_canales(meta, CANALES_HZ), fontsize=8)
        tit = est.get("titulo", "")
        if est.get("escenario"):
            tit += f"  ·  {est['escenario']} semilla {est.get('semilla')}"
        if est.get("paso") is not None:
            tit += f"  ·  paso {est['paso']}"
        if est.get("cabecera"):
            tit = f"{est['cabecera']}  ·  {tit}"
        self.titulo.set_text(tit)
        for ln in self.lineas:
            ln.remove()
        self.lineas = []
        ahora = t[-1]
        with c.captura.cond:
            eventos = list(c.captura.eventos[-400:])
        centros = centros_franjas(self.ancho)
        for ev in eventos:
            dt = ev[0] - ahora
            if dt < -self.segundos:
                continue
            if ev[1] == "paso":
                self.lineas.append(self.ax.axhline(dt, color=COLOR_PASO, lw=0.7, ls=(0, (4, 3)), alpha=0.8))
            elif ev[1] == "escudo" and not ev[2].get("permitida", True) and ev[2].get("modo", "activo") == "activo":
                a = ev[2].get("accion") or {}
                if a.get("canal") is not None:
                    self.lineas.extend(self.ax.plot(centros[int(a["canal"])], dt, marker="x", ms=11, mew=3,
                                                    color=COLOR_BLOQUEO))
        lineas = [x for x in (texto_evento(e) for e in eventos) if x]
        self.txt.set_text("\n".join(lineas[-30:]))

    def mostrar(self):
        self.plt.show()
