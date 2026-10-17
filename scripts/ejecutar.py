#!/usr/bin/env python3
"""Ejecuta una campaña de episodios y guarda las trayectorias.

Ejemplos:
  python3 scripts/ejecutar.py --campana piloto --semillas 0-4
  python3 scripts/ejecutar.py --campana campana --semillas 20-39 --modelos qwen3.8:27b,granite4.1:8b
  python3 scripts/ejecutar.py --campana reglas --sistemas reglas --semillas 20-39

Las semillas 0 a 4 son las del piloto y las de 0 a 19 se usaron durante el desarrollo. La campaña
usa las semillas 20 a 39, que no se han mirado antes, para que ningún episodio visto al ajustar el
diseño entre en el análisis.

Si se interrumpe, basta con volver a lanzar el mismo comando: los episodios que
ya tienen su JSON se saltan. En el Mac conviene lanzarlo con caffeinate -i para
que el equipo no se suspenda durante la noche.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield import __version__  # noqa: E402
from spectrum_shield.entorno import Config  # noqa: E402
from spectrum_shield.evaluacion import Campana, cargar_csv, ejecutar_episodio, id_sistema  # noqa: E402
from spectrum_shield.incidentes import PLANTILLAS, VERSION_BANCO, seleccionar  # noqa: E402
from spectrum_shield.sistemas import (AgenteLLM, ClienteOllama, ErrorConexionOllama,  # noqa: E402
                                      ErrorEscudo, ErrorOllama, Escudo, SistemaReglas)
from spectrum_shield.sistemas.prompts import esquemas_herramientas, prompt_sistema  # noqa: E402

SISTEMAS = ("reglas", "agente", "agente_escudo")
MODELOS_POR_DEFECTO = "qwen3.8:27b,granite4.1:8b"


def leer_semillas(texto: str):
    semillas = []
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-", 1)
            semillas.extend(range(int(a), int(b) + 1))
        else:
            semillas.append(int(parte))
    return sorted(set(semillas))


def formato_tiempo(segundos: float) -> str:
    segundos = int(max(0, segundos))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def plan(sistemas, modelos, escenarios, semillas):
    """Orden de ejecución: primero las reglas; después, modelo a modelo, cada semilla con sus pares."""
    tareas = []
    if "reglas" in sistemas:
        for s in semillas:
            for e in escenarios:
                tareas.append(("reglas", None, e, s))
    con_modelo = [x for x in sistemas if x != "reglas"]
    for m in modelos if con_modelo else []:
        for s in semillas:
            for e in escenarios:
                for x in con_modelo:
                    tareas.append((x, m, e, s))
    return tareas


def duracion_media_previa(raiz) -> dict:
    """Duración media por modelo de los episodios ya guardados en cualquier campaña.

    Solo sirve para estimar el tiempo restante desde el principio: sin ella, los modelos que aún no
    han empezado se estimarían con la media del primero, que puede ser mucho más lento.
    """
    suma, n = {}, {}
    for ruta in Path(raiz).glob("*/episodios.csv"):
        try:
            for f in cargar_csv(ruta):
                if f.get("modelo") and f.get("duracion_s"):
                    suma[f["modelo"]] = suma.get(f["modelo"], 0.0) + float(f["duracion_s"])
                    n[f["modelo"]] = n.get(f["modelo"], 0) + 1
        except (OSError, ValueError, KeyError):
            continue
    return {m: suma[m] / n[m] for m in suma}


def marca(v) -> str:
    return "-" if v is None else ("1" if v else "0")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campana", default="piloto", help="nombre de la campaña (carpeta en resultados/)")
    p.add_argument("--sistemas", default=",".join(SISTEMAS), help="reglas,agente,agente_escudo")
    p.add_argument("--modelos", default=MODELOS_POR_DEFECTO, help="modelos de Ollama separados por comas")
    p.add_argument("--escenarios", default="all", help="all o lista separada por comas (JAM_S,PUE_C...)")
    p.add_argument("--semillas", default="0-4", help="rango o lista: 20-39, 0,3,5")
    p.add_argument("--config", default=str(RAIZ / "config" / "experimento.json"))
    p.add_argument("--salida", default=str(RAIZ / "resultados"))
    p.add_argument("--url", default="http://localhost:11434", help="dirección de Ollama")
    p.add_argument("--motor-escudo", default="auto", choices=("auto", "opa", "python"))
    p.add_argument("--forzar", action="store_true", help="reanudar aunque cambien la configuración o el código")
    args = p.parse_args()

    sistemas = [x.strip() for x in args.sistemas.split(",") if x.strip()]
    desconocidos = set(sistemas) - set(SISTEMAS)
    if desconocidos:
        print(f"Sistemas desconocidos: {sorted(desconocidos)}. Válidos: {', '.join(SISTEMAS)}")
        return 2
    modelos = [x.strip() for x in args.modelos.split(",") if x.strip()]
    escenarios = seleccionar(args.escenarios)
    semillas = leer_semillas(args.semillas)
    ruta_cfg = Path(args.config)
    cfg = Config.desde_json(ruta_cfg) if ruta_cfg.exists() else Config()
    cfg.validar()

    escudo = Escudo(cfg, motor=args.motor_escudo)
    usa_modelo = any(x != "reglas" for x in sistemas)
    cliente = None
    version_ollama = None
    if usa_modelo:
        cliente = ClienteOllama(args.url)
        try:
            version_ollama = cliente.version()
            locales = cliente.modelos_locales()
        except ErrorConexionOllama as e:
            print(e)
            return 2
        faltan = [m for m in modelos if m not in locales]
        if faltan:
            print("Faltan modelos en Ollama: " + ", ".join(faltan))
            print("Descárgalos con: " + " && ".join(f"ollama pull {m}" for m in faltan))
            return 2

    campana = Campana(args.salida, args.campana)
    try:
        campana.preparar_manifiesto(cfg, {
            "sistemas": sistemas, "modelos": modelos if usa_modelo else [],
            "escenarios": [e.codigo for e in escenarios], "semillas": semillas,
            "ollama": version_ollama, "motor_escudo": escudo.version_motor(),
        }, forzar=args.forzar)
    except RuntimeError as e:
        print(e)
        return 2
    campana.anotar_en_manifiesto("banco_inyecciones", {"version": VERSION_BANCO,
                                                       "plantillas": [pl.id for pl in PLANTILLAS]})
    campana.anotar_en_manifiesto("prompt_sistema", prompt_sistema(cfg))
    campana.anotar_en_manifiesto("herramientas", esquemas_herramientas(cfg))
    campana.reconstruir_csv()

    tareas = plan(sistemas, modelos, escenarios, semillas)
    pendientes = [t for t in tareas if not campana.existe(id_sistema(t[0], t[1]), t[2].codigo, t[3])]
    print(f"SPECTRUM-SHIELD {__version__} | campaña {args.campana} | escudo: {escudo.version_motor()}"
          + (f" | Ollama {version_ollama}" if version_ollama else ""))
    print(f"{len(tareas)} episodios en total, {len(tareas) - len(pendientes)} ya hechos, "
          f"{len(pendientes)} pendientes.")

    duraciones = {}
    previas = duracion_media_previa(args.salida)
    fallos = []
    seguidos = 0
    restantes_llm = sum(1 for t in pendientes if t[0] != "reglas")
    try:
        for i, (sis, modelo, esc, semilla) in enumerate(pendientes, 1):
            if sis == "reglas":
                sistema, modo = SistemaReglas(cfg), "sombra"
            else:
                sistema = AgenteLLM(cfg, modelo, cliente, nombre=sis)
                modo = "activo" if sis == "agente_escudo" else "sombra"
            etiqueta = f"{sis}" + (f"[{modelo}]" if modelo else "")
            inicio = time.time()
            try:
                ep = ejecutar_episodio(sistema, esc, semilla, cfg, escudo, modo, campana=args.campana)
            except ErrorConexionOllama as e:
                print(f"\n{e}\nSe para la campaña. Vuelve a lanzar el mismo comando para continuar.")
                return 2
            except (ErrorOllama, ErrorEscudo) as e:
                fallos.append((etiqueta, esc.codigo, semilla, str(e)))
                seguidos += 1
                print(f"[{i}/{len(pendientes)}] {etiqueta} {esc.codigo} s={semilla}: ERROR {e}")
                if seguidos >= 3:
                    print("Tres errores seguidos: se para la campaña.")
                    return 1
                continue
            seguidos = 0
            campana.guardar(ep)
            campana.anadir_fila(ep)
            dur = time.time() - inicio
            m = ep["metricas"]
            texto_eta = ""
            if sis != "reglas":
                duraciones.setdefault(modelo, []).append(dur)
                restantes_llm -= 1
                medias = {**previas, **{k: sum(v) / len(v) for k, v in duraciones.items()}}
                media_global = sum(sum(v) for v in duraciones.values()) / sum(len(v) for v in duraciones.values())
                eta = sum(medias.get(t[1], media_global) for t in pendientes[i:] if t[0] != "reglas")
                texto_eta = f" | quedan {restantes_llm} con modelo, ETA {formato_tiempo(eta)}"
            print(f"[{i}/{len(pendientes)}] {etiqueta} {esc.codigo} s={semilla}: "
                  f"M1={marca(m['m1'])} M2={marca(m['m2'])} M3={marca(m['m3'])} M4={marca(m['m4'])} "
                  f"M5={marca(m['m5'])} ({dur:.1f} s){texto_eta}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrumpido. Vuelve a lanzar el mismo comando para continuar donde se quedó.")
        return 130
    finally:
        n = campana.reconstruir_csv()
        print(f"\nTabla de episodios: {campana.ruta_csv} ({n} filas)")

    if fallos:
        print(f"\n{len(fallos)} episodios con error (se repetirán al relanzar):")
        for f in fallos:
            print("  ", *f)
    print(f"Resumen: python3 scripts/resumen.py --campana {args.campana}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
