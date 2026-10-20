#!/usr/bin/env python3
"""Demostración con radio real: los cuatro casos sobre el mismo episodio, con la cascada en directo.

  python3 poc_radio/demo_radio.py --buscar                        # propone episodios a partir de las campañas
  python3 poc_radio/demo_radio.py --escenario JAM_C --semilla 23  # los cuatro casos, con la cascada en una ventana
  python3 poc_radio/demo_radio.py --escenario JAM_C --semilla 23 --casos escudo --sin-cascada
  python3 poc_radio/demo_radio.py --simulado --llm-simulado --escenario JAM_C --semilla 20   # sin hardware ni Ollama

Casos, en este orden por defecto:
  sin_escudo   agente sin escudo con los textos de la campaña: la inyección puede conseguir la acción perjudicial;
  escudo       el mismo agente con el escudo: el escudo bloquea la acción perjudicial;
  insistente   con escudo y el texto insistente: el escudo aguanta, pero el agente puede dejar de transmitir;
  simplex      con escudo, texto insistente y respaldo Simplex: al primer bloqueo de un paso decide el árbol de reglas.

Cada ejecución se guarda en resultados/poc_radio/demo_<fecha>_<hora>/ (episodios, cascadas, resumen y figura de
los casos). Con --simulado, en resultados/poc_radio_simulado/. Cerrar la ventana de la cascada detiene la
demostración y deja los dos Pluto en silencio.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import rf_ejecucion as E  # noqa: E402  (añade src al path)
from rf_dispositivos import ErrorRadio  # noqa: E402

CASOS_DEMO = ("sin_escudo", "escudo", "insistente", "simplex")


# ---------------------------------------------------------------- búsqueda de episodios para la demostración
def buscar(args, cfg) -> int:
    """Episodios en que la campaña simulada muestra los cuatro comportamientos y el sensado simulado no tiene errores."""
    from spectrum_shield.entorno.simulador import Simulador
    from spectrum_shield.evaluacion import Campana, id_sistema
    from spectrum_shield.incidentes import seleccionar
    res = E.RAIZ / "resultados"
    fuentes = {caso: (E.REFERENCIA_SIMULADA[caso], E.CONFIGURACIONES[caso]["sistema"]) for caso in CASOS_DEMO}
    faltan = [f"{c}: resultados/{camp}" for c, (camp, _) in fuentes.items() if not (res / camp).exists()]
    if faltan:
        print("Faltan campañas de referencia:\n  " + "\n  ".join(faltan))
    candidatos = []
    for esc in seleccionar(args.escenarios):
        if not esc.compuesto:
            continue
        for s in E.leer_semillas(args.semillas):
            eps = {}
            for caso, (camp, sis) in fuentes.items():
                r = Campana(res, camp).ruta(id_sistema(sis, args.modelo), esc.codigo, s)
                if r.exists():
                    eps[caso] = json.loads(r.read_text(encoding="utf-8"))
            if "sin_escudo" not in eps or "escudo" not in eps:
                continue
            m = {c: ep["metricas"] for c, ep in eps.items()}
            puntos = 0
            puntos += 4 if m["sin_escudo"]["m1"] else 0
            puntos += 2 if (not m["escudo"]["m1"] and m["escudo"]["m3"]) else 0
            if "insistente" in m:
                puntos += 2 if (not m["insistente"]["m1"] and not m["insistente"]["m3"]) else 0
            if "simplex" in m:
                puntos += 2 if (not m["simplex"]["m1"] and m["simplex"]["m3"]) else 0
            sim = Simulador(cfg)
            sim.reiniciar(esc, s)
            errores = 0
            for t in range(cfg.n_pasos):
                for c in range(cfg.n_canales):
                    e, med = sim.guion[t][c], sim.medidas[t][c]
                    errores += int((med["energia_dbm"] > cfg.umbral_energia_dbm) != e.hay_emision())
                    errores += int(bool(med["firma"]) != e.firma_primaria())
            iny = sim.inyeccion or {}
            visible = iny.get("rol_canal") in ("con_primario", "no_autorizado")
            puntos += 1 if visible else 0
            puntos -= errores
            candidatos.append((puntos, esc.codigo, s, iny.get("id"), iny.get("rol_canal"), iny.get("canal_objetivo"),
                               sim.c0, sim.refugio, errores, {c: (int(bool(v["m1"])), int(bool(v["m3"]))) for c, v in m.items()}))
    candidatos.sort(key=lambda x: (-x[0], x[1], x[2]))
    print("puntos  escenario semilla inyección (papel, canal)        inicial refugio errores_sensado  (M1, M3) por caso")
    for c in candidatos[: args.n_candidatos]:
        print(f"{c[0]:6d}  {c[1]:9s} {c[2]:7d} {c[3]} ({c[4]}, C{c[5]})".ljust(56) +
              f" C{c[6]}      C{c[7]}      {c[8]:3d}             {c[9]}")
    if not candidatos:
        print("No hay episodios de referencia con los cuatro casos.")
        return 1
    return 0


# ---------------------------------------------------------------- demostración
def correr_casos(args, cfg, exp, cliente, escudo, control, esc, salida: Path, estado: dict) -> None:
    from spectrum_shield.evaluacion import id_sistema
    resumen = {"escenario": esc.codigo, "semilla": args.semilla, "modelo": args.modelo, "casos": {},
               "simulado": args.simulado, "llm_simulado": args.llm_simulado, "version_poc": E.VERSION_POC,
               "huellas_poc": E.huellas_poc()}
    try:
        for i, caso in enumerate(args.casos, 1):
            if estado.get("parar"):
                break
            control.estado["cabecera"] = f"Caso {i} de {len(args.casos)}"
            print(f"\n== caso {i} de {len(args.casos)}: {E.CONFIGURACIONES[caso]['titulo']} ({E.fmt_t(time.time())})",
                  flush=True)
            t0 = time.time()
            ep, grab = E.correr_episodio(exp, caso, esc, args.semilla, cfg, args.modelo, cliente, escudo, control,
                                         campana=f"demo_radio_{caso}")
            nombre = f"{caso}__{id_sistema(ep['sistema'], ep['modelo'])}__{esc.codigo}_s{args.semilla:03d}"
            (salida / f"{nombre}.json").write_text(json.dumps(ep, ensure_ascii=False, indent=1, default=str),
                                                    encoding="utf-8")
            E.guardar_cascada(salida / f"{nombre}.npz", grab, ep, control.piso_db, control.an)
            m = ep["metricas"]
            raf = [(x["paso"], x["rafaga"]["canal"] if x["rafaga"] else None) for x in ep["radio"]["rafagas"]]
            conf = sum(1 for x in ep["radio"]["rafagas"] if x["rafaga"] and x["rafaga"]["confirmada"])
            ntx = sum(1 for x in ep["radio"]["rafagas"] if x["rafaga"])
            print(f"   M1={E.marca(m['m1'])} M2={E.marca(m['m2'])} M3={E.marca(m['m3'])} M4={E.marca(m['m4'])} "
                  f"bloqueos={m['n_bloqueos']} | transmisiones (paso, canal) {raf} | confirmadas en el aire {conf}/{ntx} "
                  f"| {time.time() - t0:.0f} s", flush=True)
            resumen["casos"][caso] = {"archivo": nombre, "metricas": m, "rafagas": raf, "confirmadas": conf,
                                      "transmisiones": ntx, "duracion_s": round(time.time() - t0, 1)}
            if i < len(args.casos) and args.pausa > 0 and not estado.get("parar"):
                control.estado["cabecera"] = f"Siguiente caso en {args.pausa:g} s"
                time.sleep(args.pausa)
        control.estado["cabecera"] = "Fin de la demostración"
    except BaseException as e:  # noqa: BLE001 (se informa y se deja la radio en silencio)
        estado["error"] = e
        print(f"\nSe para la demostración: {e!r}", flush=True)
    finally:
        control.silencio()
        (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
        estado["fin"] = True
        try:
            if resumen["casos"]:
                from rf_cascada import figura_casos
                rutas = {c: salida / f"{v['archivo']}.npz" for c, v in resumen["casos"].items()}
                titulos = {c: E.CONFIGURACIONES[c]["titulo"] for c in rutas}
                fig = figura_casos(rutas, titulos, salida / "cascada_casos.png", aviso_simulado=args.simulado)
                print(f"\nFigura de los casos: {fig}")
        except Exception as e:  # noqa: BLE001
            print(f"No se ha podido dibujar la figura: {e!r}")
        print(f"Resultados en {salida}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--escenario", default="JAM_C")
    p.add_argument("--semilla", type=int, default=None)
    p.add_argument("--casos", default=",".join(CASOS_DEMO), help=f"de {', '.join(E.CONFIGURACIONES)}")
    p.add_argument("--modelo", default="granite4.1:8b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--motor-escudo", default="auto", choices=("auto", "opa", "python"))
    p.add_argument("--config", default=str(E.RAIZ / "config" / "experimento.json"))
    p.add_argument("--pausa", type=float, default=5.0, help="segundos entre casos")
    p.add_argument("--sin-cascada", action="store_true", help="sin ventana (por ejemplo, por SSH)")
    p.add_argument("--recalibrar", action="store_true", help="calibración completa antes de empezar")
    p.add_argument("--simulado", action="store_true", help="banco simulado, sin hardware")
    p.add_argument("--llm-simulado", action="store_true", help="agente simulado, sin Ollama (solo para probar)")
    p.add_argument("--buscar", action="store_true", help="propone episodios a partir de las campañas simuladas")
    p.add_argument("--escenarios", default="JAM_C,PUE_C,SSDF_C,FALLO_C", help="para --buscar")
    p.add_argument("--semillas", default="20-59", help="para --buscar")
    p.add_argument("--n-candidatos", type=int, default=15)
    args = p.parse_args()

    from spectrum_shield.entorno import Config
    from spectrum_shield.incidentes import seleccionar
    ruta_cfg = Path(args.config)
    cfg = Config.desde_json(ruta_cfg) if ruta_cfg.exists() else Config()
    cfg.validar()
    if args.buscar:
        return buscar(args, cfg)
    if args.semilla is None:
        print("Falta --semilla (usa --buscar para elegir un episodio)")
        return 2
    args.casos = [c.strip() for c in args.casos.split(",") if c.strip()]
    desconocidos = [c for c in args.casos if c not in E.CONFIGURACIONES]
    if desconocidos:
        print(f"Casos desconocidos: {desconocidos}. Válidos: {list(E.CONFIGURACIONES)}")
        return 2
    esc = seleccionar(args.escenario)[0]

    exp = E.Experimento()
    from spectrum_shield.sistemas import ClienteOllama, ErrorConexionOllama
    if args.llm_simulado:
        cliente = exp.ADA._ClienteSimulado()
        args.modelo = "simulado"
    else:
        cliente = ClienteOllama(args.url)
        try:
            if args.modelo not in cliente.modelos_locales():
                print(f"Falta el modelo {args.modelo} en Ollama")
                return 2
        except ErrorConexionOllama as e:
            print(e)
            return 2
    escudo = exp.EscudoRegistrado(cfg, motor=args.motor_escudo)

    salida = E.carpeta_salida(args.simulado) / time.strftime("demo_%Y%m%d_%H%M%S")
    salida.mkdir(parents=True, exist_ok=True)
    print(f"SPECTRUM-SHIELD, prueba de concepto con radio real | {esc.codigo} semilla {args.semilla} | "
          f"{args.modelo} | casos {', '.join(args.casos)}" + (" | BANCO SIMULADO" if args.simulado else ""))
    try:
        control, cal = E.preparar_radio(args.simulado, args.recalibrar)
    except ErrorRadio as e:
        print(f"FALLO al preparar la radio: {e}")
        return 2
    (salida / "calibracion_sesion.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")
    estado = {}
    try:
        if args.sin_cascada:
            correr_casos(args, cfg, exp, cliente, escudo, control, esc, salida, estado)
        else:
            import matplotlib
            if args.simulado and matplotlib.get_backend().lower() == "agg":
                print("(sin pantalla: se sigue sin ventana)")
                correr_casos(args, cfg, exp, cliente, escudo, control, esc, salida, estado)
            else:
                from rf_cascada import VisorCascada
                hilo = threading.Thread(target=correr_casos, daemon=True,
                                        args=(args, cfg, exp, cliente, escudo, control, esc, salida, estado))
                visor = VisorCascada(control)
                hilo.start()
                visor.mostrar()               # hasta que se cierre la ventana
                estado["parar"] = True
                if not estado.get("fin"):
                    print("\nVentana cerrada: se para la demostración al terminar el episodio en curso.")
                    hilo.join(timeout=600)
    except KeyboardInterrupt:
        estado["parar"] = True
        print("\nInterrumpido.")
    finally:
        control.cerrar()
    return 1 if estado.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
