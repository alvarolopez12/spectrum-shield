#!/usr/bin/env python3
"""Minicampaña con radio real: los mismos episodios que la campaña simulada, ahora con sensado real.

  caffeinate -i python3 poc_radio/minicampana_radio.py              # 5 configuraciones x 4 escenarios x 5 semillas
  python3 poc_radio/minicampana_radio.py --resumen                   # radio real frente a simulado y calidad del sensado
  python3 poc_radio/minicampana_radio.py --simulado --llm-simulado   # prueba sin hardware ni Ollama

Por defecto: escenarios compuestos JAM_C, PUE_C, SSDF_C y FALLO_C; semillas 20 a 24 (las cinco primeras de la
campaña, fijadas antes de ver ningún resultado con radio); configuraciones reglas, sin_escudo, escudo, insistente
y simplex. Son 100 episodios, 80 con el modelo: unas 2 a 3 horas con granite4.1:8b. Se puede interrumpir con
Ctrl+C y relanzar: sigue donde se quedó. Cada 20 minutos repite la calibración de frecuencia.

Es una demostración ampliada, no una evaluación nueva de las hipótesis: 20 episodios por configuración no dan
para contrastes con potencia. Lo que sí permite medir es (a) si los sistemas se comportan con sensado real como
en la simulación, episodio a episodio y con las mismas semillas, y (b) cómo es el sensado real frente al modelo
del simulador (probabilidades de detección y de falsa alarma, sobre cientos de consultas con verdad conocida).

Cada texto se guarda como una campaña aparte en resultados/poc_radio/campanas/ (radio_original y
radio_insistente), con la misma estructura que las simuladas, más el registro de radio de cada episodio y su
cascada (cascadas/, con filas de 128 ms).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import rf_ejecucion as E  # noqa: E402  (añade src al path)
from rf_dispositivos import ErrorRadio  # noqa: E402

CONFIGURACIONES_DEFECTO = "reglas,sin_escudo,escudo,insistente,simplex"
RECALIBRAR_CADA_S = 20 * 60
# modelo del simulador para las medidas propias (config.json y simulador.py): P(energía > -78 dBm)
MODELO_SIMULADOR = {"pd_energia": 0.5 * math.erfc((-78 + 62) / (10 * math.sqrt(2))),
                    "pfa_energia_libre": 0.5 * math.erfc((-78 + 94) / (6 * math.sqrt(2))),
                    "pfa_energia_degradado": 0.5 * math.erfc((-78 + 90) / (5 * math.sqrt(2))),
                    "pd_firma": 0.90, "pfa_firma": 0.03}


def campana_de(conf: str) -> str:
    return "radio_" + E.CONFIGURACIONES[conf]["texto"]


def raiz_campanas(simulado: bool) -> Path:
    return E.carpeta_salida(simulado) / "campanas"


# ---------------------------------------------------------------- ejecución
def ejecutar(args, cfg, escenarios, semillas, confs) -> int:
    from spectrum_shield.evaluacion import Campana, id_sistema
    from spectrum_shield.sistemas import ClienteOllama, ErrorConexionOllama, ErrorEscudo, ErrorOllama

    exp = E.Experimento()
    if args.llm_simulado:
        cliente, modelo = exp.ADA._ClienteSimulado(), "simulado"
    else:
        cliente, modelo = ClienteOllama(args.url), args.modelo
        try:
            if modelo not in cliente.modelos_locales():
                print(f"Falta el modelo {modelo} en Ollama")
                return 2
        except ErrorConexionOllama as e:
            print(e)
            return 2
    escudo = exp.EscudoRegistrado(cfg, motor=args.motor_escudo)
    raiz = raiz_campanas(args.simulado)
    campanas = {}
    for conf in confs:
        nombre = campana_de(conf)
        if nombre not in campanas:
            c = Campana(raiz, nombre)
            try:
                c.preparar_manifiesto(cfg, {"sistemas": sorted({E.CONFIGURACIONES[k]["sistema"] for k in confs
                                                                if campana_de(k) == nombre}),
                                            "modelos": [modelo], "escenarios": [e.codigo for e in escenarios],
                                            "semillas": semillas, "ollama": cliente.version(),
                                            "motor_escudo": escudo.version_motor()}, forzar=args.forzar)
            except RuntimeError as e:
                print(e)
                return 2
            c.anotar_en_manifiesto("radio", {"version_poc": E.VERSION_POC, "huellas_poc": E.huellas_poc(),
                                             "simulado": args.simulado, "texto": E.CONFIGURACIONES[conf]["texto"],
                                             "huella_respaldo_simplex": E.huella(E.RAIZ / "scripts" / "respaldo_simplex.py"),
                                             "huella_ataque_adaptativo": E.huella(E.RAIZ / "scripts" / "ataque_adaptativo.py")})
            campanas[nombre] = c
    tareas = [(conf, esc, s) for s in semillas for esc in escenarios for conf in confs]
    pendientes = [(conf, esc, s) for conf, esc, s in tareas
                  if not campanas[campana_de(conf)].existe(id_sistema(E.CONFIGURACIONES[conf]["sistema"],
                                                                      None if conf == "reglas" else modelo),
                                                           esc.codigo, s)]
    print(f"SPECTRUM-SHIELD, minicampaña con radio real | {modelo} | {len(tareas)} episodios, {len(pendientes)} pendientes"
          + (" | BANCO SIMULADO" if args.simulado else ""))
    if not pendientes:
        print("No queda nada pendiente. Resumen: añade --resumen")
        return 0
    try:
        control, cal = E.preparar_radio(args.simulado, args.recalibrar)
    except ErrorRadio as e:
        print(f"FALLO al preparar la radio: {e}")
        return 2
    sesion = {"inicio": time.strftime("%Y-%m-%dT%H:%M:%S"), "calibracion": cal, "recalibraciones": []}
    t_cal = time.time()
    inicio, hechos = time.time(), 0
    try:
        for k, (conf, esc, s) in enumerate(pendientes, 1):
            if time.time() - t_cal > RECALIBRAR_CADA_S:
                rec = {p: control.calibrar_frecuencia(p) for p in control.tx}
                rec["t"] = time.strftime("%H:%M:%S")
                sesion["recalibraciones"].append(rec)
                print(f"   recalibración de frecuencia: entorno {rec['entorno']['desfase_hz']:+.0f} Hz, "
                      f"secundario {rec['secundario']['desfase_hz']:+.0f} Hz", flush=True)
                t_cal = time.time()
            campana = campanas[campana_de(conf)]
            control.estado["cabecera"] = f"Minicampaña {k}/{len(pendientes)}"
            t0 = time.time()
            ep, grab = E.correr_episodio(exp, conf, esc, s, cfg, modelo, cliente, escudo, control, campana.nombre,
                                         decimar=4)
            ruta = campana.guardar(ep)
            campana.anadir_fila(ep)
            rel = ruta.relative_to(campana.dir_episodios).with_suffix(".npz")
            E.guardar_cascada(campana.dir / "cascadas" / rel, grab, ep, control.piso_db, control.an)
            hechos += 1
            m, r = ep["metricas"], ep["radio"]
            dif = sum(1 for x in r["medidas"] if x["real"]["deteccion_local"] != x["simulado"]["deteccion_local"]
                      or x["real"]["firma_primario"] != x["simulado"]["firma_primario"])
            conf_tx = sum(1 for x in r["rafagas"] if x["rafaga"] and x["rafaga"]["confirmada"])
            ntx = sum(1 for x in r["rafagas"] if x["rafaga"])
            print(f"[{k}/{len(pendientes)}] {conf:10s} {esc.codigo} s={s}: M1={E.marca(m['m1'])} M2={E.marca(m['m2'])} "
                  f"M3={E.marca(m['m3'])} M4={E.marca(m['m4'])} | sensado distinto del simulado {dif}/{len(r['medidas'])} "
                  f"| ráfagas confirmadas {conf_tx}/{ntx} ({time.time() - t0:.0f} s)", flush=True)
    except ErrorConexionOllama as e:
        print(f"\n{e}\nSe para. Vuelve a lanzar el mismo comando para continuar.")
        return 2
    except (ErrorOllama, ErrorEscudo, ErrorRadio) as e:
        print(f"\nError: {e}. Vuelve a lanzar el mismo comando para continuar.")
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido. Vuelve a lanzar el mismo comando para continuar donde se quedó.")
        return 130
    finally:
        control.cerrar()
        for c in campanas.values():
            c.reconstruir_csv()
        sesion["fin"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        sesion["episodios"] = hechos
        ruta_s = E.carpeta_salida(args.simulado) / "campanas" / "sesiones.jsonl"
        ruta_s.parent.mkdir(parents=True, exist_ok=True)
        with ruta_s.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sesion, ensure_ascii=False, default=str) + "\n")
    print(f"\n{hechos} episodios nuevos en {(time.time() - inicio) / 60:.0f} min. Resumen: añade --resumen")
    return 0


# ---------------------------------------------------------------- resumen
def tasa(k: int, n: int) -> dict:
    from spectrum_shield.evaluacion import agresti_coull
    p, lo, hi = agresti_coull(k, n)
    return {"k": k, "n": n, "p": None if n == 0 else round(p, 4), "ic95": None if n == 0 else [round(lo, 4), round(hi, 4)]}


def fmt(t: dict) -> str:
    if not t["n"]:
        return "  -  "
    return f"{t['k']}/{t['n']} ({100 * t['k'] / t['n']:.1f} %, IC95 {100 * t['ic95'][0]:.1f}-{100 * t['ic95'][1]:.1f})"


def resumen(args, cfg, escenarios, semillas, confs) -> int:
    from spectrum_shield.evaluacion import Campana, id_sistema
    raiz = raiz_campanas(args.simulado)
    res_sim = E.RAIZ / "resultados"
    modelo = "simulado" if args.llm_simulado else args.modelo
    datos = {"version_poc": E.VERSION_POC, "simulado": args.simulado, "modelo": modelo,
             "escenarios": [e.codigo for e in escenarios], "semillas": semillas, "configuraciones": {},
             "modelo_simulador": {k: round(v, 4) for k, v in MODELO_SIMULADOR.items()}}
    medidas, origenes, rafagas = [], [], []
    print(f"Minicampaña con radio real{' (BANCO SIMULADO)' if args.simulado else ''} | {modelo} | "
          f"{len(escenarios)} escenarios x {len(semillas)} semillas\n")
    print("Configuración   n    M1 real  M1 sim   M3 real  M3 sim   M4 real  M4 sim   iguales M1  iguales M3")
    for conf in confs:
        sis = E.CONFIGURACIONES[conf]["sistema"]
        mod = None if conf == "reglas" else modelo
        camp = Campana(raiz, campana_de(conf))
        ref = Campana(res_sim, E.REFERENCIA_SIMULADA[conf])
        filas = []
        for esc in escenarios:
            for s in semillas:
                r = camp.ruta(id_sistema(sis, mod), esc.codigo, s)
                if not r.exists():
                    continue
                ep = json.loads(r.read_text(encoding="utf-8"))
                rr = ref.ruta(id_sistema(sis, None if conf == "reglas" else args.modelo), esc.codigo, s)
                ep_ref = json.loads(rr.read_text(encoding="utf-8")) if rr.exists() else None
                filas.append((esc.codigo, s, ep, ep_ref))
                for x in ep["radio"]["medidas"]:
                    medidas.append({**x, "conf": conf, "escenario": esc.codigo, "semilla": s})
                for x in ep["radio"]["origenes"]:
                    origenes.append({**x, "conf": conf})
                for x in ep["radio"]["rafagas"]:
                    if x["rafaga"]:
                        rafagas.append({**x["rafaga"], "conf": conf, "perjudicial": x["perjudicial"]})
        if not filas:
            continue
        d = {"n": len(filas), "sistema": sis, "referencia": E.REFERENCIA_SIMULADA[conf]}
        linea = f"{conf:14s} {len(filas):3d}"
        for met in ("m1", "m2", "m3", "m4", "m5"):
            real = [bool(ep["metricas"][met]) for _, _, ep, _ in filas if ep["metricas"].get(met) is not None]
            pares = [(bool(ep["metricas"][met]), bool(ref["metricas"][met])) for _, _, ep, ref in filas
                     if ref is not None and ep["metricas"].get(met) is not None and ref["metricas"].get(met) is not None]
            d[met] = {"real": tasa(sum(real), len(real)),
                      "simulado_mismas_semillas": tasa(sum(b for _, b in pares), len(pares)),
                      "pares": len(pares), "iguales": sum(1 for a, b in pares if a == b),
                      "solo_real": sum(1 for a, b in pares if a and not b),
                      "solo_simulado": sum(1 for a, b in pares if b and not a)}
        for met in ("m1", "m3", "m4"):
            x = d[met]
            linea += f"   {x['real']['k']:2d}/{x['real']['n']:<3d}  " + (
                f"{x['simulado_mismas_semillas']['k']:2d}/{x['simulado_mismas_semillas']['n']:<3d}" if x["pares"] else "  -    ")
        linea += f"   {d['m1']['iguales']}/{d['m1']['pares']}".ljust(13) + f"{d['m3']['iguales']}/{d['m3']['pares']}"
        print(linea)
        d["episodios"] = [{"escenario": e, "semilla": s, "metricas_real": {k: ep["metricas"][k] for k in ("m1", "m2", "m3", "m4")},
                           "metricas_simulado": None if ref is None else {k: ref["metricas"][k] for k in ("m1", "m2", "m3", "m4")}}
                          for e, s, ep, ref in filas]
        datos["configuraciones"][conf] = d

    if not medidas:
        print("\nTodavía no hay episodios con radio.")
        return 1
    # ------------------------------------------------ calidad del sensado real
    def cuenta(cond_verdad, cond_det, lista=medidas):
        sel = [m for m in lista if cond_verdad(m)]
        return tasa(sum(1 for m in sel if cond_det(m)), len(sel))
    ocupado = lambda m: m["real"]["deteccion_local"] == "ocupado"  # noqa: E731
    ocupado_sim = lambda m: m["simulado"]["deteccion_local"] == "ocupado"  # noqa: E731
    sens = {
        "consultas": len(medidas),
        "pd_energia": cuenta(lambda m: m["verdad"]["emision"], ocupado),
        "pfa_energia_libre": cuenta(lambda m: m["verdad"]["tipo"] == "libre", ocupado),
        "pfa_energia_degradado": cuenta(lambda m: m["verdad"]["tipo"] == "degradado", ocupado),
        "pd_firma": cuenta(lambda m: m["verdad"]["firma"], lambda m: m["real"]["firma_primario"]),
        "pfa_firma_interferencia": cuenta(lambda m: m["verdad"]["tipo"] == "interferencia",
                                          lambda m: m["real"]["firma_primario"]),
        "pfa_firma_sin_emision": cuenta(lambda m: not m["verdad"]["emision"], lambda m: m["real"]["firma_primario"]),
        "simulado_pd_energia": cuenta(lambda m: m["verdad"]["emision"], ocupado_sim),
        "simulado_pfa_energia": cuenta(lambda m: not m["verdad"]["emision"], ocupado_sim),
        "acuerdo_deteccion_real_simulado": cuenta(lambda m: True, lambda m: ocupado(m) == ocupado_sim(m)),
        "acuerdo_firma_real_simulado": cuenta(lambda m: True,
                                              lambda m: m["real"]["firma_primario"] == m["simulado"]["firma_primario"]),
    }
    por_tipo = defaultdict(list)
    for m in medidas:
        por_tipo[m["verdad"]["tipo"]].append(m["medida"]["energia_dbm"])
    sens["energia_por_tipo"] = {t: {"n": len(v), "media": round(statistics.mean(v), 2),
                                    "desv": round(statistics.pstdev(v), 2), "min": round(min(v), 1), "max": round(max(v), 1)}
                                for t, v in sorted(por_tipo.items())}
    sens["origen_detectado_con_emision"] = cuenta(lambda m: m["verdad"]["emision"],
                                                  lambda m: m["real"]["emisor_detectado"], origenes)
    snr_tx = [r["durante_snr_db"] for r in rafagas if r.get("durante_snr_db") is not None]
    sens["rafagas"] = {"n": len(rafagas), "confirmadas": sum(1 for r in rafagas if r["confirmada"]),
                       "snr_media_db": round(statistics.mean(snr_tx), 1) if snr_tx else None}
    datos["sensado"] = sens
    ms = MODELO_SIMULADOR
    print(f"\nSensado real: {len(medidas)} consultas con verdad del terreno conocida")
    print(f"  detección de energía: Pd {fmt(sens['pd_energia'])}  (modelo del simulador {100 * ms['pd_energia']:.1f} %)")
    print(f"                        Pfa canal libre {fmt(sens['pfa_energia_libre'])}  (modelo {100 * ms['pfa_energia_libre']:.2f} %)")
    print(f"                        Pfa canal degradado {fmt(sens['pfa_energia_degradado'])}  (modelo {100 * ms['pfa_energia_degradado']:.2f} %)")
    print(f"  firma de primario:    Pd {fmt(sens['pd_firma'])}  (modelo {100 * ms['pd_firma']:.0f} %)")
    print(f"                        Pfa con interferencia {fmt(sens['pfa_firma_interferencia'])}  (modelo {100 * ms['pfa_firma']:.0f} %)")
    print(f"                        Pfa sin emisión {fmt(sens['pfa_firma_sin_emision'])}")
    print(f"  acuerdo con el sensado simulado de la misma consulta: detección {fmt(sens['acuerdo_deteccion_real_simulado'])}, "
          f"firma {fmt(sens['acuerdo_firma_real_simulado'])}")
    print("  energía medida por tipo de canal (dBm en la escala del simulador):")
    for t, v in sens["energia_por_tipo"].items():
        print(f"    {t:13s} n={v['n']:4d}  media {v['media']:7.1f}  desv {v['desv']:5.2f}  [{v['min']}, {v['max']}]")
    r = sens["rafagas"]
    print(f"  transmisiones del secundario confirmadas en el aire: {r['confirmadas']}/{r['n']} (SNR media {r['snr_media_db']} dB)")
    salida = E.carpeta_salida(args.simulado) / "datos_radio.json"
    salida.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nDatos para el capítulo 5: {salida}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configuraciones", default=CONFIGURACIONES_DEFECTO)
    p.add_argument("--escenarios", default="JAM_C,PUE_C,SSDF_C,FALLO_C")
    p.add_argument("--semillas", default="20-24")
    p.add_argument("--modelo", default="granite4.1:8b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--motor-escudo", default="auto", choices=("auto", "opa", "python"))
    p.add_argument("--config", default=str(E.RAIZ / "config" / "experimento.json"))
    p.add_argument("--recalibrar", action="store_true", help="calibración completa antes de empezar")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--resumen", action="store_true")
    p.add_argument("--simulado", action="store_true", help="banco simulado, sin hardware")
    p.add_argument("--llm-simulado", action="store_true", help="agente simulado, sin Ollama (solo para probar)")
    args = p.parse_args()
    from spectrum_shield.entorno import Config
    from spectrum_shield.incidentes import seleccionar
    ruta_cfg = Path(args.config)
    cfg = Config.desde_json(ruta_cfg) if ruta_cfg.exists() else Config()
    cfg.validar()
    confs = [c.strip() for c in args.configuraciones.split(",") if c.strip()]
    desconocidas = [c for c in confs if c not in E.CONFIGURACIONES]
    if desconocidas:
        print(f"Configuraciones desconocidas: {desconocidas}. Válidas: {list(E.CONFIGURACIONES)}")
        return 2
    escenarios = seleccionar(args.escenarios)
    semillas = E.leer_semillas(args.semillas)
    if args.resumen:
        return resumen(args, cfg, escenarios, semillas, confs)
    return ejecutar(args, cfg, escenarios, semillas, confs)


if __name__ == "__main__":
    sys.exit(main())
