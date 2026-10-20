#!/usr/bin/env python3
"""Comprobación del montaje de radio y calibración (prueba de concepto, apartado de radio real del capítulo 5).

  python3 poc_radio/comprobar_radio.py --listar       # bibliotecas y dispositivos conectados
  python3 poc_radio/comprobar_radio.py                # comprobación completa y calibración (menos de 1 min)
  python3 poc_radio/comprobar_radio.py --vigilar 120  # ocupación de la banda por emisiones ajenas, solo escucha
  python3 poc_radio/comprobar_radio.py --simulado     # lo mismo con el banco simulado, sin hardware

Pasos de la comprobación completa:
  1. plan de frecuencias frente a la normativa SRD (banda, canalización, potencia estimada, imágenes);
  2. bibliotecas (pyadi-iio, libiio, pyrtlsdr, librtlsdr) y dispositivos (dos Pluto y un RTL-SDR);
  3. ruido de fondo de cada canal con los dos Pluto en silencio;
  4. error de frecuencia de cada Pluto respecto del RTL-SDR, con un tono, y comprobación de la corrección;
  5. ganancias para que las emisiones lleguen con la SNR objetivo (30 dB el entorno, 36 dB el secundario);
  6. detector de firma: primario en los seis canales (firma) e interferencia en los seis (sin firma);
  7. artefactos: imagen del Pluto, fuga con los Pluto en silencio y saturación del receptor en el peor caso.
La calibración se guarda en resultados/poc_radio/calibracion.json (o en poc_radio_simulado con --simulado) y la
usan la demostración y la minicampaña, que en cada sesión repiten la parte rápida (ruido y frecuencia).
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import rf_plan as P  # noqa: E402
from rf_dispositivos import ErrorRadio, listar_pluto, preparar_bibliotecas, serie_de  # noqa: E402
from rf_ondas import ANCHOS_OCUPADOS_HZ  # noqa: E402


def titulo(n: int, texto: str) -> None:
    print(f"\n{n}. {texto}")


def bibliotecas() -> bool:
    preparar_bibliotecas()
    ok = True
    for nombre, modulo, ayuda in (("numpy", "numpy", "pip install numpy"),
                                  ("pyadi-iio", "adi", "pip install pyadi-iio (y libiio: README, paso 2)"),
                                  ("libiio (Python)", "iio", "pip install pylibiio, con libiio instalada"),
                                  ("pyrtlsdr", "rtlsdr", "pip install pyrtlsdr (y librtlsdr: README, paso 2)")):
        try:
            m = importlib.import_module(modulo)
            v = getattr(m, "__version__", "") or getattr(m, "version", "")
            v = ".".join(str(x) for x in v[:2]) if isinstance(v, tuple) else str(v)
            print(f"   correcto: {nombre} {v} ({Path(getattr(m, '__file__', '') or '?').parent})")
        except Exception as e:  # noqa: BLE001 (también falla si no encuentra la biblioteca en C)
            ok = False
            print(f"   FALLO: {nombre}: {e}. {ayuda}")
    return ok


def dispositivos() -> bool:
    ok = True
    try:
        plutos = listar_pluto()
        if not plutos:
            ok = False
            print("   FALLO: no se ve ningún Pluto por USB")
        for u, d in plutos.items():
            print(f"   Pluto {u}  serie {serie_de(d)}")
        if len(plutos) == 1:
            print("   aviso: solo hay un Pluto conectado; hacen falta dos (entorno y secundario)")
    except ErrorRadio as e:
        ok = False
        print(f"   FALLO: {e}")
    try:
        from rtlsdr import RtlSdr
        series = RtlSdr.get_device_serial_addresses()
        print(f"   RTL-SDR conectados: {len(series)} (series {series})")
        ok = ok and bool(series)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"   FALLO: RTL-SDR: {e}")
    conf_ruta = AQUI / "radio.json"
    conf = json.loads(conf_ruta.read_text(encoding="utf-8")) if conf_ruta.exists() else {}
    for papel in ("entorno", "secundario"):
        print(f"   radio.json, Pluto del {papel}: serie «{conf.get(f'serie_pluto_{papel}', '')}» "
              f"uri «{conf.get(f'uri_pluto_{papel}', '')}»")
    return ok


def vigilar(control, segundos: float) -> dict:
    """Fracción de filas con energía ajena sobre el umbral (16 dB sobre el ruido) en cada canal, solo escuchando."""
    from rf_espectro import db
    control.silencio()
    control.medir_ruido()
    inicio = time.time()
    vistas, ocupadas, total = set(), [0] * P.N_CANALES, 0
    while time.time() - inicio < segundos:
        control.captura.esperar_filas(1)
        with control.captura.cond:
            nuevas = [f for f in control.captura.filas if f[0] >= inicio and f[0] not in vistas]
        for f in nuevas:
            vistas.add(f[0])
            total += 1
            for k in range(P.N_CANALES):
                if float(db(control.an.potencia_canal(f[2], k))) - control.ruido_db[k] > 16:
                    ocupadas[k] += 1
    return {"segundos": segundos, "filas": total, "ocupacion": [round(o / max(total, 1), 4) for o in ocupadas]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listar", action="store_true", help="solo bibliotecas y dispositivos")
    p.add_argument("--simulado", action="store_true", help="banco simulado, sin hardware")
    p.add_argument("--vigilar", type=float, default=0, help="segundos de escucha de la banda (ocupación ajena)")
    p.add_argument("--sin-ganancias", action="store_true", help="no reajusta las ganancias (usa las guardadas)")
    args = p.parse_args()

    print(f"Prueba de concepto con radio real | {platform.platform()} | Python {platform.python_version()}")
    titulo(1, "Plan de frecuencias y normativa")
    for linea in P.comprobar_plan(ANCHOS_OCUPADOS_HZ):
        print("   " + linea)

    if not args.simulado:
        titulo(2, "Bibliotecas y dispositivos")
        ok = bibliotecas()
        ok = dispositivos() and ok
        if args.listar or not ok:
            return 0 if ok else 2
    elif args.listar:
        return 0

    import rf_ejecucion as E
    conf = E.leer_conf()
    ruta = E.ruta_calibracion(args.simulado)
    previa = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
    try:
        control = E.abrir_radio(args.simulado, conf, previa)
    except ErrorRadio as e:
        print(f"\nFALLO al abrir la radio: {e}")
        return 2
    try:
        if args.vigilar:
            titulo(3, f"Ocupación ajena de la banda durante {args.vigilar:g} s (los Pluto en silencio)")
            v = vigilar(control, args.vigilar)
            for k, o in enumerate(v["ocupacion"]):
                print(f"   C{k} {P.CANALES_HZ[k] / 1e6:.3f} MHz: {100 * o:.2f} % de las filas sobre el umbral")
            return 0
        titulo(3, "Ruido de fondo, frecuencia, ganancias, firma y artefactos")
        cal = control.calibrar(E.objetivo_snr(conf), ganancias=not (args.sin_ganancias and previa))
        if args.sin_ganancias and previa:
            for papel in control.tx:
                for k in ("snr_primario_db", "firma_primario", "snr_interferencia_db", "firma_interferencia",
                          "margen_pilotos_db", "snr_db"):
                    if k in previa.get(papel, {}):
                        cal[papel].setdefault(k, previa[papel][k])
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
        r = cal["ruido"]["ruido_db"]
        print(f"   ruido por canal (dB respecto de la plena escala): {', '.join(f'{x:.1f}' for x in r)}")
        problemas = []
        for papel in ("entorno", "secundario"):
            f = cal[papel]["frecuencia"]
            print(f"   Pluto del {papel}: error de frecuencia {f['desfase_hz']:+.0f} Hz ({f['ppm']:+.2f} ppm), residuo "
                  f"{f['residuo_hz']} Hz; ganancia {cal[papel]['ganancia_db']:.1f} dB")
            if f["residuo_hz"] is None or abs(f["residuo_hz"]) > 60:
                problemas.append(f"la corrección de frecuencia del Pluto del {papel} no ha quedado bien")
        e = cal["entorno"]
        if "snr_primario_db" in e:
            print(f"   primario: SNR {e['snr_primario_db']} dB, firma {e['firma_primario']}, margen de las pilotos "
                  f"{e['margen_pilotos_db']} dB")
            print(f"   interferencia: SNR {e['snr_interferencia_db']} dB, firma {e['firma_interferencia']}")
            print(f"   secundario: SNR {cal['secundario']['snr_db']} dB")
            if not all(e["firma_primario"]):
                problemas.append("el detector no ve la firma del primario en algún canal")
            if any(e["firma_interferencia"]):
                problemas.append("el detector ve firma de primario en la interferencia")
            if min(e["snr_primario_db"]) < 22 or min(e["snr_interferencia_db"]) < 22:
                problemas.append("alguna emisión del entorno llega con menos de 22 dB de SNR (umbral: 16 dB)")
        a = cal["artefactos"]
        img = (f"{a['imagen_dbc']} dBc" if a["imagen_dbc"] is not None
               else f"por debajo del ruido (< {a['imagen_dbc_cota']} dBc)")
        print(f"   imagen del Pluto del entorno: {img}; fuga con los Pluto en silencio: {a['fuga_en_silencio_db']} dB; "
              f"pico del receptor en el peor caso: {a['pico_adc_peor_caso']} (satura a partir de 0,95)")
        if a["fuga_en_silencio_db"] > 1.0:
            problemas.append("con los Pluto en silencio queda señal en algún canal (fuga o emisión ajena)")
        if a["saturacion"]:
            problemas.append("el receptor satura en el peor caso: baja ganancia_rtl_db en radio.json")
        if a["imagen_dbc"] is not None and a["imagen_dbc"] > -25:
            problemas.append("la imagen del Pluto es alta; no cae en ningún canal, pero conviene revisarlo")
        print(f"\n   calibración guardada en {ruta} ({cal['duracion_s']:.0f} s)")
        if problemas:
            print("\nRevisar:\n  - " + "\n  - ".join(problemas))
            return 1
        print("\nTodo correcto: el montaje está listo para la demostración y la minicampaña.")
        return 0
    except ErrorRadio as e:
        print(f"\nFALLO: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130
    finally:
        control.cerrar()


if __name__ == "__main__":
    sys.exit(main())
