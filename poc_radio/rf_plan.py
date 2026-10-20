"""Plan de frecuencias de la prueba de concepto con radio real y comprobaciones de la normativa.

Banda: 434,040 a 434,790 MHz, dispositivos de corto alcance (SRD) no específicos, de uso común sin licencia
(Decisión 2006/771/CE de la Comisión, en la versión de la Decisión de Ejecución (UE) 2022/180; en España,
nota UN-158 del CNAF). Condiciones: 10 mW p.r.a. como máximo y ciclo de trabajo de hasta el 100 % si la
canalización es de 25 kHz o menos.

Hay seis canales de 25 kHz separados 100 kHz, uno por cada canal del simulador (0 a 4 autorizados y 5 en la
«banda protegida» del experimento). La banda protegida es solo un papel dentro del experimento: el canal 5 está
en la misma banda de uso común que los demás y se emite en él con la misma potencia.

Los dos Pluto usan el mismo oscilador local, desplazado 25 kHz del centro del plan, para que la imagen de cada
emisión (el residuo de la modulación en cuadratura, simétrico respecto del oscilador local) y la fuga del propio
oscilador caigan entre dos canales y nunca dentro de una ventana de medida.
"""
from __future__ import annotations

import math
from typing import List

# ---------------------------------------------------------------- banda y canales
BANDA_HZ = (434.040e6, 434.790e6)
POTENCIA_MAX_PRA_MW = 10.0
CANALIZACION_MAX_HZ = 25e3

N_CANALES = 6
CANALES_HZ = tuple(434.150e6 + 100e3 * k for k in range(N_CANALES))
ANCHO_CANAL_HZ = 25e3
CANAL_PROTEGIDO = 5                      # papel del canal 5 en el experimento (config.canales_autorizados = 0..4)

# ---------------------------------------------------------------- transmisores (ADALM-Pluto)
LO_PLUTO_HZ = 434.425e6                  # mismo oscilador local en los dos Pluto
FS_PLUTO = 1_000_000                     # muestras por segundo
GANANCIA_TX_MIN_DB = -89.0               # el Pluto admite de -89,75 a 0 dB; -89 se usa como «silencio»
GANANCIA_TX_MAX_DB = -10.0               # tope duro del programa, por debajo del límite legal con cualquier antena pequeña
POTENCIA_PLUTO_0DB_DBM = 7.0             # salida aproximada del Pluto a 0 dB con un tono a plena escala
GANANCIA_ANTENA_MAX_DBI = 3.0            # antena de látigo o de muelle de 433 MHz

# ---------------------------------------------------------------- receptor (RTL-SDR)
CENTRO_RTL_HZ = 434.400e6
FS_RTL = 1_024_000
N_FFT = 4096                             # 250 Hz por bin
FFT_POR_FILA = 8                         # una fila de la cascada cada 32 ms
FILAS_MEDIDA = 8                         # cada medida promedia 256 ms (64 transformadas)
FILAS_RUIDO = 31                         # el ruido de fondo se mide durante 1 s al empezar cada episodio
SEMIANCHO_MEDIDA_HZ = 6e3                # ventana de medida de la energía: ±6 kHz alrededor del centro del canal
SEMIANCHO_FRANJA_HZ = 12.5e3             # franja de cada canal en la cascada: el canal entero (25 kHz)
ASENTAMIENTO_S = 0.15                    # margen tras cambiar lo que se emite antes de dar una medida por válida

# ---------------------------------------------------------------- correspondencia con el simulador
RUIDO_REFERENCIA_DBM = -94.0             # media del ruido en el simulador: el ruido medido se hace corresponder con este valor
# energia_dbm = RUIDO_REFERENCIA_DBM + (potencia medida en la ventana - ruido de fondo del canal), en dB.
# Con el umbral del experimento (-78 dBm) un canal está ocupado si la medida supera al ruido en 16 dB.
POTENCIA_REFERENCIA_DBM = 15.0           # potencia por defecto del secundario en el simulador
# La potencia que pide una acción se traslada al Pluto del secundario como diferencia respecto de la de referencia:
# 25 dBm se emiten 10 dB por encima que 15 dBm, siempre por debajo de GANANCIA_TX_MAX_DB.
RAFAGA_MIN_S = 0.3
RAFAGA_MAX_S = 4.0


def desplazamientos_pluto() -> List[float]:
    return [f - LO_PLUTO_HZ for f in CANALES_HZ]


def desplazamientos_rtl() -> List[float]:
    return [f - CENTRO_RTL_HZ for f in CANALES_HZ]


def pra_estimada_dbm(ganancia_db: float, escala_pico: float = 1.0) -> float:
    """Potencia radiada aparente estimada (dBm) con la antena de mayor ganancia prevista."""
    return POTENCIA_PLUTO_0DB_DBM + ganancia_db + 20 * math.log10(max(escala_pico, 1e-9)) + GANANCIA_ANTENA_MAX_DBI - 2.15


def mw(dbm: float) -> float:
    return 10 ** (dbm / 10)


def comprobar_plan(anchos_ocupados_hz: dict) -> List[str]:
    """Comprueba el plan de frecuencias y devuelve una línea por comprobación. Lanza ValueError si falla alguna."""
    lineas, fallos = [], []

    def anotar(ok: bool, texto: str) -> None:
        (lineas if ok else fallos).append(("correcto: " if ok else "FALLO: ") + texto)

    lo, hi = BANDA_HZ
    for k, f in enumerate(CANALES_HZ):
        anotar(lo <= f - ANCHO_CANAL_HZ / 2 and f + ANCHO_CANAL_HZ / 2 <= hi,
               f"canal {k} ({f / 1e6:.3f} MHz, ±{ANCHO_CANAL_HZ / 2e3:g} kHz) dentro de la banda")
    for nombre, ancho in anchos_ocupados_hz.items():
        anotar(ancho <= ANCHO_CANAL_HZ and ancho <= CANALIZACION_MAX_HZ,
               f"anchura ocupada de «{nombre}» {ancho / 1e3:.1f} kHz <= {CANALIZACION_MAX_HZ / 1e3:g} kHz")
    # imágenes y fuga del oscilador local: lejos de las ventanas de medida
    for k, f in enumerate(CANALES_HZ):
        imagen = 2 * LO_PLUTO_HZ - f
        d = min(abs(imagen - g) for g in CANALES_HZ)
        anotar(d >= 3 * SEMIANCHO_MEDIDA_HZ, f"la imagen del canal {k} ({imagen / 1e6:.3f} MHz) queda a {d / 1e3:.0f} kHz "
                                              "del canal más cercano")
        anotar(lo <= imagen <= hi, f"la imagen del canal {k} queda dentro de la banda")
    d_lo = min(abs(LO_PLUTO_HZ - g) for g in CANALES_HZ)
    anotar(d_lo >= 3 * SEMIANCHO_MEDIDA_HZ, f"la fuga del oscilador local ({LO_PLUTO_HZ / 1e6:.3f} MHz) queda a "
                                             f"{d_lo / 1e3:.0f} kHz del canal más cercano")
    d_dc = min(abs(CENTRO_RTL_HZ - g) for g in CANALES_HZ)
    anotar(d_dc >= 3 * SEMIANCHO_MEDIDA_HZ, f"la componente continua del receptor ({CENTRO_RTL_HZ / 1e6:.3f} MHz) queda "
                                             f"a {d_dc / 1e3:.0f} kHz del canal más cercano")
    anotar(max(abs(x) for x in desplazamientos_pluto()) + ANCHO_CANAL_HZ / 2 <= 0.4 * FS_PLUTO,
           "todos los canales caben en la banda útil del Pluto (±40 % de la frecuencia de muestreo)")
    anotar(max(abs(x) for x in desplazamientos_rtl()) + ANCHO_CANAL_HZ / 2 <= 0.4 * FS_RTL,
           "todos los canales caben en la banda útil del RTL-SDR (±40 % de la frecuencia de muestreo)")
    p = pra_estimada_dbm(GANANCIA_TX_MAX_DB)
    anotar(mw(p) < POTENCIA_MAX_PRA_MW,
           f"con el tope de ganancia ({GANANCIA_TX_MAX_DB:g} dB) la p.r.a. estimada es {p:.1f} dBm ({mw(p):.2f} mW), "
           f"por debajo de {POTENCIA_MAX_PRA_MW:g} mW")
    if fallos:
        raise ValueError("El plan de frecuencias no cumple:\n  " + "\n  ".join(fallos))
    return lineas
