"""Análisis espectral de las muestras del RTL-SDR: potencia por canal, firma de primario y frecuencia de un tono.

La densidad espectral se estima promediando transformadas de 4096 puntos con ventana de Blackman (250 Hz por
bin). Se normaliza de forma que la suma de todos los bins es la potencia media de la señal (unidades de plena
escala del receptor). La potencia de un canal es la suma de los bins de su ventana de medida (±6 kHz).

La firma de primario se busca como dos líneas espectrales, una en cada posición de las pilotos (±4 kHz del centro
del canal). Cada piloto cuenta como presente si su pico supera en 10 dB a la media de los bins vecinos (entre 1,5
y 2,5 kHz a cada lado; se toma el lado más alto) y en 12 dB al ruido de fondo del bin. La interferencia MSK
reparte su potencia sin líneas: en la posición de las pilotos su espectro queda por debajo del de sus vecinos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from rf_ondas import PILOTO_HZ
from rf_plan import (CANALES_HZ, CENTRO_RTL_HZ, FS_RTL, N_FFT, RUIDO_REFERENCIA_DBM, SEMIANCHO_FRANJA_HZ,
                     SEMIANCHO_MEDIDA_HZ)

UMBRAL_PILOTO_VECINOS_DB = 10.0
UMBRAL_PILOTO_RUIDO_DB = 12.0
TOLERANCIA_PILOTO_BINS = 3          # ±750 Hz de error de frecuencia residual
VECINOS_BINS = (6, 10)              # de 1,5 a 2,5 kHz del piloto


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


@dataclass
class Medida:
    canal: int
    potencia_db: float          # potencia en la ventana de medida, dB respecto de la plena escala
    ruido_db: float             # ruido de fondo del canal en la misma ventana
    snr_db: float
    energia_dbm: float          # valor que recibe el sistema evaluado (escala del simulador)
    firma: bool
    piloto_inf_db: float        # margen del piloto inferior sobre sus vecinos
    piloto_sup_db: float
    piloto_snr_db: float        # el menor de los dos márgenes sobre el ruido
    t_inicio: float
    t_fin: float
    n_filas: int

    def a_dict(self) -> Dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4 if k.startswith("t_") else 2)
        return d


class Analizador:
    def __init__(self, fs: float = FS_RTL, n_fft: int = N_FFT, centro_hz: float = CENTRO_RTL_HZ,
                 canales_hz: Sequence[float] = CANALES_HZ):
        self.fs, self.n_fft, self.centro_hz = fs, n_fft, centro_hz
        self.ventana = np.blackman(n_fft).astype(np.float32)
        self.norma = n_fft * float(np.sum(self.ventana.astype(np.float64) ** 2))
        self.df = fs / n_fft
        self.freqs = (np.arange(n_fft) - n_fft // 2) * self.df          # frecuencia de cada bin tras fftshift
        self.desplazamientos = [f - centro_hz for f in canales_hz]
        self.medida: List[np.ndarray] = []
        self.franja: List[np.ndarray] = []
        self.pilotos: List[List[Dict]] = []
        n_franja = int(round(SEMIANCHO_FRANJA_HZ / self.df))
        for d in self.desplazamientos:
            c = self.bin(d)
            self.medida.append(np.where(np.abs(self.freqs - self.freqs[c]) <= SEMIANCHO_MEDIDA_HZ)[0])
            self.franja.append(np.arange(c - n_franja, c + n_franja + 1))
            ps = []
            for signo in (-1, 1):
                p = self.bin(d + signo * PILOTO_HZ)
                pico = np.arange(p - TOLERANCIA_PILOTO_BINS, p + TOLERANCIA_PILOTO_BINS + 1)
                v0, v1 = VECINOS_BINS
                ps.append({"pico": pico, "izq": np.arange(p - v1, p - v0 + 1), "der": np.arange(p + v0, p + v1 + 1)})
            self.pilotos.append(ps)
        self.ancho_franja = 2 * n_franja + 1

    def bin(self, desplazamiento_hz: float) -> int:
        return int(round(desplazamiento_hz / self.df)) + self.n_fft // 2

    # ------------------------------------------------------------ densidad espectral
    def psd(self, iq: np.ndarray) -> np.ndarray:
        """Promedio de |FFT|² de los bloques de n_fft muestras; suma de bins = potencia media."""
        n = (len(iq) // self.n_fft) * self.n_fft
        if n == 0:
            raise ValueError("faltan muestras para una transformada")
        bloques = np.asarray(iq[:n], dtype=np.complex64).reshape(-1, self.n_fft) * self.ventana
        X = np.fft.fftshift(np.fft.fft(bloques, axis=1), axes=1)
        return ((X.real ** 2 + X.imag ** 2).mean(axis=0) / self.norma).astype(np.float64)

    def franjas_db(self, psd: np.ndarray) -> np.ndarray:
        """Los seis canales de la cascada, uno por fila, en dB."""
        return np.stack([db(psd[idx]) for idx in self.franja])

    # ------------------------------------------------------------ medidas por canal
    def potencia_canal(self, psd: np.ndarray, canal: int) -> float:
        return float(psd[self.medida[canal]].sum())

    def ruido_por_canal(self, psd: np.ndarray) -> List[float]:
        return [float(db(self.potencia_canal(psd, k))) for k in range(len(self.medida))]

    def indices(self, desplazamiento_hz: float, semiancho_hz: float = SEMIANCHO_MEDIDA_HZ) -> np.ndarray:
        return np.where(np.abs(self.freqs - desplazamiento_hz) <= semiancho_hz)[0]

    def potencia_en(self, psd: np.ndarray, desplazamiento_hz: float, semiancho_hz: float = SEMIANCHO_MEDIDA_HZ) -> float:
        """Potencia (dB) en una ventana cualquiera, por ejemplo donde cae la imagen de un Pluto."""
        return float(db(psd[self.indices(desplazamiento_hz, semiancho_hz)].sum()))

    def margenes_pilotos(self, psd: np.ndarray, canal: int, piso_bin_db: np.ndarray):
        """(margen sobre vecinos, margen sobre ruido) de cada piloto, en dB."""
        out = []
        for p in self.pilotos[canal]:
            pico = float(db(psd[p["pico"]].max()))
            vecinos = max(float(db(psd[p["izq"]].mean())), float(db(psd[p["der"]].mean())))
            ruido = float(np.median(piso_bin_db[p["pico"]]))
            out.append((pico - vecinos, pico - ruido))
        return out

    def medir(self, psd: np.ndarray, canal: int, ruido_canal_db: float, piso_bin_db: np.ndarray,
              t_inicio: float = 0.0, t_fin: float = 0.0, n_filas: int = 0) -> Medida:
        pot = float(db(self.potencia_canal(psd, canal)))
        snr = pot - ruido_canal_db
        (vi, ri), (vs, rs) = self.margenes_pilotos(psd, canal, piso_bin_db)
        firma = (min(vi, vs) >= UMBRAL_PILOTO_VECINOS_DB) and (min(ri, rs) >= UMBRAL_PILOTO_RUIDO_DB)
        return Medida(canal=canal, potencia_db=pot, ruido_db=ruido_canal_db, snr_db=snr,
                      energia_dbm=RUIDO_REFERENCIA_DBM + snr, firma=bool(firma), piloto_inf_db=vi, piloto_sup_db=vs,
                      piloto_snr_db=min(ri, rs), t_inicio=t_inicio, t_fin=t_fin, n_filas=n_filas)

    # ------------------------------------------------------------ calibración de frecuencia
    def frecuencia_pico(self, psd: np.ndarray, esperado_hz: float, busqueda_hz: float = 40e3) -> Optional[Dict]:
        """Frecuencia (respecto del centro del receptor) del tono más fuerte a menos de busqueda_hz de esperado_hz.

        Interpolación parabólica del pico en dB: con la ventana de Blackman y un tono muy por encima del ruido, el
        error queda en unos pocos hercios, muy por debajo de los 250 Hz de un bin.
        """
        sel = np.where(np.abs(self.freqs - esperado_hz) <= busqueda_hz)[0]
        if len(sel) < 5:
            return None
        i = int(sel[np.argmax(psd[sel])])
        if i <= 0 or i >= self.n_fft - 1:
            return None
        a, b, c = (float(db(psd[j])) for j in (i - 1, i, i + 1))
        den = a - 2 * b + c
        delta = 0.5 * (a - c) / den if den != 0 else 0.0
        fondo = float(np.median(db(psd[sel])))
        return {"frecuencia_hz": float(self.freqs[i] + delta * self.df), "pico_sobre_fondo_db": b - fondo}
