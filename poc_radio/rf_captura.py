"""Captura continua del RTL-SDR en un hilo propio.

El receptor solo lo puede abrir un proceso, así que un único hilo lee las muestras y calcula una fila de
densidad espectral cada 32 ms. De esas filas salen las tres cosas que se necesitan a la vez:

  - las medidas que piden las herramientas del sistema evaluado (promedio de las últimas filas válidas);
  - la cascada en directo (las seis franjas de canal de cada fila);
  - la grabación de cada episodio, con los eventos (cambios de paso, medidas, ráfagas, decisiones del escudo)
    marcados en el tiempo, para dibujar después las figuras.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from rf_dispositivos import ErrorRadio
from rf_espectro import Analizador
from rf_plan import FFT_POR_FILA, FILAS_MEDIDA


class Captura(threading.Thread):
    def __init__(self, receptor, analizador: Analizador, fft_por_fila: int = FFT_POR_FILA,
                 memoria_filas: int = 400, vista_s: float = 40.0):
        super().__init__(daemon=True, name="captura-rtl")
        self.rx, self.an = receptor, analizador
        self.n = fft_por_fila * analizador.n_fft
        self.dt_fila = self.n / receptor.fs
        self.filas: deque = deque(maxlen=memoria_filas)                    # (t_ini, t_fin, psd, pico)
        self.vista: deque = deque(maxlen=int(vista_s / self.dt_fila) + 1)  # (t_ini, franjas) para la cascada
        self.cond = threading.Condition()
        self._parar = threading.Event()
        self.error: Optional[BaseException] = None
        self.grabando = False
        self._grab_t: List[float] = []
        self._grab_f: List[np.ndarray] = []
        self.eventos: List[Tuple[float, str, Dict]] = []
        self.contador = 0

    # ------------------------------------------------------------ hilo
    def run(self) -> None:
        try:
            while not self._parar.is_set():
                iq = self.rx.leer(self.n)
                t_fin = time.time()
                t_ini = t_fin - self.dt_fila
                psd = self.an.psd(iq).astype(np.float32)
                pico = float(max(np.abs(iq.real).max(), np.abs(iq.imag).max()))
                franjas = self.an.franjas_db(psd).astype(np.float16)
                with self.cond:
                    self.filas.append((t_ini, t_fin, psd, pico))
                    self.vista.append((t_ini, franjas))
                    if self.grabando:
                        self._grab_t.append(t_ini)
                        self._grab_f.append(franjas)
                    self.contador += 1
                    self.cond.notify_all()
        except BaseException as e:  # noqa: BLE001 (se informa al hilo principal en la siguiente medida)
            self.error = e
            with self.cond:
                self.cond.notify_all()

    def parar(self) -> None:
        self._parar.set()
        self.join(timeout=3)

    def _comprobar(self) -> None:
        if self.error is not None:
            raise ErrorRadio(f"La captura del RTL-SDR se ha detenido: {self.error!r}")
        if not self.is_alive():
            raise ErrorRadio("La captura del RTL-SDR no está en marcha")

    # ------------------------------------------------------------ medidas
    def filas_desde(self, desde: float, n_filas: int, timeout: float = 6.0) -> list:
        """Las n_filas más recientes que empiezan después de «desde», como (t_ini, t_fin, psd, pico). Espera si faltan."""
        limite = time.time() + max(timeout, n_filas * self.dt_fila + 2.0)
        with self.cond:
            while True:
                self._comprobar()
                validas = [f for f in self.filas if f[0] >= desde]
                if len(validas) >= n_filas:
                    return validas[-n_filas:]
                resto = limite - time.time()
                if resto <= 0:
                    raise ErrorRadio("El RTL-SDR no entrega muestras a tiempo")
                self.cond.wait(min(resto, 0.5))

    def medir(self, desde: float, n_filas: int = FILAS_MEDIDA, timeout: float = 6.0):
        """Promedio de las n_filas más recientes que empiezan después de «desde». Espera si aún no las hay."""
        sel = self.filas_desde(desde, n_filas, timeout)
        psd = np.mean(np.stack([f[2] for f in sel]).astype(np.float64), axis=0)
        return psd, sel[0][0], sel[-1][1], len(sel), max(f[3] for f in sel)

    def medir_entre(self, desde: float, hasta: float, minimo: int = 2):
        """Promedio de las filas enteras entre dos instantes (por ejemplo, durante una ráfaga)."""
        with self.cond:
            self._comprobar()
            sel = [f for f in self.filas if f[0] >= desde and f[1] <= hasta]
        if len(sel) < minimo:
            return None
        psd = np.mean(np.stack([f[2] for f in sel]).astype(np.float64), axis=0)
        return psd, sel[0][0], sel[-1][1], len(sel), max(f[3] for f in sel)

    def esperar_filas(self, n: int, timeout: float = 5.0) -> None:
        with self.cond:
            objetivo = self.contador + n
            limite = time.time() + timeout
            while self.contador < objetivo:
                self._comprobar()
                resto = limite - time.time()
                if resto <= 0:
                    raise ErrorRadio("El RTL-SDR no entrega muestras a tiempo")
                self.cond.wait(min(resto, 0.5))

    # ------------------------------------------------------------ eventos y grabación
    def evento(self, tipo: str, **datos) -> None:
        with self.cond:
            self.eventos.append((time.time(), tipo, datos))

    def empezar_grabacion(self) -> float:
        with self.cond:
            self._grab_t, self._grab_f, self.eventos = [], [], []
            self.grabando = True
        return time.time()

    def terminar_grabacion(self, decimar: int = 1) -> Dict:
        with self.cond:
            self.grabando = False
            t = np.array(self._grab_t, dtype=np.float64)
            f = np.stack(self._grab_f) if self._grab_f else np.zeros((0, 6, self.an.ancho_franja), dtype=np.float16)
            eventos = list(self.eventos)
            self._grab_t, self._grab_f = [], []
        if decimar > 1 and len(t) >= decimar:
            m = (len(t) // decimar) * decimar
            lin = 10 ** (f[:m].astype(np.float32) / 10)
            f = (10 * np.log10(lin.reshape(-1, decimar, *f.shape[1:]).mean(axis=1))).astype(np.float16)
            t = t[:m:decimar]
        return {"t": t, "franjas": f, "eventos": eventos, "dt_fila": self.dt_fila * max(1, decimar)}

    def ultimas_franjas(self, segundos: float):
        with self.cond:
            if not self.vista:
                return None, None
            t_ult = self.vista[-1][0]
            sel = [(t, fr) for t, fr in self.vista if t >= t_ult - segundos]
        return np.array([t for t, _ in sel]), np.stack([fr for _, fr in sel])
