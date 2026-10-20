"""Acceso a los dispositivos: dos ADALM-Pluto (transmisores) y un RTL-SDR (receptor), o su versión simulada.

Las dos versiones tienen la misma interfaz, de modo que todo lo demás (medidas, calibración, episodios y
cascada) se puede probar sin hardware:

  transmisor.emitir(senal, ganancia_db)   empieza a emitir en bucle la señal (búfer cíclico)
  transmisor.silenciar()                  deja de emitir y baja la ganancia al mínimo
  receptor.leer(n)                        devuelve n muestras complejas (escala ±1)

Los Pluto se abren con pyadi-iio y el RTL-SDR con pyrtlsdr; las dos bibliotecas solo se importan si se usan los
dispositivos reales. Con dos Pluto en el mismo ordenador, los dos tienen por defecto la misma dirección IP
(192.168.2.1): por eso se abren por USB («usb:x.y.z») y se distinguen por su número de serie.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from rf_ondas import Senal
from rf_plan import FS_PLUTO, FS_RTL, GANANCIA_TX_MAX_DB, GANANCIA_TX_MIN_DB, LO_PLUTO_HZ


class ErrorRadio(Exception):
    pass


_BUSQUEDA_PREPARADA = False


def preparar_bibliotecas() -> None:
    """En macOS, ayuda a pylibiio y a pyrtlsdr a encontrar libiio y librtlsdr.

    Las dos usan ctypes.util.find_library, que en macOS no mira en /opt/homebrew/lib (Homebrew en Apple Silicon)
    ni en el entorno de conda. Si la búsqueda normal falla, se prueba en esas carpetas. Hay que llamarla antes de
    importar iio, adi o rtlsdr.
    """
    global _BUSQUEDA_PREPARADA
    if _BUSQUEDA_PREPARADA or sys.platform != "darwin":
        return
    import ctypes.util
    original = ctypes.util.find_library
    carpetas = [os.environ.get("CONDA_PREFIX", ""), "/opt/homebrew", "/usr/local"]
    nombres = {"iio": "libiio.dylib", "rtlsdr": "librtlsdr.dylib", "librtlsdr": "librtlsdr.dylib"}

    def buscar(nombre):
        r = original(nombre)
        if r is None and nombre in nombres:
            for c in carpetas:
                p = Path(c) / "lib" / nombres[nombre]
                if c and p.exists():
                    return str(p)
            if nombre == "iio" and Path("/Library/Frameworks/iio.framework/iio").exists():
                return "/Library/Frameworks/iio.framework/iio"
        return r

    ctypes.util.find_library = buscar
    _BUSQUEDA_PREPARADA = True


def limitar_ganancia(g: float) -> float:
    return float(min(GANANCIA_TX_MAX_DB, max(GANANCIA_TX_MIN_DB, g)))


# ====================================================================== dispositivos reales
def listar_pluto() -> Dict[str, str]:
    """{uri: descripción} de los Pluto conectados por USB (la descripción incluye el número de serie)."""
    preparar_bibliotecas()
    try:
        import iio  # pylibiio, se instala con pyadi-iio
        contextos = iio.scan_contexts()
        return {u: d for u, d in contextos.items() if u.startswith("usb:")}
    except Exception:  # noqa: BLE001 (sin pylibiio o sin soporte de escaneo: se prueba con iio_info)
        pass
    if not shutil.which("iio_info"):
        raise ErrorRadio("No se encuentra libiio (ni el módulo de Python iio ni el programa iio_info). "
                         "Revisa la instalación en poc_radio/README.md, paso 2.")
    salida = subprocess.run(["iio_info", "-s"], capture_output=True, text=True, timeout=20).stdout
    out = {}
    for linea in salida.splitlines():
        m = re.search(r"\[(usb:[^\]]+)\]", linea)
        if m:
            out[m.group(1)] = linea.strip()
    return out


def serie_de(descripcion: str) -> Optional[str]:
    m = re.search(r"serial=([0-9a-fA-F]+)", descripcion)
    return m.group(1) if m else None


def resolver_uri(conf: Dict, papel: str) -> str:
    """URI del Pluto de un papel («entorno» o «secundario») según radio.json: uri fija o número de serie."""
    uri = conf.get(f"uri_pluto_{papel}") or ""
    if uri:
        return uri
    serie = (conf.get(f"serie_pluto_{papel}") or "").lower()
    plutos = listar_pluto()
    if not plutos:
        raise ErrorRadio("No se ve ningún Pluto por USB. Comprueba el cable (el puerto de datos es el del centro) "
                         "y ejecuta «iio_info -s».")
    if not serie:
        lista = "\n".join(f"  {u}  serie {serie_de(d)}" for u, d in plutos.items())
        raise ErrorRadio(f"Falta el número de serie del Pluto del {papel} en poc_radio/radio.json. Conectados:\n{lista}")
    for u, d in plutos.items():
        s = (serie_de(d) or "").lower()
        if s and (s == serie or s.endswith(serie)):
            return u
    raise ErrorRadio(f"No está conectado el Pluto del {papel} (serie {serie}). Conectados: {list(plutos.values())}")


class TxPluto:
    def __init__(self, uri: str, nombre: str, lo_hz: float = LO_PLUTO_HZ, fs: float = FS_PLUTO):
        preparar_bibliotecas()
        try:
            import adi  # pyadi-iio
        except ImportError as e:
            raise ErrorRadio(f"Falta pyadi-iio ({e}). Revisa poc_radio/README.md, paso 2.") from None
        self.nombre, self.uri, self.fs, self.lo_hz = nombre, uri, fs, lo_hz
        try:
            self.sdr = adi.Pluto(uri)
        except Exception as e:  # noqa: BLE001
            raise ErrorRadio(f"No se puede abrir el Pluto del {nombre} en {uri}: {e}") from None
        self.sdr.tx_hardwaregain_chan0 = GANANCIA_TX_MIN_DB     # lo primero: que no salga nada mientras se configura
        self.sdr.sample_rate = int(fs)
        self.sdr.tx_rf_bandwidth = int(fs)
        self.sdr.tx_lo = int(round(lo_hz))
        self.sdr.tx_enabled_channels = [0]
        self.sdr.tx_cyclic_buffer = True
        self.activo = False
        self.ganancia_db = GANANCIA_TX_MIN_DB
        self.senal: Optional[Senal] = None
        try:
            ctx = getattr(self.sdr, "ctx", None) or getattr(self.sdr, "_ctx", None)
            self.serie = ctx.attrs.get("hw_serial", "") if ctx is not None else ""
        except Exception:  # noqa: BLE001
            self.serie = ""

    def _destruir_bufer(self) -> None:
        # pyadi-iio no deja enviar otro búfer cíclico sin destruir antes el anterior
        if self.activo or getattr(self.sdr, "_txbuf", None):
            self.sdr.tx_destroy_buffer()
        self.activo = False

    def emitir(self, senal: Senal, ganancia_db: float) -> None:
        self._destruir_bufer()
        self.ganancia_db = limitar_ganancia(ganancia_db)
        self.sdr.tx_hardwaregain_chan0 = self.ganancia_db
        self.sdr.tx(senal.bufer(self.fs) * (2 ** 14))      # pyadi-iio espera muestras entre -2^14 y 2^14
        self.activo, self.senal = True, senal

    def silenciar(self) -> None:
        self._destruir_bufer()
        self.sdr.tx_hardwaregain_chan0 = GANANCIA_TX_MIN_DB
        self.ganancia_db, self.senal = GANANCIA_TX_MIN_DB, None

    def info(self) -> Dict:
        return {"nombre": self.nombre, "uri": self.uri, "serie": self.serie, "lo_hz": self.lo_hz, "fs": self.fs}

    def cerrar(self) -> None:
        try:
            self.silenciar()
        finally:
            self.sdr = None


class RxRtl:
    def __init__(self, centro_hz: float, fs: float = FS_RTL, ganancia_db: float = 28.0, ppm: int = 0,
                 serie: str = "", indice: int = 0):
        preparar_bibliotecas()
        try:
            from rtlsdr import RtlSdr  # pyrtlsdr
        except Exception as e:  # noqa: BLE001 (ImportError o fallo al cargar librtlsdr)
            raise ErrorRadio(f"No se puede cargar pyrtlsdr o librtlsdr ({e}). Revisa poc_radio/README.md, paso 2.") from None
        try:
            if serie:
                indice = RtlSdr.get_device_index_by_serial(serie)
            self.sdr = RtlSdr(device_index=indice)
        except Exception as e:  # noqa: BLE001
            raise ErrorRadio(f"No se puede abrir el RTL-SDR ({e}). ¿Está conectado y libre (sin otro programa usándolo)?") from None
        self.sdr.sample_rate = fs
        self.sdr.center_freq = centro_hz
        if ppm:
            self.sdr.freq_correction = int(ppm)     # pyrtlsdr da error si se fija 0 sin cambio
        self.sdr.gain = ganancia_db                 # ganancia manual: sin control automático
        self.fs, self.centro_hz, self.ppm = fs, centro_hz, ppm
        self.ganancia_db = float(self.sdr.gain)
        self.serie = serie

    def leer(self, n: int) -> np.ndarray:
        return np.asarray(self.sdr.read_samples(n), dtype=np.complex64)

    def info(self) -> Dict:
        return {"centro_hz": self.centro_hz, "fs": self.fs, "ganancia_db": self.ganancia_db, "ppm": self.ppm,
                "serie": self.serie}

    def cerrar(self) -> None:
        try:
            self.sdr.close()
        except Exception:  # noqa: BLE001
            pass


# ====================================================================== banco simulado
class BancoSimulado:
    """Enlace radioeléctrico simulado entre los dos Pluto y el RTL-SDR.

    Reproduce lo que importa para probar la cadena: el error de frecuencia de cada Pluto (en ppm), la imagen y la
    fuga del oscilador local de cada uno, la pérdida del trayecto, el ruido térmico del receptor, la
    cuantificación a 8 bits y la saturación del RTL-SDR. Las muestras se entregan a ritmo de tiempo real.
    """

    def __init__(self, semilla: int = 0, ruido_dbfs: float = -40.0, enlace_db: Optional[Dict[str, float]] = None,
                 ppm: Optional[Dict[str, float]] = None, imagen_dbc: float = -42.0, fuga_lo_dbc: float = -48.0,
                 tiempo_real: bool = True):
        self.rng = np.random.default_rng(semilla)
        self.ruido_dbfs = ruido_dbfs
        self.enlace_db = enlace_db or {"entorno": 38.0, "secundario": 34.0}
        self.ppm = ppm or {"entorno": 11.3, "secundario": -8.1}
        self.imagen_dbc, self.fuga_lo_dbc = imagen_dbc, fuga_lo_dbc
        self.tiempo_real = tiempo_real
        self.transmisores: Dict[str, "TxSimulado"] = {}
        self.lock = threading.Lock()

    def transmisor(self, nombre: str, lo_hz: float = LO_PLUTO_HZ, fs: float = FS_PLUTO) -> "TxSimulado":
        tx = TxSimulado(self, nombre, lo_hz, fs)
        self.transmisores[nombre] = tx
        return tx

    def receptor(self, centro_hz: float, fs: float = FS_RTL, ganancia_db: float = 28.0) -> "RxSimulado":
        return RxSimulado(self, centro_hz, fs, ganancia_db)


class TxSimulado:
    def __init__(self, banco: BancoSimulado, nombre: str, lo_hz: float, fs: float):
        self.banco, self.nombre, self.lo_hz, self.fs = banco, nombre, lo_hz, fs
        self.senal: Optional[Senal] = None
        self.ganancia_db = GANANCIA_TX_MIN_DB
        self.activo = False
        self.fase = float(banco.rng.uniform(0, 2 * np.pi))
        self.serie = f"simulado-{nombre}"
        self.uri = "simulado"

    def emitir(self, senal: Senal, ganancia_db: float) -> None:
        senal.bufer(self.fs)                      # se genera igual que para el Pluto, para probar el búfer
        with self.banco.lock:
            self.senal, self.ganancia_db, self.activo = senal, limitar_ganancia(ganancia_db), True

    def silenciar(self) -> None:
        with self.banco.lock:
            self.senal, self.ganancia_db, self.activo = None, GANANCIA_TX_MIN_DB, False

    def info(self) -> Dict:
        return {"nombre": self.nombre, "uri": "simulado", "serie": self.serie, "lo_hz": self.lo_hz, "fs": self.fs,
                "ppm_simulado": self.banco.ppm.get(self.nombre, 0.0)}

    def cerrar(self) -> None:
        self.silenciar()


class RxSimulado:
    def __init__(self, banco: BancoSimulado, centro_hz: float, fs: float, ganancia_db: float):
        self.banco, self.centro_hz, self.fs, self.ganancia_db = banco, centro_hz, fs, ganancia_db
        self.n0 = 0
        self.t_inicio = time.time()
        self.ppm = 0
        self.serie = "simulado"

    def leer(self, n: int) -> np.ndarray:
        b = self.banco
        t = (self.n0 + np.arange(n)) / self.fs
        self.n0 += n
        sigma = math.sqrt(10 ** (b.ruido_dbfs / 10) / 2)
        x = sigma * (b.rng.standard_normal(n) + 1j * b.rng.standard_normal(n))
        with b.lock:
            activos = [(tx, tx.senal, tx.ganancia_db) for tx in b.transmisores.values() if tx.activo and tx.senal]
        for tx, senal, g in activos:
            ppm = b.ppm.get(tx.nombre, 0.0)
            f_lo = tx.lo_hz * (1 + ppm * 1e-6) - self.centro_hz       # oscilador local real, visto desde el receptor
            amp = 10 ** ((g + b.enlace_db.get(tx.nombre, 30.0)) / 20)
            s = senal.evaluar(t * (1 + ppm * 1e-6))                   # el reloj del Pluto también se desvía
            portadora = np.exp(1j * (2 * np.pi * f_lo * t + tx.fase))
            x += amp * portadora * (s + 10 ** (b.imagen_dbc / 20) * np.conj(s) + 10 ** (b.fuga_lo_dbc / 20))
        # RTL-SDR: 8 bits y saturación a plena escala
        x = np.clip(np.round(x.real * 127.5) / 127.5, -1, 1) + 1j * np.clip(np.round(x.imag * 127.5) / 127.5, -1, 1)
        if b.tiempo_real:
            espera = self.t_inicio + self.n0 / self.fs - time.time()
            if espera > 0:
                time.sleep(espera)
        return x.astype(np.complex64)

    def info(self) -> Dict:
        return {"centro_hz": self.centro_hz, "fs": self.fs, "ganancia_db": self.ganancia_db, "ppm": 0,
                "serie": "simulado", "ruido_dbfs": self.banco.ruido_dbfs}

    def cerrar(self) -> None:
        pass
