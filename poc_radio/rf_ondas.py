"""Formas de onda de la prueba de concepto, en banda base compleja.

Tres señales, todas de banda estrecha (menos de 12,5 kHz ocupados, dentro de un canal de 25 kHz):

  - primario (y su emulación en el ataque PUE): datos MSK a 2400 baudios más dos portadoras piloto a ±4 kHz del
    centro del canal. Las pilotos son la «firma» que busca el detector de características, como las portadoras
    piloto que se usan para reconocer señales de televisión digital en radio cognitiva. El emulador emite la misma
    forma de onda: por eso la firma no distingue al primario legítimo de su emulación.
  - interferencia: datos MSK a 4800 baudios sin pilotos. Es otra emisión de datos legal, en nuestro propio canal y
    con la misma potencia que las demás: emula la interferencia sobre el enlace del secundario, no es un
    inhibidor de frecuencia (prohibidos en España por el artículo 106.9 de la Ley 11/2022).
  - secundario: barrido lineal de ±5 kHz que se repite cada 0,25 s. En la cascada se ve como una diagonal en
    diente de sierra, distinta de las otras dos, y deja ver de un vistazo dónde transmite el sistema evaluado.

Cada señal es una función del tiempo absoluto. El mismo código rellena el búfer cíclico de un Pluto (tiempos
0, 1/fs, ..., T-1/fs) y genera lo que «recibe» el receptor simulado en cualquier instante; para que el búfer
cíclico no tenga saltos al repetirse, los datos MSK cierran la fase al final del periodo y todas las frecuencias
son múltiplos de 1/T.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

T_BUF_ENTORNO = 0.5        # s, periodo del búfer cíclico del Pluto del entorno
T_BUF_SECUNDARIO = 0.25    # s, periodo del búfer del secundario (un barrido)
RB_PRIMARIO = 2400         # baudios
RB_INTERFERENCIA = 4800    # baudios
PILOTO_HZ = 4e3
AMP_PILOTO = 0.5           # amplitud de cada piloto respecto de los datos
BARRIDO_HZ = 10e3          # el secundario barre de -5 a +5 kHz
PERIODO_BARRIDO_S = 0.25

# Potencia media (respecto de la plena escala) que se fija para todas las emisiones del entorno, para que el
# detector de energía vea el mismo nivel tanto si emite un primario como si emite la interferencia.
POTENCIA_ENTORNO = 0.375   # potencia del primario sin escalar: (1 + 2·0,5²)/4
AMP_ENTORNO = 0.18         # escala de cada emisión del entorno en el búfer (hasta 5 a la vez sin saturar)
AMP_SECUNDARIO = 0.7

# anchura ocupada (aprox. 99 % de la potencia), para la comprobación del plan
ANCHOS_OCUPADOS_HZ = {
    "primario": 2 * PILOTO_HZ + 2 * 250.0,       # pilotos a ±4 kHz (los datos ocupan unos ±1,5 kHz)
    "interferencia": 1.2 * RB_INTERFERENCIA,
    "secundario": BARRIDO_HZ + 2 * 250.0,
}
TIPOS = ("primario", "emulador", "interferencia", "secundario", "tono")


@lru_cache(maxsize=64)
def _patron(n_simbolos: int, semilla: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bits ±1 de un periodo y su suma acumulada; la suma total es múltiplo de 4 (fase cerrada en MSK)."""
    if n_simbolos % 2:
        raise ValueError("el periodo MSK debe tener un número par de símbolos")
    rng = np.random.default_rng(semilla)
    b = rng.choice(np.array([-1, 1], dtype=np.int64), size=n_simbolos)
    if int(b.sum()) % 4:
        b[-1] = -b[-1]          # con un número par de bits, la suma solo puede quedar en 2 (mod 4)
    prefijo = np.concatenate(([0], np.cumsum(b)[:-1]))
    return b, prefijo


def msk(t: np.ndarray, rb: float, periodo_s: float, semilla: int) -> np.ndarray:
    """MSK de envolvente constante: la fase avanza ±pi/2 en cada símbolo (desviación de ±rb/4)."""
    n = int(round(rb * periodo_s))
    bits, prefijo = _patron(n, semilla)
    x = t * rb
    k = np.floor(x).astype(np.int64)
    frac = x - k
    km = k % n
    fase = (np.pi / 2) * (prefijo[km] + bits[km] * frac)
    return np.exp(1j * fase)


def primario(t: np.ndarray, semilla: int) -> np.ndarray:
    datos = msk(t, RB_PRIMARIO, T_BUF_ENTORNO, semilla)
    pilotos = AMP_PILOTO * (np.exp(2j * np.pi * PILOTO_HZ * t) + np.exp(-2j * np.pi * PILOTO_HZ * t + 1j * np.pi / 3))
    return (datos + pilotos) / 2.0                       # pico <= 1, potencia media 0,375


def interferencia(t: np.ndarray, semilla: int) -> np.ndarray:
    return np.sqrt(POTENCIA_ENTORNO) * msk(t, RB_INTERFERENCIA, T_BUF_ENTORNO, semilla)


def secundario(t: np.ndarray) -> np.ndarray:
    tau = np.mod(t, PERIODO_BARRIDO_S)
    fase = 2 * np.pi * (-BARRIDO_HZ / 2 * tau + BARRIDO_HZ * tau ** 2 / (2 * PERIODO_BARRIDO_S))
    return np.exp(1j * fase)                             # la fase vuelve a 0 al final de cada barrido


def onda(tipo: str, t: np.ndarray, semilla: int = 0) -> np.ndarray:
    if tipo in ("primario", "emulador"):
        return primario(t, semilla)
    if tipo == "interferencia":
        return interferencia(t, semilla)
    if tipo == "secundario":
        return secundario(t)
    if tipo == "tono":
        return np.ones_like(t, dtype=np.complex128)
    raise ValueError(f"tipo de señal desconocido: {tipo}")


@dataclass(frozen=True)
class Componente:
    tipo: str
    desplazamiento_hz: float     # respecto del oscilador local del transmisor
    amplitud: float
    semilla: int = 0


@dataclass
class Senal:
    """Suma de componentes que emite un transmisor. Se convierte en búfer cíclico o se evalúa en cualquier instante."""
    componentes: List[Componente] = field(default_factory=list)
    periodo_s: float = T_BUF_ENTORNO

    def __post_init__(self):
        for c in self.componentes:
            if c.tipo not in TIPOS:
                raise ValueError(f"tipo de señal desconocido: {c.tipo}")
            ciclos = c.desplazamiento_hz * self.periodo_s
            if abs(ciclos - round(ciclos)) > 1e-6:
                raise ValueError(f"{c.desplazamiento_hz} Hz no es múltiplo de 1/{self.periodo_s:g} s: el búfer saltaría")
        if sum(c.amplitud for c in self.componentes) > 1.0 + 1e-9:
            raise ValueError("la suma de amplitudes supera la plena escala")

    def evaluar(self, t: np.ndarray) -> np.ndarray:
        x = np.zeros(t.shape, dtype=np.complex128)
        for c in self.componentes:
            x += c.amplitud * onda(c.tipo, t, c.semilla) * np.exp(2j * np.pi * c.desplazamiento_hz * t)
        return x

    def bufer(self, fs: float) -> np.ndarray:
        n = int(round(self.periodo_s * fs))
        return self.evaluar(np.arange(n) / fs).astype(np.complex64)

    def describir(self) -> List[dict]:
        return [{"tipo": c.tipo, "desplazamiento_hz": c.desplazamiento_hz, "amplitud": c.amplitud} for c in self.componentes]


def redondear_a_periodo(f_hz: float, periodo_s: float) -> float:
    return round(f_hz * periodo_s) / periodo_s


def semilla_emision(canal: int, tipo: str) -> int:
    return 1000 * (TIPOS.index(tipo) + 1) + canal


def emisiones_a_senal(emisiones: Sequence[Tuple[int, str]], desplazamientos_hz: Sequence[float]) -> Senal:
    """Señal del Pluto del entorno: una componente por emisión, cada una en el desplazamiento de su canal."""
    comps = [Componente(tipo, redondear_a_periodo(desplazamientos_hz[c], T_BUF_ENTORNO), AMP_ENTORNO,
                        semilla_emision(c, tipo)) for c, tipo in emisiones]
    return Senal(comps, T_BUF_ENTORNO)
