"""Estadística para proporciones por episodio y comparaciones emparejadas.

  - Intervalo de Agresti-Coull al 95 % para una proporción (Agresti y Coull, 1998).
  - Contraste exacto de McNemar para dos sistemas sobre los mismos episodios
    (McNemar, 1947): solo cuentan los pares discordantes.
  - Bootstrap de percentiles estratificado por escenario (Efron y Tibshirani,
    1994), para una proporción y para la diferencia entre dos sistemas sobre
    los mismos episodios: se remuestrean episodios (o pares) dentro de cada
    escenario, de forma que cada réplica conserva la composición de la muestra.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Z95 = 1.959963984540054


def agresti_coull(k: int, n: int, z: float = Z95) -> Tuple[float, float, float]:
    """Devuelve (proporción, límite inferior, límite superior)."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    n_t = n + z * z
    p_t = (k + z * z / 2.0) / n_t
    margen = z * math.sqrt(p_t * (1.0 - p_t) / n_t)
    return (k / n, max(0.0, p_t - margen), min(1.0, p_t + margen))


def mcnemar_exacto(b: int, c: int) -> float:
    """p bilateral del contraste exacto de McNemar con b y c pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * cola)


def tabla_pares(a: Dict, b: Dict) -> Dict[str, int]:
    """Cruza dos diccionarios {clave_episodio: 0/1} sobre sus claves comunes."""
    comunes = sorted(set(a) & set(b))
    t = {"n": len(comunes), "ambos": 0, "solo_a": 0, "solo_b": 0, "ninguno": 0}
    for k in comunes:
        x, y = bool(a[k]), bool(b[k])
        if x and y:
            t["ambos"] += 1
        elif x:
            t["solo_a"] += 1
        elif y:
            t["solo_b"] += 1
        else:
            t["ninguno"] += 1
    t["p_mcnemar"] = mcnemar_exacto(t["solo_a"], t["solo_b"])
    return t


def _percentiles(muestras: List[float], nivel: float) -> Tuple[float, float]:
    muestras = sorted(muestras)
    a = (1.0 - nivel) / 2.0
    i = int(math.floor(a * (len(muestras) - 1)))
    j = int(math.ceil((1.0 - a) * (len(muestras) - 1)))
    return muestras[i], muestras[j]


def bootstrap_proporcion(estratos: Dict[str, Sequence[bool]], b: int = 10000, semilla: int = 2026,
                         nivel: float = 0.95) -> Tuple[float, float, float]:
    """Proporción global e intervalo bootstrap remuestreando dentro de cada estrato."""
    grupos = [list(map(bool, v)) for v in estratos.values() if len(v)]
    n = sum(len(g) for g in grupos)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(semilla)
    total = sum(sum(g) for g in grupos) / n
    replicas = []
    for _ in range(b):
        k = 0
        for g in grupos:
            m = len(g)
            k += sum(g[rng.randrange(m)] for _ in range(m))
        replicas.append(k / n)
    inf, sup = _percentiles(replicas, nivel)
    return (total, inf, sup)


def bootstrap_diferencia(estratos: Dict[str, Sequence[Tuple[bool, bool]]], b: int = 10000, semilla: int = 2026,
                         nivel: float = 0.95) -> Tuple[float, float, float]:
    """Diferencia de proporciones (A menos B) sobre pares emparejados, remuestreando pares por estrato."""
    grupos = [[(bool(x), bool(y)) for x, y in v] for v in estratos.values() if len(v)]
    n = sum(len(g) for g in grupos)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(semilla)
    total = sum(x - y for g in grupos for x, y in g) / n
    replicas = []
    for _ in range(b):
        d = 0
        for g in grupos:
            m = len(g)
            for _ in range(m):
                x, y = g[rng.randrange(m)]
                d += x - y
        replicas.append(d / n)
    inf, sup = _percentiles(replicas, nivel)
    return (total, inf, sup)


def proporcion(valores: Iterable[Optional[bool]]) -> Tuple[int, int]:
    """Cuenta (éxitos, casos) ignorando los valores ausentes."""
    k = n = 0
    for v in valores:
        if v is None:
            continue
        n += 1
        k += int(bool(v))
    return k, n
