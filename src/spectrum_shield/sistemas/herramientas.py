"""Herramientas de consulta que comparten todos los sistemas.

Son la única vía de percepción del entorno: el sistema no ve la verdad del
terreno, solo lo que le devuelven estas funciones. Cada llamada queda
registrada, para la trayectoria del episodio y para que el escudo pueda leer
la evidencia con su procedencia (herramienta y paso).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from ..entorno.simulador import Simulador


def _a_entero(valor: Any):
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return valor
    return int(f) if f == int(f) else valor


class Herramientas:
    def __init__(self, sim: Simulador):
        self.sim = sim
        self.llamadas: List[Dict] = []

    def _registrar(self, nombre: str, argumentos: Dict, resultado: Dict) -> Dict:
        self.llamadas.append({"paso": self.sim.paso, "herramienta": nombre,
                              "argumentos": copy.deepcopy(argumentos),
                              "resultado": copy.deepcopy(resultado)})
        return resultado

    def consultar_sensado(self, canal) -> Dict:
        canal = _a_entero(canal)
        return self._registrar("consultar_sensado", {"canal": canal}, self.sim.informe_sensado(canal))

    def estimar_origen(self, canal) -> Dict:
        canal = _a_entero(canal)
        return self._registrar("estimar_origen", {"canal": canal}, self.sim.estimacion_origen(canal))

    def consultar_politica(self) -> Dict:
        return self._registrar("consultar_politica", {}, self.sim.politica())

    def consultar_historial(self) -> Dict:
        return self._registrar("consultar_historial", {}, self.sim.historial_publico())

    def diagnosticar(self, etiqueta, justificacion: str = "") -> Dict:
        justificacion = "" if justificacion is None else str(justificacion)
        return self._registrar("diagnosticar", {"etiqueta": etiqueta, "justificacion": justificacion},
                               self.sim.registrar_diagnostico(etiqueta, justificacion))
