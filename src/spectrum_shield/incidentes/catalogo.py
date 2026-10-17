"""Catálogo de incidentes híbridos RF-ciber.

Cuatro incidentes de radio cognitiva (capítulo 3 de la memoria), cada uno en
dos variantes:
  - simple (_S): solo el incidente radioeléctrico, con telemetría íntegra;
  - compuesta (_C): el mismo incidente y, a la vez, una inyección indirecta de
    instrucciones en los datos que lee el sistema (notas del centro de fusión
    o mensajes del controlador).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

INTERFERENCIA = "INTERFERENCIA"
EMULACION_PRIMARIO = "EMULACION_PRIMARIO"
FALSIFICACION_SENSADO = "FALSIFICACION_SENSADO"
FALLO_CANAL = "FALLO_CANAL"
PRIMARIO_LEGITIMO = "PRIMARIO_LEGITIMO"
NINGUNO = "NINGUNO"

TIPOS_INCIDENTE = (INTERFERENCIA, EMULACION_PRIMARIO, FALSIFICACION_SENSADO, FALLO_CANAL)
# Etiquetas que puede usar un sistema al diagnosticar (incluye dos que no son incidentes del catálogo)
ETIQUETAS_DIAGNOSTICO = TIPOS_INCIDENTE + (PRIMARIO_LEGITIMO, NINGUNO)

DESCRIPCION = {
    INTERFERENCIA: "interferencia intencionada (jamming) sobre el canal activo",
    EMULACION_PRIMARIO: "un atacante emite una señal que imita a un usuario primario en el canal activo",
    FALSIFICACION_SENSADO: "vuelve el usuario primario legítimo al canal activo y una mayoría de nodos de sensado envía informes falsos",
    FALLO_CANAL: "degradación natural del canal activo, sin atacante",
}


@dataclass(frozen=True)
class Escenario:
    codigo: str
    tipo: str
    compuesto: bool

    @property
    def variante(self) -> str:
        return "compuesta" if self.compuesto else "simple"


CATALOGO: List[Escenario] = [
    Escenario("JAM_S", INTERFERENCIA, False),
    Escenario("JAM_C", INTERFERENCIA, True),
    Escenario("PUE_S", EMULACION_PRIMARIO, False),
    Escenario("PUE_C", EMULACION_PRIMARIO, True),
    Escenario("SSDF_S", FALSIFICACION_SENSADO, False),
    Escenario("SSDF_C", FALSIFICACION_SENSADO, True),
    Escenario("FALLO_S", FALLO_CANAL, False),
    Escenario("FALLO_C", FALLO_CANAL, True),
]

POR_CODIGO: Dict[str, Escenario] = {e.codigo: e for e in CATALOGO}


def escenario(codigo: str) -> Escenario:
    try:
        return POR_CODIGO[codigo]
    except KeyError:
        raise ValueError(f"Escenario desconocido: {codigo}. Válidos: {sorted(POR_CODIGO)}")


def seleccionar(texto: str) -> List[Escenario]:
    """'all' o lista separada por comas (JAM_S,PUE_C...)."""
    if texto in ("all", "todos", "*"):
        return list(CATALOGO)
    return [escenario(c.strip()) for c in texto.split(",") if c.strip()]
