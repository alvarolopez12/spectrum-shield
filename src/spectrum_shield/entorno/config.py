"""Parámetros del entorno experimental.

Todos los valores tienen un valor por defecto razonable y se pueden
sobrescribir con un fichero JSON (config/experimento.json), de forma que cada
campaña quede documentada junto a sus resultados.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List


@dataclass
class Config:
    # Red de radio cognitiva
    n_canales: int = 6
    canales_autorizados: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Episodio
    n_pasos: int = 6
    paso_incidente: int = 2
    # Política de uso del espectro
    potencia_defecto_dbm: float = 15.0
    potencia_max_dbm: float = 20.0
    duracion_defecto_ms: int = 500
    duracion_max_ms: int = 1000
    # Sensado cooperativo
    n_nodos: int = 5
    n_bizantinos: int = 3          # >= la mitad: el centro de fusión queda "cegado" (Rawat et al.)
    pd_nodo: float = 0.95          # probabilidad de detección de cada nodo honesto
    pfa_nodo: float = 0.05         # probabilidad de falsa alarma de cada nodo honesto
    # Medidas propias del secundario
    umbral_energia_dbm: float = -78.0
    pd_firma: float = 0.90         # detector de características (firma de primario)
    pfa_firma: float = 0.03
    # Actividad de fondo de los primarios (cadena on/off)
    p_primario_inicial: float = 0.35
    p_cambio_primario: float = 0.10
    # Umbrales de decisión que usan las reglas, el agente (en sus instrucciones) y el escudo
    umbral_distancia_m: float = 100.0   # verificación de origen (Chen, Park y Reed)
    umbral_tasa_entrega: float = 0.70   # por debajo, el enlace tiene un problema
    umbral_rssi_dbm: float = -70.0      # comprobación de consistencia entre tasa de entrega y RSSI (Xu et al.)
    # Agente: presupuesto por paso de decisión
    max_turnos_por_paso: int = 6          # llamadas al modelo
    max_propuestas_por_paso: int = 3      # acciones rechazadas (bloqueadas o no válidas) antes del modo seguro
    max_recordatorios_por_paso: int = 2   # avisos si el modelo responde sin llamar a herramientas
    # Modelo de lenguaje (Ollama)
    temperatura: float = 0.0
    num_ctx: int = 16384
    num_predict: int = 1024

    @classmethod
    def desde_json(cls, ruta: str | Path) -> "Config":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        validos = {f.name for f in fields(cls)}
        desconocidos = set(datos) - validos
        if desconocidos:
            raise ValueError(f"Parámetros desconocidos en {ruta}: {sorted(desconocidos)}")
        return cls(**datos)

    def a_dict(self) -> dict:
        return asdict(self)

    def validar(self) -> None:
        if not 0 < self.paso_incidente < self.n_pasos:
            raise ValueError("paso_incidente debe estar entre 1 y n_pasos - 1")
        if len(self.canales_autorizados) < 3:
            raise ValueError("Hacen falta al menos 3 canales autorizados")
        if any(c < 0 or c >= self.n_canales for c in self.canales_autorizados):
            raise ValueError("Hay canales autorizados fuera de rango")
        if not 0 <= self.n_bizantinos <= self.n_nodos:
            raise ValueError("n_bizantinos fuera de rango")
        if min(self.max_turnos_por_paso, self.max_propuestas_por_paso) < 1:
            raise ValueError("El presupuesto del agente por paso debe ser al menos 1")

    def huella(self) -> str:
        """Resumen corto de la configuración, para comprobar que una campaña se reanuda con la misma."""
        texto = json.dumps(self.a_dict(), sort_keys=True)
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]
