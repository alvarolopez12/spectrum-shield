"""Ejecución de un episodio completo y registro de su trayectoria.

Un episodio tiene n_pasos pasos. Antes del incidente (pasos 0 a t0 - 1) el
secundario transmite de forma automática en su canal inicial. Desde t0 decide
el sistema evaluado. Cada acción que propone pasa por un validador que:
  1. normaliza la acción y rechaza las que no son válidas;
  2. la clasifica contra la verdad del terreno (solo para el registro);
  3. consulta al escudo, que la bloquea si está en modo activo o solo anota
     lo que habría hecho si está en modo sombra.
"""
from __future__ import annotations

import copy
import time
from datetime import datetime
from typing import Dict, List, Optional

from .. import __version__
from ..entorno.config import Config
from ..entorno.simulador import ESPERAR, TRANSMITIR, Simulador
from ..incidentes.catalogo import Escenario
from ..sistemas.escudo import Escudo
from ..sistemas.herramientas import Herramientas
from . import metricas

MODOS_ESCUDO = ("ninguno", "sombra", "activo")


def _resumen_evaluacion(ev: Dict) -> Dict:
    return {"accion": copy.deepcopy(ev["accion"]), "perjudicial": ev["perjudicial"],
            "motivos": list(ev["motivos"]), "exito": ev["exito"]}


class Validador:
    """Valida las acciones que propone un sistema en un paso de decisión."""

    def __init__(self, sim: Simulador, h: Herramientas, escudo: Optional[Escudo], modo: str):
        self.sim, self.h, self.escudo, self.modo = sim, h, escudo, modo
        self.paso = sim.paso
        self.propuestas: List[Dict] = []

    def __call__(self, propuesta: Dict) -> Dict:
        norm = self.sim.normalizar(propuesta)
        registro = {"orden": len(self.propuestas) + 1, "propuesta": copy.deepcopy(propuesta), "accion": norm,
                    "evaluacion": None, "escudo": None}
        self.propuestas.append(registro)
        if not norm["valida"]:
            registro["estado"] = "no_valida"
            return {"estado": "no_valida", "accion": norm, "motivo": norm.get("error")}
        ev = self.sim.evaluar_accion(norm)
        registro["evaluacion"] = {"perjudicial": ev["perjudicial"], "motivos": list(ev["motivos"]),
                                  "exito": ev["exito"]}
        decision = None
        if self.escudo is not None and self.modo != "ninguno":
            decision = self.escudo.evaluar(norm, self.paso, self.h.llamadas)
            decision["modo"] = self.modo
        registro["escudo"] = decision
        if self.modo == "activo" and decision is not None and not decision["permitida"]:
            registro["estado"] = "bloqueada"
            return {"estado": "bloqueada", "accion": norm, "motivo": decision["motivo"]}
        registro["estado"] = "aceptada"
        return {"estado": "aceptada", "accion": norm, "motivo": None}


def ejecutar_episodio(sistema, escenario: Escenario, semilla: int, cfg: Config,
                      escudo: Optional[Escudo] = None, modo_escudo: str = "sombra",
                      campana: Optional[str] = None) -> Dict:
    if modo_escudo not in MODOS_ESCUDO:
        raise ValueError(f"modo_escudo no válido: {modo_escudo}")
    if escudo is None:
        modo_escudo = "ninguno"
    inicio = time.time()
    sim = Simulador(cfg)
    sim.reiniciar(escenario, semilla)
    h = Herramientas(sim)
    sistema.iniciar_episodio(sim.info_publica(), semilla)
    if escudo is not None:
        escudo.iniciar_episodio()

    pasos: List[Dict] = []
    while not sim.terminado:
        t = sim.paso
        inicio_paso = time.time()
        obs = sim.observacion()
        estado = sim.estado_real(t)
        if escudo is not None:
            escudo.observar(obs)
        n_llamadas = len(h.llamadas)
        paso = {"paso": t, "automatico": t < cfg.paso_incidente, "estado_real": estado,
                "observacion": copy.deepcopy(obs), "propuestas": [], "agente": None}

        if paso["automatico"]:
            accion = {"accion": TRANSMITIR, "canal": sim.canal_activo,
                      "potencia_dbm": cfg.potencia_defecto_dbm, "duracion_ms": cfg.duracion_defecto_ms}
            paso["origen_accion"] = "automatica"
            ev = sim.ejecutar(accion)
        else:
            validador = Validador(sim, h, escudo, modo_escudo)
            final = sistema.decidir(obs, h, validador)
            origen, extra, registro = "propuesta", None, None
            if getattr(sistema, "usa_modelo", False):
                # el agente valida cada propuesta dentro de su bucle
                registro = sistema.ultimo_registro
                if registro.get("modo_seguro"):
                    origen = "modo_seguro"
                    extra = {"resultado": "modo seguro: se rechazaron las acciones propuestas y no se transmitió"}
                elif registro.get("sin_accion"):
                    origen = "sin_accion"
                    extra = {"resultado": "sin acción: no se llamó a ejecutar_accion y no se transmitió"}
            else:
                # las reglas no validan dentro de su bucle: se valida aquí su acción
                v = validador(final)
                if v["estado"] != "aceptada":
                    origen, final = "modo_seguro", {"accion": ESPERAR}
                    extra = {"resultado": f"no ejecutada ({v['estado']}): {v['motivo']}"}
            paso["origen_accion"] = origen
            paso["propuestas"] = validador.propuestas
            paso["agente"] = copy.deepcopy(registro)
            ev = sim.ejecutar(final, extra)

        paso["llamadas"] = copy.deepcopy(h.llamadas[n_llamadas:])
        paso["ejecutada"] = _resumen_evaluacion(ev)
        paso["duracion_s"] = round(time.time() - inicio_paso, 3)
        pasos.append(paso)

    episodio = {
        "version": __version__,
        "campana": campana,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "sistema": sistema.nombre,
        "modelo": getattr(sistema, "modelo", None) if getattr(sistema, "usa_modelo", False) else None,
        "modo_escudo": modo_escudo,
        "motor_escudo": escudo.motor if escudo is not None else None,
        "escenario": escenario.codigo,
        "tipo": escenario.tipo,
        "variante": escenario.variante,
        "semilla": semilla,
        "config_huella": cfg.huella(),
        "resumen": sim.resumen_episodio(),
        "diagnosticos": copy.deepcopy(sim.diagnosticos),
        "pasos": pasos,
        "conversacion": sistema.conversacion() if hasattr(sistema, "conversacion") else None,
        "duracion_s": round(time.time() - inicio, 3),
    }
    episodio["metricas"] = metricas.calcular(episodio, cfg)
    return episodio
