"""Línea base de reglas deterministas.

Codifica la buena práctica del dominio, con el mismo conocimiento que se da al
agente en sus instrucciones, para que la comparación de H3 no sea contra un
hombre de paja:
  - comprobación de consistencia entre la tasa de entrega y la intensidad de
    señal del enlace para separar la interferencia intencionada del fallo del
    canal (Xu et al.);
  - verificación del origen del emisor frente al transmisor primario
    registrado para detectar la emulación de primario (Chen, Park y Reed);
  - un canal solo se usa si el centro de fusión y la medida propia coinciden
    en que está libre y no hay firma de primario; si se ha diagnosticado
    falsificación del sensado, se ignoran los informes cooperativos y se usan
    solo las medidas propias.
Las reglas no leen texto libre (notas, mensajes), por lo que la inyección
indirecta no les afecta: es su ventaja frente al agente y también su límite,
porque no pueden aprovechar información que no esté prevista en el árbol.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..entorno.config import Config
from ..incidentes import catalogo as cat
from .herramientas import Herramientas


class SistemaReglas:
    nombre = "reglas"
    usa_modelo = False

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.iniciar_episodio({})

    def iniciar_episodio(self, info: Dict, semilla: int = 0) -> None:
        self.diagnostico: Optional[str] = None
        self.desconfiar_nodos = False
        self.politica = info.get("politica", {})

    # ------------------------------------------------------------------ decisión
    def decidir(self, obs: Dict, h: Herramientas, validar=None) -> Dict:
        # las reglas no usan validar: el ejecutor valida su acción después (y el escudo, en sombra)
        cfg = self.cfg
        activo = obs["canal_activo"]
        s = obs["sensado_canal_activo"]
        enlace = obs["enlace"]
        enlace_ok = enlace is not None and enlace["tasa_entrega"] >= cfg.umbral_tasa_entrega
        ocupacion = self._emision_propia(s)

        if obs["transmitiendo"] and enlace_ok and not ocupacion:
            return self._transmitir(activo)

        if obs["transmitiendo"] and self.diagnostico is None:
            self.diagnostico = self._clasificar(s, enlace, h, activo)
            h.diagnosticar(self.diagnostico, "árbol de decisión de la línea base")
            if self.diagnostico == cat.FALSIFICACION_SENSADO:
                self.desconfiar_nodos = True

        excluir = {activo} if obs["transmitiendo"] else set()
        nuevo = self._elegir_canal(h, excluir)
        if nuevo is None:
            return {"accion": "ESPERAR"}
        return self._transmitir(nuevo)

    # ------------------------------------------------------------------ piezas
    def _transmitir(self, canal: int) -> Dict:
        return {"accion": "TRANSMITIR", "canal": canal,
                "potencia_dbm": self.cfg.potencia_defecto_dbm, "duracion_ms": self.cfg.duracion_defecto_ms}

    @staticmethod
    def _emision_propia(informe: Dict) -> bool:
        return informe.get("deteccion_local") == "ocupado" or bool(informe.get("firma_primario"))

    def _clasificar(self, s: Dict, enlace: Optional[Dict], h: Herramientas, canal: int) -> str:
        cfg = self.cfg
        if s.get("firma_primario"):
            origen = h.estimar_origen(canal)
            d = origen.get("distancia_al_primario_registrado_m")
            if origen.get("emisor_detectado") and d is not None and d > cfg.umbral_distancia_m:
                return cat.EMULACION_PRIMARIO
            if s.get("decision_fusion") == "libre":
                return cat.FALSIFICACION_SENSADO
            return cat.PRIMARIO_LEGITIMO
        rssi = enlace.get("rssi_dbm") if enlace else None
        if rssi is not None and rssi > cfg.umbral_rssi_dbm:
            return cat.INTERFERENCIA
        return cat.FALLO_CANAL

    def _elegir_canal(self, h: Herramientas, excluir: set) -> Optional[int]:
        autorizados = self.politica.get("canales_autorizados", self.cfg.canales_autorizados)
        candidatos = []
        for c in autorizados:
            if c in excluir:
                continue
            r = h.consultar_sensado(c)
            if "error" in r:
                continue
            libre_propio = r["deteccion_local"] == "libre" and not r["firma_primario"]
            libre_fusion = r["decision_fusion"] == "libre"
            if libre_propio and (libre_fusion or self.desconfiar_nodos):
                candidatos.append((r["energia_dbm"], c))
        if not candidatos:
            return None
        return min(candidatos)[1]
