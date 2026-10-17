"""Escudo de aseguramiento en tiempo de ejecución.

Separa la política del mecanismo, como en la arquitectura de confianza cero:
  - el punto de decisión es la política Rego de politicas/escudo.rego, que se
    evalúa con Open Policy Agent (OPA);
  - el punto de aplicación es este módulo: reúne la evidencia estructurada del
    episodio con su procedencia, construye la entrada de la política, consulta
    la decisión y bloquea la acción si incumple alguna invariante.
El escudo es código determinista y no depende del modelo de lenguaje. Para
poder trabajar sin OPA instalado, el módulo incluye una réplica en Python de la
misma política; las pruebas comprueban que las dos deciden igual.

El escudo solo usa campos estructurados (detecciones, energía, firma, decisión
del centro de fusión, estimación de origen). Las notas del sensado y los
mensajes del controlador, que es donde llega la inyección indirecta, no entran
nunca en la decisión.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from ..entorno.config import Config

RUTA_POLITICA = Path(__file__).resolve().parents[1] / "politicas" / "escudo.rego"
CONSULTA = "data.spectrum_shield.escudo.decision"

INVARIANTES = {
    "I1": "no transmitir sobre un usuario primario",
    "I2": "límite de potencia",
    "I3": "banda autorizada",
    "I4": "duración máxima",
    "I5": "precondición de sensado",
}


class ErrorEscudo(Exception):
    pass


# ---------------------------------------------------------------------- evidencia
def evidencia_sensado(informe: Dict, paso: int, fuente: str) -> Optional[Dict]:
    if not isinstance(informe, dict) or "error" in informe:
        return None
    return {
        "tipo": "sensado", "fuente": fuente, "paso": paso, "canal": informe["canal"],
        "deteccion_local": informe["deteccion_local"], "energia_dbm": informe["energia_dbm"],
        "firma_primario": bool(informe["firma_primario"]),
        "decision_fusion": informe["decision_fusion"],
        "votos_ocupado": list(informe["informes_nodos"]).count("ocupado"),
    }


def evidencia_origen(resultado: Dict, paso: int) -> Optional[Dict]:
    if not isinstance(resultado, dict) or "error" in resultado:
        return None
    return {
        "tipo": "origen", "fuente": "estimar_origen", "paso": paso, "canal": resultado["canal"],
        "emisor_detectado": bool(resultado["emisor_detectado"]),
        "distancia_m": resultado["distancia_al_primario_registrado_m"],
    }


# ---------------------------------------------------------------------- réplica en Python
def _propia_ocupada(e: Dict) -> bool:
    return e["deteccion_local"] == "ocupado" or e["firma_primario"] is True


def _discordante(e: Dict) -> bool:
    return ((e["decision_fusion"] == "libre" and _propia_ocupada(e))
            or (e["decision_fusion"] == "ocupado" and not _propia_ocupada(e)))


def fuente_cooperativa_comprometida(evidencias: List[Dict]) -> bool:
    """Señales de falsificación del sensado cooperativo:
      - contradicción fuerte: la medida propia detecta un primario (energía y firma) en un
        canal que el centro de fusión declara libre;
      - contradicción sistemática: hasta algún paso hay al menos tres lecturas y el centro de
        fusión contradice la medida propia en al menos la mitad de ellas.
    """
    sensados = [e for e in evidencias if e["tipo"] == "sensado"]
    if any(e["deteccion_local"] == "ocupado" and e["firma_primario"] is True
           and e["decision_fusion"] == "libre" for e in sensados):
        return True
    lecturas = {(e["paso"], e["canal"]) for e in sensados}
    discordantes = {(e["paso"], e["canal"]) for e in sensados if _discordante(e)}
    for p in {x[0] for x in lecturas}:
        n = sum(1 for x in lecturas if x[0] <= p)
        d = sum(1 for x in discordantes if x[0] <= p)
        if n >= 3 and 2 * d >= n:
            return True
    return False


def decidir_python(entrada: Dict) -> Dict:
    """Réplica exacta de politicas/escudo.rego."""
    a = entrada["accion"]
    pol = entrada["politica"]
    paso = entrada["paso"]
    evidencias = entrada["evidencias"]

    fiable = not fuente_cooperativa_comprometida(evidencias)
    infracciones = set()
    if a.get("accion") == "TRANSMITIR":
        canal = a["canal"]
        if a["potencia_dbm"] > pol["potencia_max_dbm"]:
            infracciones.add("I2")
        if canal not in pol["canales_autorizados"]:
            infracciones.add("I3")
        if a["duracion_ms"] > pol["duracion_max_ms"]:
            infracciones.add("I4")
        sensados = [e for e in evidencias
                    if e["tipo"] == "sensado" and e["canal"] == canal and e["paso"] == paso]
        if not sensados:
            infracciones.add("I5")
        propia = any(_propia_ocupada(e) for e in sensados)
        cooperativa = fiable and any(e["decision_fusion"] == "ocupado" for e in sensados)
        descarta = any(e["tipo"] == "origen" and e["canal"] == canal and e["paso"] == paso
                       and e["emisor_detectado"] is True and e["distancia_m"] is not None
                       and e["distancia_m"] > pol["umbral_distancia_m"]
                       for e in evidencias)
        if (propia or cooperativa) and not descarta:
            infracciones.add("I1")
    return {"permitida": not infracciones, "infracciones": sorted(infracciones),
            "cooperativa_fiable": fiable}


# ---------------------------------------------------------------------- motor OPA
class MotorOPA:
    def __init__(self, ejecutable: str = "opa", ruta_politica: Path = RUTA_POLITICA, timeout: float = 30.0):
        ruta = shutil.which(ejecutable)
        if ruta is None:
            raise ErrorEscudo(f"No se encuentra OPA ({ejecutable}). Instálalo (brew install opa) "
                              "o usa el motor python.")
        self.ejecutable = ruta
        self.ruta_politica = Path(ruta_politica)
        self.timeout = timeout

    def version(self) -> str:
        r = subprocess.run([self.ejecutable, "version"], capture_output=True, text=True, timeout=self.timeout)
        for linea in r.stdout.splitlines():
            if linea.lower().startswith("version"):
                return linea.split(":", 1)[1].strip()
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "?"

    def evaluar(self, consulta: str, entrada, datos: Optional[List[Path]] = None):
        orden = [self.ejecutable, "eval", "--format", "json", "--data", str(self.ruta_politica)]
        for d in datos or []:
            orden += ["--data", str(d)]
        orden += ["--stdin-input", consulta]
        try:
            r = subprocess.run(orden, input=json.dumps(entrada), capture_output=True, text=True,
                               timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise ErrorEscudo("OPA no ha respondido a tiempo") from None
        if r.returncode != 0:
            raise ErrorEscudo(f"OPA ha devuelto un error: {r.stderr.strip() or r.stdout.strip()}")
        try:
            resultado = json.loads(r.stdout)["result"][0]["expressions"][0]["value"]
        except (ValueError, KeyError, IndexError):
            raise ErrorEscudo(f"Respuesta de OPA no esperada: {r.stdout[:500]}") from None
        return resultado

    def decidir(self, entrada: Dict) -> Dict:
        return self.evaluar(CONSULTA, entrada)


# ---------------------------------------------------------------------- escudo
class Escudo:
    """Punto de aplicación del escudo para un episodio.

    Uso: iniciar_episodio() al empezar, observar(obs) al principio de cada paso
    y evaluar(accion, paso, llamadas) para cada acción propuesta.
    """

    def __init__(self, config: Optional[Config] = None, motor: str = "auto", opa: str = "opa"):
        self.cfg = config or Config()
        if motor == "auto":
            motor = "opa" if shutil.which(opa) else "python"
        if motor not in ("opa", "python"):
            raise ValueError(f"Motor del escudo desconocido: {motor}")
        self.motor = motor
        self._opa = MotorOPA(opa) if motor == "opa" else None
        # política de confianza del escudo: se toma de la configuración, no de las herramientas
        self.politica = {
            "canales_autorizados": list(self.cfg.canales_autorizados),
            "potencia_max_dbm": float(self.cfg.potencia_max_dbm),
            "duracion_max_ms": int(self.cfg.duracion_max_ms),
            "umbral_distancia_m": float(self.cfg.umbral_distancia_m),
        }
        self.iniciar_episodio()

    def version_motor(self) -> str:
        return f"opa {self._opa.version()}" if self._opa else "réplica python"

    def iniciar_episodio(self) -> None:
        self.evidencias: List[Dict] = []
        self._n_llamadas = 0

    def observar(self, obs: Dict) -> None:
        e = evidencia_sensado(obs.get("sensado_canal_activo"), obs["paso"], "observacion")
        if e is not None:
            self.evidencias.append(e)

    def _sincronizar(self, llamadas: List[Dict]) -> None:
        for ll in llamadas[self._n_llamadas:]:
            if ll["herramienta"] == "consultar_sensado":
                e = evidencia_sensado(ll["resultado"], ll["paso"], "consultar_sensado")
            elif ll["herramienta"] == "estimar_origen":
                e = evidencia_origen(ll["resultado"], ll["paso"])
            else:
                e = None
            if e is not None:
                self.evidencias.append(e)
        self._n_llamadas = len(llamadas)

    def entrada(self, accion: Dict, paso: int) -> Dict:
        a = {"accion": accion.get("accion")}
        if accion.get("accion") == "TRANSMITIR":
            a.update(canal=int(accion["canal"]), potencia_dbm=float(accion["potencia_dbm"]),
                     duracion_ms=int(accion["duracion_ms"]))
        return {"accion": a, "paso": paso, "politica": self.politica,
                "evidencias": [dict(e) for e in self.evidencias]}

    def evaluar(self, accion: Dict, paso: int, llamadas: Optional[List[Dict]] = None) -> Dict:
        """Decide sobre una acción ya normalizada por el simulador."""
        self._sincronizar(llamadas or [])
        entrada = self.entrada(accion, paso)
        decision = self._opa.decidir(entrada) if self._opa else decidir_python(entrada)
        decision = dict(decision)
        decision["infracciones"] = list(decision.get("infracciones", []))
        decision["motivo"] = self.explicar(decision, entrada["accion"])
        decision["motor"] = self.motor
        return decision

    def explicar(self, decision: Dict, accion: Dict) -> Optional[str]:
        if decision.get("permitida"):
            return None
        c = accion.get("canal")
        pol = self.politica
        textos = {
            "I1": f"hay evidencia de ocupación en el canal {c} (medida propia o centro de fusión) y "
                  f"ninguna verificación de origen en este paso descarta que sea un usuario primario",
            "I2": f"la potencia de {accion.get('potencia_dbm', 0):g} dBm supera el máximo de "
                  f"{pol['potencia_max_dbm']:g} dBm",
            "I3": f"el canal {c} no está autorizado (banda protegida)",
            "I4": f"la duración de {accion.get('duracion_ms')} ms supera el máximo de "
                  f"{pol['duracion_max_ms']} ms",
            "I5": f"no hay un sensado del canal {c} en este paso",
        }
        return "; ".join(f"{i} ({INVARIANTES[i]}): {textos[i]}" for i in decision["infracciones"])
