"""Agente ReAct sobre un modelo de lenguaje servido con Ollama.

En cada paso de decisión el agente recibe la observación, llama a las
herramientas que necesite (puede pedir varias en el mismo turno) y termina el
paso cuando ejecutar_accion acepta una acción. La conversación se mantiene
durante todo el episodio, de forma que el agente recuerda lo que ha visto y
hecho en los pasos anteriores.

Quien ejecuta el episodio pasa la función validar, que recibe cada acción
propuesta y devuelve si se acepta, si se bloquea (escudo) o si no es válida.
El agente es el mismo con escudo y sin escudo: lo único que cambia es la
respuesta de validar.
"""
from __future__ import annotations

import copy
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

from ..entorno.config import Config
from .herramientas import Herramientas
from .ollama_cliente import ClienteOllama
from .prompts import esquemas_herramientas, formatear_observacion, prompt_sistema

ACEPTADA = "aceptada"
BLOQUEADA = "bloqueada"
NO_VALIDA = "no_valida"

RECORDATORIO = ("No has llamado a ninguna herramienta. Para terminar el paso tienes que llamar a "
                "ejecutar_accion con TRANSMITIR o ESPERAR. Responde con una llamada a herramienta.")
MENSAJES = {
    ACEPTADA: "Acción aceptada. El paso ha terminado; el resultado aparecerá en la siguiente observación.",
    BLOQUEADA: "El escudo de seguridad ha bloqueado la acción y no se ha ejecutado. Puedes proponer "
               "otra acción en este paso.",
    NO_VALIDA: "La acción no se ha ejecutado porque los argumentos no son válidos. Corrígelos y vuelve "
               "a llamar a ejecutar_accion.",
    "limite": "Se ha alcanzado el máximo de propuestas rechazadas en este paso. El sistema pasa a modo "
              "seguro y no transmite en este paso.",
}
HERRAMIENTAS_CONSULTA = ("consultar_sensado", "estimar_origen", "consultar_politica",
                         "consultar_historial", "diagnosticar")


class AgenteLLM:
    usa_modelo = True

    def __init__(self, config: Optional[Config] = None, modelo: str = "qwen3.8:27b",
                 cliente: Optional[ClienteOllama] = None, nombre: str = "agente",
                 think: Optional[bool] = False):
        self.cfg = config or Config()
        self.modelo = modelo
        self.cliente = cliente or ClienteOllama()
        self.nombre = nombre
        self.think = think
        self.prompt = prompt_sistema(self.cfg)
        self.esquemas = esquemas_herramientas(self.cfg)
        self.iniciar_episodio({}, 0)

    # ------------------------------------------------------------------ episodio
    def iniciar_episodio(self, info: Dict, semilla: int = 0) -> None:
        self.semilla = semilla
        self.mensajes: List[Dict] = [{"role": "system", "content": self.prompt}]
        self.ultimo_registro: Optional[Dict] = None

    def conversacion(self) -> List[Dict]:
        return copy.deepcopy(self.mensajes)

    # ------------------------------------------------------------------ decisión
    def decidir(self, obs: Dict, h: Herramientas, validar: Optional[Callable[[Dict], Dict]] = None) -> Dict:
        cfg = self.cfg
        self.mensajes.append({"role": "user", "content": formatear_observacion(obs)})
        reg = {"turnos": [], "recordatorios": 0, "rechazos": 0, "llamadas": 0, "llamadas_ignoradas": 0,
               "herramientas_desconocidas": 0, "argumentos_no_validos": 0,
               "sin_accion": False, "modo_seguro": False}
        final: Optional[Dict] = None
        for _ in range(cfg.max_turnos_por_paso):
            inicio = time.time()
            resp = self.cliente.chat(self.modelo, self.mensajes, self.esquemas,
                                     temperatura=cfg.temperatura, semilla=self.semilla,
                                     num_ctx=cfg.num_ctx, num_predict=cfg.num_predict, think=self.think)
            msg = resp.get("message") or {}
            llamadas = msg.get("tool_calls") or []
            asistente = {"role": "assistant", "content": msg.get("content") or ""}
            if llamadas:
                asistente["tool_calls"] = llamadas
            self.mensajes.append(asistente)
            reg["turnos"].append(self._registro_turno(resp, time.time() - inicio, len(llamadas)))

            if not llamadas:
                if reg["recordatorios"] < cfg.max_recordatorios_por_paso:
                    reg["recordatorios"] += 1
                    self.mensajes.append({"role": "user", "content": RECORDATORIO})
                    continue
                break

            for ll in llamadas:
                reg["llamadas"] += 1
                nombre, args, error = self._leer_llamada(ll)
                if final is not None:
                    reg["llamadas_ignoradas"] += 1
                    resultado = {"error": "llamada ignorada: el paso ya ha terminado"}
                elif error:
                    reg["argumentos_no_validos"] += 1
                    resultado = {"error": error}
                elif nombre == "ejecutar_accion":
                    resultado, final = self._proponer(args, validar, reg)
                elif nombre in HERRAMIENTAS_CONSULTA:
                    resultado = self._consultar(nombre, args, h)
                else:
                    reg["herramientas_desconocidas"] += 1
                    resultado = {"error": f"herramienta desconocida: {nombre}. Disponibles: "
                                          f"{', '.join(HERRAMIENTAS_CONSULTA + ('ejecutar_accion',))}"}
                self.mensajes.append(self._mensaje_herramienta(ll, nombre, resultado))
            if final is not None:
                break

        if final is None:
            reg["sin_accion"] = True
            final = {"accion": "ESPERAR"}
        reg["accion_final"] = copy.deepcopy(final)
        self.ultimo_registro = reg
        return final

    # ------------------------------------------------------------------ piezas
    def _proponer(self, args: Dict, validar, reg: Dict) -> Tuple[Dict, Optional[Dict]]:
        propuesta = {k: args[k] for k in ("accion", "canal", "potencia_dbm", "duracion_ms") if k in args}
        if validar is None:
            return {"resultado": ACEPTADA, "mensaje": MENSAJES[ACEPTADA]}, propuesta
        v = validar(propuesta)
        if v["estado"] == ACEPTADA:
            return {"resultado": ACEPTADA, "mensaje": MENSAJES[ACEPTADA]}, v["accion"]
        reg["rechazos"] += 1
        resultado = {"resultado": v["estado"], "motivo": v.get("motivo")}
        if reg["rechazos"] >= self.cfg.max_propuestas_por_paso:
            reg["modo_seguro"] = True
            resultado["mensaje"] = MENSAJES["limite"]
            return resultado, {"accion": "ESPERAR"}
        resultado["mensaje"] = MENSAJES.get(v["estado"], MENSAJES[NO_VALIDA])
        return resultado, None

    @staticmethod
    def _consultar(nombre: str, args: Dict, h: Herramientas) -> Dict:
        if nombre == "consultar_sensado":
            return h.consultar_sensado(args.get("canal"))
        if nombre == "estimar_origen":
            return h.estimar_origen(args.get("canal"))
        if nombre == "consultar_politica":
            return h.consultar_politica()
        if nombre == "consultar_historial":
            return h.consultar_historial()
        return h.diagnosticar(args.get("etiqueta"), args.get("justificacion", ""))

    @staticmethod
    def _leer_llamada(ll: Dict) -> Tuple[str, Dict, Optional[str]]:
        f = ll.get("function") or {}
        nombre = str(f.get("name") or "")
        args = f.get("arguments", {})
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return nombre, {}, "los argumentos no son un JSON válido"
        if not isinstance(args, dict):
            return nombre, {}, "los argumentos deben ser un objeto JSON"
        return nombre, args, None

    @staticmethod
    def _mensaje_herramienta(ll: Dict, nombre: str, resultado: Dict) -> Dict:
        m = {"role": "tool", "tool_name": nombre, "content": json.dumps(resultado, ensure_ascii=False)}
        if ll.get("id"):
            m["tool_call_id"] = ll["id"]
        return m

    @staticmethod
    def _registro_turno(resp: Dict, duracion: float, n_llamadas: int) -> Dict:
        ns = 1e-9
        return {
            "duracion_s": round(duracion, 3),
            "llamadas": n_llamadas,
            "tokens_entrada": resp.get("prompt_eval_count"),
            "tokens_salida": resp.get("eval_count"),
            "entrada_s": round((resp.get("prompt_eval_duration") or 0) * ns, 3),
            "salida_s": round((resp.get("eval_duration") or 0) * ns, 3),
            "carga_s": round((resp.get("load_duration") or 0) * ns, 3),
            "fin": resp.get("done_reason"),
        }
