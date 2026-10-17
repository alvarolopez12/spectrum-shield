"""Cliente mínimo de la API de Ollama, sin dependencias externas.

Usa la API local (http://localhost:11434). Las peticiones no pasan por ningún
proxy del sistema, para que una configuración corporativa no intercepte el
tráfico a localhost.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional


class ErrorOllama(Exception):
    """Error devuelto por Ollama al procesar una petición."""


class ErrorConexionOllama(ErrorOllama):
    """Ollama no responde (no está abierto o la dirección no es correcta)."""


class ClienteOllama:
    def __init__(self, url: str = "http://localhost:11434", timeout: float = 600.0,
                 reintentos: int = 2, espera_reintento_s: float = 5.0, keep_alive: str = "30m"):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.reintentos = reintentos
        self.espera_reintento_s = espera_reintento_s
        self.keep_alive = keep_alive
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._sin_think: set = set()

    def _peticion(self, ruta: str, cuerpo: Optional[Dict] = None) -> Dict:
        datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
        req = urllib.request.Request(self.url + ruta, data=datos,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if datos is not None else "GET")
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detalle = e.read().decode("utf-8", "replace")
            raise ErrorOllama(f"HTTP {e.code} en {ruta}: {detalle}") from None
        except urllib.error.URLError as e:
            raise ErrorConexionOllama(f"No se puede conectar con Ollama en {self.url}: {e.reason}. "
                                      "Comprueba que Ollama está abierto.") from None

    def version(self) -> str:
        return self._peticion("/api/version").get("version", "?")

    def modelos_locales(self) -> List[str]:
        return [m.get("name", "") for m in self._peticion("/api/tags").get("models", [])]

    def chat(self, modelo: str, mensajes: List[Dict], herramientas: Optional[List[Dict]] = None,
             temperatura: float = 0.0, semilla: int = 0, num_ctx: int = 16384,
             num_predict: int = 1024, think: Optional[bool] = False) -> Dict:
        cuerpo: Dict = {"model": modelo, "messages": mensajes, "stream": False,
                        "keep_alive": self.keep_alive,
                        "options": {"temperature": temperatura, "seed": semilla, "num_ctx": num_ctx,
                                    "num_predict": num_predict}}
        if herramientas:
            cuerpo["tools"] = herramientas
        if think is not None and modelo not in self._sin_think:
            cuerpo["think"] = think
        intento = 0
        while True:
            try:
                return self._peticion("/api/chat", cuerpo)
            except ErrorConexionOllama:
                raise
            except ErrorOllama as e:
                texto = str(e).lower()
                # algunos modelos sin modo de razonamiento rechazan el parámetro "think"
                if "think" in cuerpo and "think" in texto:
                    self._sin_think.add(modelo)
                    cuerpo.pop("think")
                    continue
                # errores transitorios del servidor (por ejemplo, al cargar o descargar un modelo)
                if intento < self.reintentos and ("http 5" in texto):
                    intento += 1
                    time.sleep(self.espera_reintento_s)
                    continue
                raise
