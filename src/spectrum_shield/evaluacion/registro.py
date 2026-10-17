"""Registro de campañas: un JSON por episodio, una tabla CSV y un manifiesto.

Estructura de una campaña:
  resultados/<campaña>/manifiesto.json    configuración, versiones y ejecuciones
  resultados/<campaña>/episodios/<sistema>/<escenario>/s000.json
  resultados/<campaña>/episodios.csv      una fila por episodio (se rehace desde los JSON)

Los JSON son la fuente de verdad: si una ejecución se interrumpe, al volver a
lanzarla se saltan los episodios que ya tienen su JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .. import __version__

RAIZ_CODIGO = Path(__file__).resolve().parents[1]

COLUMNAS = [
    "campana", "sistema", "modelo", "escenario", "tipo", "variante", "semilla",
    "m1", "m2", "m3", "m4", "m4_propuesta", "m5", "m1_propuesta",
    "diagnostico_incidente", "diagnostico_final", "primer_diagnostico", "paso_primer_diagnostico", "paso_recuperacion",
    "n_pasos_exito", "n_propuestas", "n_bloqueos", "n_bloqueos_no_perjudiciales", "n_no_validas",
    "n_modo_seguro", "n_sin_accion", "sombra_bloquearia", "sombra_bloquearia_perjudiciales",
    "sombra_perjudiciales_no_detectadas", "turnos_llm", "tokens_entrada", "tokens_salida",
    "recordatorios", "salidas_truncadas", "duracion_s", "inyeccion", "objetivo_inyeccion",
    "canal_inicial", "canal_refugio", "modo_escudo", "motor_escudo", "version", "fecha",
]


def nombre_seguro(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", texto).strip("-")


def id_sistema(sistema: str, modelo: Optional[str]) -> str:
    return sistema if not modelo else f"{sistema}__{nombre_seguro(modelo)}"


# Código que determina las trayectorias. Las métricas, la estadística y el registro no
# entran: se pueden recalcular sobre trayectorias ya guardadas sin repetir la campaña.
CODIGO_TRAYECTORIAS = ("__init__.py", "entorno", "incidentes", "sistemas", "politicas", "evaluacion/ejecutor.py")


def archivos_trayectorias() -> List[str]:
    """Ficheros del paquete que entran en la huella del código (rutas relativas)."""
    salida = []
    for ruta in sorted(RAIZ_CODIGO.rglob("*")):
        rel = ruta.relative_to(RAIZ_CODIGO).as_posix()
        if ruta.suffix not in (".py", ".rego") or "__pycache__" in ruta.parts or ruta.name.endswith("_test.rego"):
            continue
        if any(rel == c or rel.startswith(c + "/") for c in CODIGO_TRAYECTORIAS):
            salida.append(rel)
    return salida


def huella_codigo() -> str:
    """Resumen del código que genera las trayectorias (Python y Rego), para detectar cambios a mitad de campaña."""
    h = hashlib.sha256()
    for rel in archivos_trayectorias():
        h.update(rel.encode("utf-8"))
        h.update((RAIZ_CODIGO / rel).read_bytes())
    return h.hexdigest()[:12]


def _valor_csv(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    return v


def fila(ep: Dict) -> Dict:
    m = ep["metricas"]
    iny = ep["resumen"]["inyeccion"] or {}
    datos = {
        "campana": ep.get("campana"), "sistema": ep["sistema"], "modelo": ep.get("modelo"),
        "escenario": ep["escenario"], "tipo": ep["tipo"], "variante": ep["variante"], "semilla": ep["semilla"],
        "duracion_s": ep["duracion_s"], "inyeccion": iny.get("id"), "objetivo_inyeccion": iny.get("objetivo"),
        "canal_inicial": ep["resumen"]["canal_inicial"], "canal_refugio": ep["resumen"]["canal_refugio"],
        "modo_escudo": ep.get("modo_escudo"), "motor_escudo": ep.get("motor_escudo"),
        "version": ep.get("version"), "fecha": ep.get("fecha"),
    }
    for k in COLUMNAS:
        if k not in datos:
            datos[k] = m.get(k)
    return {k: _valor_csv(datos.get(k)) for k in COLUMNAS}


def escribir_json(ruta: Path, datos) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, ruta)


class Campana:
    def __init__(self, raiz, nombre: str):
        self.nombre = nombre
        self.dir = Path(raiz) / nombre
        self.dir_episodios = self.dir / "episodios"
        self.ruta_manifiesto = self.dir / "manifiesto.json"
        self.ruta_csv = self.dir / "episodios.csv"

    # ------------------------------------------------------------------ episodios
    def ruta(self, sistema_id: str, escenario: str, semilla: int) -> Path:
        return self.dir_episodios / sistema_id / escenario / f"s{semilla:03d}.json"

    def existe(self, sistema_id: str, escenario: str, semilla: int) -> bool:
        return self.ruta(sistema_id, escenario, semilla).exists()

    def guardar(self, ep: Dict) -> Path:
        ruta = self.ruta(id_sistema(ep["sistema"], ep.get("modelo")), ep["escenario"], ep["semilla"])
        escribir_json(ruta, ep)
        return ruta

    def episodios(self) -> Iterator[Dict]:
        if not self.dir_episodios.exists():
            return
        for ruta in sorted(self.dir_episodios.rglob("s*.json")):
            yield json.loads(ruta.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ tabla
    def config(self):
        """Configuración con la que se creó la campaña (la del manifiesto) o la de por defecto."""
        from ..entorno.config import Config
        if self.ruta_manifiesto.exists():
            man = json.loads(self.ruta_manifiesto.read_text(encoding="utf-8"))
            if "config" in man:
                return Config(**man["config"])
        return Config()

    def reconstruir_csv(self, recalcular: bool = True) -> int:
        """Rehace la tabla desde los JSON; por defecto recalcula las métricas con la definición actual."""
        from . import metricas
        cfg = self.config()
        filas = []
        for ep in self.episodios():
            if recalcular:
                ep["metricas"] = metricas.calcular(ep, cfg)
            filas.append(fila(ep))
        filas.sort(key=lambda f: (str(f["sistema"]), str(f["modelo"]), str(f["escenario"]), int(f["semilla"])))
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.ruta_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            w.writeheader()
            w.writerows(filas)
        return len(filas)

    def anadir_fila(self, ep: Dict) -> None:
        nuevo = not self.ruta_csv.exists()
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.ruta_csv.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            if nuevo:
                w.writeheader()
            w.writerow(fila(ep))

    # ------------------------------------------------------------------ manifiesto
    def preparar_manifiesto(self, cfg, extra: Dict, forzar: bool = False) -> Dict:
        """Crea el manifiesto o comprueba que la campaña se reanuda con la misma configuración y código."""
        ahora = datetime.now().isoformat(timespec="seconds")
        ejecucion = dict(extra)
        ejecucion.update(fecha=ahora, python=sys.version.split()[0], plataforma=platform.platform(),
                         maquina=platform.node())
        actual = {"config_huella": cfg.huella(), "huella_trayectorias": huella_codigo(), "version": __version__}
        if self.ruta_manifiesto.exists():
            man = json.loads(self.ruta_manifiesto.read_text(encoding="utf-8"))
            # las campañas creadas con la versión anterior del registro no tienen huella_trayectorias:
            # se añade al reanudar y desde entonces se comprueba
            if "huella_trayectorias" not in man:
                man["huella_trayectorias"] = actual["huella_trayectorias"]
                ejecucion["aviso"] = "huella de trayectorias añadida al reanudar"
            distintos = [k for k in actual if man.get(k) != actual[k]]
            if distintos and not forzar:
                raise RuntimeError(
                    f"La campaña '{self.nombre}' se creó con otra configuración o con otro código "
                    f"({', '.join(distintos)}). Usa otro nombre de campaña o --forzar si sabes lo que haces.")
            if distintos:
                ejecucion["aviso"] = f"reanudada con cambios en {', '.join(distintos)}"
            man.setdefault("ejecuciones", []).append(ejecucion)
        else:
            man = {"campana": self.nombre, "creada": ahora, **actual, "config": cfg.a_dict(),
                   "ejecuciones": [ejecucion]}
        escribir_json(self.ruta_manifiesto, man)
        return man

    def anotar_en_manifiesto(self, clave: str, valor) -> None:
        man = json.loads(self.ruta_manifiesto.read_text(encoding="utf-8"))
        man[clave] = valor
        escribir_json(self.ruta_manifiesto, man)


def cargar_csv(ruta: Path) -> List[Dict]:
    with Path(ruta).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
