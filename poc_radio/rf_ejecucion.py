"""Piezas comunes de los scripts de la prueba de concepto: carga del experimento, apertura de la radio y episodios.

Los episodios se ejecutan con el mismo ejecutor, los mismos sistemas, el mismo escudo y las mismas semillas que
la campaña simulada. Lo único que cambia es la clase del simulador, que se sustituye en el ejecutor igual que
hacen los scripts de los ataques adaptativos y del respaldo Simplex.

Todo lo que se genera con el banco simulado (sin hardware) va a carpetas con «simulado» en el nombre, para que
no se pueda confundir nunca con una medida real.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import rf_plan as P  # noqa: E402
from rf_dispositivos import BancoSimulado, RxRtl, TxPluto, resolver_uri  # noqa: E402
from rf_entorno import VERSION_POC, ControlRadio, crear_escudo_registrado, crear_simulador_radio  # noqa: E402

RUTA_CONF = AQUI / "radio.json"
ARCHIVOS_POC = ("rf_plan.py", "rf_ondas.py", "rf_espectro.py", "rf_dispositivos.py", "rf_captura.py",
                "rf_entorno.py", "rf_ejecucion.py")

# configuración de cada caso: sistema, textos de la inyección y modo del escudo (como en la campaña)
CONFIGURACIONES = {
    "reglas": {"sistema": "reglas", "texto": "original", "modo": "sombra",
               "titulo": "Reglas deterministas"},
    "sin_escudo": {"sistema": "agente", "texto": "original", "modo": "sombra",
                   "titulo": "Agente sin escudo"},
    "escudo": {"sistema": "agente_escudo", "texto": "original", "modo": "activo",
               "titulo": "Agente con escudo"},
    "insistente": {"sistema": "agente_escudo", "texto": "insistente", "modo": "activo",
                   "titulo": "Agente con escudo, ataque insistente"},
    "simplex": {"sistema": "agente_escudo_simplex", "texto": "insistente", "modo": "activo",
                "titulo": "Escudo con respaldo Simplex, ataque insistente"},
}
# campaña simulada con la que se empareja cada configuración (mismas semillas)
REFERENCIA_SIMULADA = {"reglas": "campana", "sin_escudo": "campana", "escudo": "campana",
                       "insistente": "adaptativa_insistente", "simplex": "simplex_insistente"}


def huella(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:12]


def huellas_poc() -> Dict[str, str]:
    return {a: huella(AQUI / a) for a in ARCHIVOS_POC}


def leer_conf() -> Dict:
    if not RUTA_CONF.exists():
        return {}
    return json.loads(RUTA_CONF.read_text(encoding="utf-8"))


def carpeta_salida(simulado: bool) -> Path:
    return RAIZ / "resultados" / ("poc_radio_simulado" if simulado else "poc_radio")


def ruta_calibracion(simulado: bool) -> Path:
    return carpeta_salida(simulado) / "calibracion.json"


# ---------------------------------------------------------------- módulos del experimento
class Experimento:
    """Carga respaldo_simplex.py (que a su vez carga ataque_adaptativo.py) y sustituye el simulador del ejecutor."""

    def __init__(self):
        spec = importlib.util.spec_from_file_location("respaldo_simplex", RAIZ / "scripts" / "respaldo_simplex.py")
        self.RS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.RS)
        self.ADA = self.RS.ADA
        from spectrum_shield.evaluacion import ejecutor as _ejecutor
        from spectrum_shield.sistemas import Escudo
        self.SimuladorRadio = crear_simulador_radio(self.ADA.SimuladorConSobreescritura)
        _ejecutor.Simulador = self.SimuladorRadio
        self._ejecutor = _ejecutor
        self.EscudoRegistrado = crear_escudo_registrado(Escudo)

    def comprobar_sustitucion(self) -> None:
        if self._ejecutor.Simulador is not self.SimuladorRadio:
            raise RuntimeError("Otro módulo ha vuelto a sustituir el simulador del ejecutor")

    def construir(self, nombre_conf: str, cfg, modelo: str, cliente):
        c = CONFIGURACIONES[nombre_conf]
        from spectrum_shield.sistemas import AgenteLLM, SistemaReglas
        if c["sistema"] == "reglas":
            return SistemaReglas(cfg)
        if c["sistema"] == "agente_escudo_simplex":
            return self.RS.AgenteSimplex(cfg, modelo, cliente, nombre=c["sistema"])
        return AgenteLLM(cfg, modelo, cliente, nombre=c["sistema"])

    def sobreescritura(self, nombre_conf: str, cfg, esc, semilla: int) -> Optional[Dict]:
        if CONFIGURACIONES[nombre_conf]["texto"] != "insistente" or not esc.compuesto:
            return None
        iny = self.ADA.inyeccion_de(cfg, esc, semilla)
        return {"texto": self.ADA.texto_insistente(iny), "id": iny["id"] + "+INS"}


# ---------------------------------------------------------------- radio
def abrir_radio(simulado: bool, conf: Optional[Dict] = None, calibracion: Optional[Dict] = None,
                aviso=print) -> ControlRadio:
    conf = conf if conf is not None else leer_conf()
    abiertos = []
    try:
        if simulado:
            banco = BancoSimulado(semilla=int(conf.get("semilla_simulada", 0)))
            a, b = banco.transmisor("entorno"), banco.transmisor("secundario")
            rx = banco.receptor(P.CENTRO_RTL_HZ, ganancia_db=float(conf.get("ganancia_rtl_db", 28.0)))
        else:
            a = TxPluto(resolver_uri(conf, "entorno"), "entorno")
            abiertos.append(a)
            b = TxPluto(resolver_uri(conf, "secundario"), "secundario")
            abiertos.append(b)
            rx = RxRtl(P.CENTRO_RTL_HZ, ganancia_db=float(conf.get("ganancia_rtl_db", 28.0)),
                       ppm=int(conf.get("ppm_rtl", 0)), serie=str(conf.get("serie_rtl", "")))
        return ControlRadio(a, b, rx, calibracion, aviso)
    except BaseException:
        for d in abiertos:
            try:
                d.cerrar()
            except Exception:  # noqa: BLE001
                pass
        raise


def objetivo_snr(conf: Dict) -> Dict[str, float]:
    return {"entorno": float(conf.get("objetivo_snr_entorno_db", 30.0)),
            "secundario": float(conf.get("objetivo_snr_secundario_db", 36.0))}


def preparar_radio(simulado: bool, recalibrar_todo: bool = False, aviso=print) -> Tuple[ControlRadio, Dict]:
    """Abre la radio con la calibración guardada y repite la parte rápida (frecuencia y ruido) en cada sesión."""
    conf = leer_conf()
    ruta = ruta_calibracion(simulado)
    previa = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
    control = abrir_radio(simulado, conf, previa, aviso)
    try:
        if previa is None or recalibrar_todo:
            aviso("   calibración completa (no hay calibración guardada o se ha pedido de nuevo)")
            cal = control.calibrar(objetivo_snr(conf))
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            cal = {"base": previa.get("fecha"), "ruido": control.medir_ruido()}
            for p in control.tx:
                cal[p] = {"frecuencia": control.calibrar_frecuencia(p), "ganancia_db": control.ganancia_db[p]}
                cal[p]["desfase_hz"] = round(control.desfase_hz[p], 1)
            control.medir_ruido()
        aviso("   " + resumen_calibracion(control))
        return control, cal
    except BaseException:
        control.cerrar()
        raise


def resumen_calibracion(control: ControlRadio) -> str:
    return (f"desfase entorno {control.desfase_hz['entorno']:+.0f} Hz, secundario {control.desfase_hz['secundario']:+.0f} Hz; "
            f"ganancias {control.ganancia_db['entorno']:.1f} y {control.ganancia_db['secundario']:.1f} dB; "
            f"ruido {min(control.ruido_db):.1f} a {max(control.ruido_db):.1f} dB")


# ---------------------------------------------------------------- episodios
def correr_episodio(exp: Experimento, nombre_conf: str, esc, semilla: int, cfg, modelo: str, cliente, escudo,
                    control: ControlRadio, campana: str, decimar: int = 1) -> Tuple[Dict, Dict]:
    """Un episodio con radio real. Devuelve el episodio (con su registro de radio) y la grabación de la cascada."""
    from spectrum_shield.evaluacion import ejecutar_episodio
    exp.comprobar_sustitucion()
    c = CONFIGURACIONES[nombre_conf]
    exp.SimuladorRadio.radio = control
    escudo.captura, escudo.modo = control.captura, c["modo"]
    control.estado.update(caso=nombre_conf, titulo=c["titulo"], sistema=c["sistema"], texto=c["texto"], modo=c["modo"])
    exp.ADA.SimuladorConSobreescritura.sobreescritura = exp.sobreescritura(nombre_conf, cfg, esc, semilla)
    t0 = control.captura.empezar_grabacion()
    try:
        sistema = exp.construir(nombre_conf, cfg, modelo, cliente)
        ep = ejecutar_episodio(sistema, esc, semilla, cfg, escudo, c["modo"], campana=campana)
    finally:
        exp.ADA.SimuladorConSobreescritura.sobreescritura = None
        control.silencio()
        grab = control.captura.terminar_grabacion(decimar)
    sim = exp.SimuladorRadio.ultimo
    ep["radio"] = sim.registro_radio
    ep["radio"].update(configuracion=nombre_conf, texto=c["texto"], t0=round(t0, 3),
                       desfase_hz=dict(control.desfase_hz), ganancia_db=dict(control.ganancia_db),
                       dispositivos={p: tx.info() for p, tx in control.tx.items()}, receptor=control.rx.info())
    grab["t0"] = t0
    return ep, grab


def guardar_cascada(ruta: Path, grab: Dict, ep: Optional[Dict] = None, piso_db: Optional[np.ndarray] = None,
                    analizador=None) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    eventos = [{"t": round(t - grab["t0"], 3), "tipo": tipo, **datos} for t, tipo, datos in grab["eventos"]]
    meta = {"version_poc": VERSION_POC, "dt_fila": grab["dt_fila"], "canales_hz": list(P.CANALES_HZ),
            "semiancho_franja_hz": P.SEMIANCHO_FRANJA_HZ}
    if ep is not None:
        r = ep["resumen"]
        meta.update(sistema=ep["sistema"], modelo=ep["modelo"], escenario=ep["escenario"], semilla=ep["semilla"],
                    configuracion=ep["radio"]["configuracion"], canal_inicial=r["canal_inicial"],
                    canal_refugio=r["canal_refugio"], inyeccion=(r["inyeccion"] or {}).get("id"),
                    canal_objetivo=(r["inyeccion"] or {}).get("canal_objetivo"),
                    rol_canal=(r["inyeccion"] or {}).get("rol_canal"), metricas=ep["metricas"],
                    pasos=[{"paso": p["paso"], "origen": p.get("origen_accion"),
                            "respaldo": bool((p.get("agente") or {}).get("respaldo")),
                            "accion": p["ejecutada"]["accion"].get("accion"), "canal": p["ejecutada"]["accion"].get("canal"),
                            "perjudicial": p["ejecutada"]["perjudicial"]} for p in ep["pasos"]])
    extra = {}
    if piso_db is not None and analizador is not None:
        extra["piso_franjas_db"] = np.stack([piso_db[idx] for idx in analizador.franja]).astype(np.float32)
    np.savez_compressed(ruta, t=(grab["t"] - grab["t0"]).astype(np.float32), franjas=grab["franjas"],
                        eventos=np.array(json.dumps(eventos, ensure_ascii=False, default=str)),
                        meta=np.array(json.dumps(meta, ensure_ascii=False, default=str)), **extra)


def leer_semillas(texto: str):
    out = []
    for parte in str(texto).split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = parte.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif parte:
            out.append(int(parte))
    return sorted(set(out))


def marca(v) -> str:
    return "-" if v is None else ("1" if v else "0")


def fmt_t(s: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(s))
