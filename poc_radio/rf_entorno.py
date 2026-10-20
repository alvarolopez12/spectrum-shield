"""La radio real dentro de los episodios: control de las emisiones, calibración y simulador con sensado real.

ControlRadio maneja los tres dispositivos durante un episodio:
  - el Pluto del entorno emite, en cada paso, lo que dice el guion del simulador para ese paso (primarios,
    emulador del ataque PUE e interferencia, cada uno en su canal);
  - el Pluto del secundario emite una ráfaga en el canal que decide el sistema evaluado, con la duración que
    pide la acción y una potencia que sube o baja con la que pide la acción (siempre dentro del límite legal);
  - el RTL-SDR mide. Cuando el sistema consulta el sensado de un canal, la energía, la detección local y la firma
    de primario salen de esa medida, no de las tablas del simulador.

crear_simulador_radio() devuelve una subclase del simulador que hace exactamente eso y deja todo lo demás igual:
la fusión cooperativa (con los nodos bizantinos), las notas y mensajes con la inyección, la estimación de la
distancia del emisor y el enlace siguen saliendo del simulador, con la misma semilla que en la campaña. La
estimación de origen solo puede localizar un emisor si la medida real supera el umbral de energía, como en el
simulador. El código de las trayectorias (src/spectrum_shield) no se modifica.
"""
from __future__ import annotations

import copy
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import rf_plan as P
from rf_captura import Captura
from rf_dispositivos import ErrorRadio, limitar_ganancia
from rf_espectro import Analizador, Medida, db
from rf_ondas import (AMP_SECUNDARIO, T_BUF_ENTORNO, T_BUF_SECUNDARIO, Componente, Senal, emisiones_a_senal,
                      redondear_a_periodo)

VERSION_POC = "radio 1.0 (19/10/2026)"
CANAL_REFERENCIA = 2
OBJETIVO_SNR_DB = {"entorno": 30.0, "secundario": 36.0}
GANANCIA_INICIAL_DB = {"entorno": -45.0, "secundario": -50.0}
AUMENTO_CONFIRMACION_DB = 3.0        # una ráfaga se da por confirmada si sube la potencia del canal al menos esto
MARGEN_FILA_AJENA_DB = 3.0           # al medir el ruido, se descartan las filas con emisiones ajenas


class ControlRadio:
    def __init__(self, tx_entorno, tx_secundario, receptor, calibracion: Optional[Dict] = None, aviso=print):
        self.tx = {"entorno": tx_entorno, "secundario": tx_secundario}
        self.rx = receptor
        self.an = Analizador(fs=receptor.fs, centro_hz=receptor.centro_hz)
        self.aviso = aviso
        cal = calibracion or {}
        self.desfase_hz = {p: float(cal.get(p, {}).get("desfase_hz", 0.0)) for p in self.tx}
        self.ganancia_db = {p: float(cal.get(p, {}).get("ganancia_db", GANANCIA_INICIAL_DB[p])) for p in self.tx}
        self.ruido_db: Optional[List[float]] = None
        self.piso_db: Optional[np.ndarray] = None
        self.emisiones: Tuple = ()
        self.t_valido = 0.0
        self.estado: Dict = {}
        for tx in self.tx.values():
            tx.silenciar()
        self.captura = Captura(receptor, self.an)
        self.captura.start()
        self.captura.esperar_filas(4, timeout=10)

    # ------------------------------------------------------------ primitivas
    def _cambio(self) -> None:
        self.t_valido = time.time() + P.ASENTAMIENTO_S

    def silencio(self) -> None:
        for tx in self.tx.values():
            tx.silenciar()
        self.emisiones = ()
        self._cambio()

    def desplazamiento(self, canal: int, papel: str, periodo: float) -> float:
        return redondear_a_periodo(P.CANALES_HZ[canal] - P.LO_PLUTO_HZ - self.desfase_hz[papel], periodo)

    def psd(self, n_filas: int = P.FILAS_MEDIDA):
        return self.captura.medir(self.t_valido, n_filas)

    def medir_ruido(self) -> Dict:
        """Ruido de fondo con los dos Pluto en silencio (1 s). Descarta las filas con emisiones ajenas."""
        self.silencio()
        filas = self.captura.filas_desde(self.t_valido, P.FILAS_RUIDO)
        pot = np.array([self.an.ruido_por_canal(f[2]) for f in filas])          # filas x canales, dB
        mediana = np.median(pot, axis=0)
        limpias = [f for f, p in zip(filas, pot) if np.all(p <= mediana + MARGEN_FILA_AJENA_DB)]
        usadas = limpias if len(limpias) >= len(filas) // 2 else filas
        psd = np.mean(np.stack([f[2] for f in usadas]).astype(np.float64), axis=0)
        self.ruido_db = self.an.ruido_por_canal(psd)
        self.piso_db = db(psd)
        info = {"ruido_db": [round(r, 2) for r in self.ruido_db], "filas": len(filas),
                "filas_descartadas": len(filas) - len(usadas)}
        if len(usadas) < len(filas):
            self.aviso(f"   aviso: {len(filas) - len(usadas)} de {len(filas)} filas con emisiones ajenas al medir el ruido")
        self.captura.evento("ruido", **info)
        return info

    def medir(self, canal: int) -> Medida:
        if self.ruido_db is None:
            self.medir_ruido()
        psd, t0, t1, n, _ = self.psd()
        m = self.an.medir(psd, canal, self.ruido_db[canal], self.piso_db, t0, t1, n)
        self.captura.evento("medida", canal=canal, energia_dbm=round(m.energia_dbm, 1), firma=m.firma,
                            snr_db=round(m.snr_db, 1))
        return m

    # ------------------------------------------------------------ emisiones del episodio
    def entorno(self, emisiones: Sequence[Tuple[int, str]]) -> None:
        emisiones = tuple(sorted((int(c), str(t)) for c, t in emisiones))
        if emisiones == self.emisiones:
            return
        tx = self.tx["entorno"]
        if emisiones:
            desp = [self.desplazamiento(k, "entorno", T_BUF_ENTORNO) for k in range(P.N_CANALES)]
            tx.emitir(emisiones_a_senal(emisiones, desp), self.ganancia_db["entorno"])
        else:
            tx.silenciar()
        self.emisiones = emisiones
        self._cambio()
        self.captura.evento("entorno", emisiones=[list(e) for e in emisiones])

    def senal_secundario(self, canal: int) -> Senal:
        return Senal([Componente("secundario", self.desplazamiento(canal, "secundario", T_BUF_SECUNDARIO), AMP_SECUNDARIO)],
                     T_BUF_SECUNDARIO)

    def rafaga(self, canal: int, potencia_dbm: float, duracion_ms: float) -> Dict:
        """Transmisión del secundario: una ráfaga en el canal decidido. Devuelve la confirmación medida en el aire."""
        previa = self.medir(canal)
        g = limitar_ganancia(self.ganancia_db["secundario"] + (float(potencia_dbm) - P.POTENCIA_REFERENCIA_DBM))
        d = float(min(max(float(duracion_ms) / 1000.0, P.RAFAGA_MIN_S), P.RAFAGA_MAX_S))
        tx = self.tx["secundario"]
        t_on = time.time()
        tx.emitir(self.senal_secundario(canal), g)
        self.captura.evento("rafaga", canal=canal, potencia_dbm=float(potencia_dbm), ganancia_db=g, duracion_s=d)
        try:
            time.sleep(d)
            durante = self.captura.medir_entre(t_on + P.ASENTAMIENTO_S, time.time())
        finally:
            tx.silenciar()
            t_off = time.time()
            self.captura.evento("fin_rafaga", canal=canal)
            self.t_valido = max(self.t_valido, t_off + P.ASENTAMIENTO_S)
        r = {"canal": canal, "potencia_dbm": float(potencia_dbm), "ganancia_db": g, "duracion_s": d,
             "previa_db": round(previa.potencia_db, 2), "previa_snr_db": round(previa.snr_db, 2),
             "t_on": round(t_on, 3), "t_off": round(t_off, 3), "confirmada": False}
        if durante is not None:
            psd, _, _, n, pico = durante
            pot = float(db(self.an.potencia_canal(psd, canal)))
            r.update(durante_db=round(pot, 2), durante_snr_db=round(pot - self.ruido_db[canal], 2),
                     aumento_db=round(pot - previa.potencia_db, 2), filas=n, pico_adc=round(pico, 3),
                     confirmada=bool(pot - previa.potencia_db >= AUMENTO_CONFIRMACION_DB))
        return r

    # ------------------------------------------------------------ calibración
    def calibrar_frecuencia(self, papel: str) -> Dict:
        """Error de frecuencia de un Pluto respecto del RTL-SDR, con un tono en el canal de referencia."""
        self.silencio()
        tx = self.tx[papel]
        f_nom = P.CANALES_HZ[CANAL_REFERENCIA] - P.LO_PLUTO_HZ
        r = None
        try:
            for g in (-50.0, -40.0, -30.0, -20.0):
                f_bb = redondear_a_periodo(f_nom, T_BUF_ENTORNO)
                tx.emitir(Senal([Componente("tono", f_bb, 0.5)], T_BUF_ENTORNO), g)
                self._cambio()
                psd = self.psd()[0]
                esperado = P.LO_PLUTO_HZ + f_bb - self.rx.centro_hz
                r = self.an.frecuencia_pico(psd, esperado, busqueda_hz=45e3)
                if r and r["pico_sobre_fondo_db"] >= 25:
                    break
            else:
                raise ErrorRadio(f"El RTL-SDR no ve el tono del Pluto del {papel} (ni con {g:g} dB). Comprueba las "
                                 "antenas, la distancia (1 a 3 m) y que el Pluto emite (LED).")
            desfase = r["frecuencia_hz"] - esperado
            self.desfase_hz[papel] = desfase
            # comprobación con la corrección aplicada
            f_bb = self.desplazamiento(CANAL_REFERENCIA, papel, T_BUF_ENTORNO)
            tx.emitir(Senal([Componente("tono", f_bb, 0.5)], T_BUF_ENTORNO), g)
            self._cambio()
            r2 = self.an.frecuencia_pico(self.psd()[0], P.CANALES_HZ[CANAL_REFERENCIA] - self.rx.centro_hz, 5e3)
        finally:
            tx.silenciar()
            self._cambio()
        residuo = None if r2 is None else r2["frecuencia_hz"] - (P.CANALES_HZ[CANAL_REFERENCIA] - self.rx.centro_hz)
        return {"desfase_hz": round(desfase, 1), "ppm": round(desfase / P.CANALES_HZ[CANAL_REFERENCIA] * 1e6, 2),
                "residuo_hz": None if residuo is None else round(residuo, 1), "ganancia_prueba_db": g,
                "tono_sobre_fondo_db": round(r["pico_sobre_fondo_db"], 1)}

    def _medir_canales(self, canales: Sequence[int]) -> List[Medida]:
        psd, t0, t1, n, pico = self.psd()
        self._ultimo_pico = pico
        return [self.an.medir(psd, k, self.ruido_db[k], self.piso_db, t0, t1, n) for k in canales]

    def calibrar_ganancias(self, objetivo: Optional[Dict[str, float]] = None) -> Dict:
        """Ajusta la ganancia de cada Pluto para que sus emisiones lleguen con la SNR objetivo (mediana de los canales)."""
        objetivo = objetivo or OBJETIVO_SNR_DB
        if self.ruido_db is None:
            self.medir_ruido()
        out = {}
        grupos = ([0, 2, 4], [1, 3, 5])
        # entorno: primario en todos los canales (en dos grupos), a la ganancia de prueba y a la ajustada
        def snr_entorno(g: float, tipo: str) -> Dict[int, Medida]:
            res = {}
            for grupo in grupos:
                self.tx["entorno"].emitir(emisiones_a_senal([(k, tipo) for k in grupo],
                                                            [self.desplazamiento(k, "entorno", T_BUF_ENTORNO)
                                                             for k in range(P.N_CANALES)]), g)
                self._cambio()
                res.update(dict(zip(grupo, self._medir_canales(grupo))))
            self.tx["entorno"].silenciar()
            self._cambio()
            return res
        g0 = self.ganancia_db["entorno"]
        m0 = snr_entorno(g0, "primario")
        mediana = statistics.median(m.snr_db for m in m0.values())
        g = limitar_ganancia(g0 + objetivo["entorno"] - mediana)
        self.ganancia_db["entorno"] = g
        mp, mi = snr_entorno(g, "primario"), snr_entorno(g, "interferencia")
        out["entorno"] = {"ganancia_db": g, "snr_primario_db": [round(mp[k].snr_db, 1) for k in range(P.N_CANALES)],
                          "firma_primario": [mp[k].firma for k in range(P.N_CANALES)],
                          "snr_interferencia_db": [round(mi[k].snr_db, 1) for k in range(P.N_CANALES)],
                          "firma_interferencia": [mi[k].firma for k in range(P.N_CANALES)],
                          "margen_pilotos_db": [round(min(mp[k].piloto_inf_db, mp[k].piloto_sup_db), 1)
                                                for k in range(P.N_CANALES)]}
        # secundario: el barrido en cada canal
        def snr_secundario(g: float) -> List[float]:
            res = []
            for k in range(P.N_CANALES):
                self.tx["secundario"].emitir(self.senal_secundario(k), g)
                self._cambio()
                res.append(self._medir_canales([k])[0].snr_db)
                self.tx["secundario"].silenciar()
            self._cambio()
            return res
        g0 = self.ganancia_db["secundario"]
        s0 = snr_secundario(g0)
        g = limitar_ganancia(g0 + objetivo["secundario"] - statistics.median(s0))
        self.ganancia_db["secundario"] = g
        out["secundario"] = {"ganancia_db": g, "snr_db": [round(x, 1) for x in snr_secundario(g)]}
        if g >= P.GANANCIA_TX_MAX_DB or self.ganancia_db["entorno"] >= P.GANANCIA_TX_MAX_DB:
            self.aviso("   aviso: algún Pluto ha llegado al tope de ganancia sin alcanzar la SNR objetivo; acerca las antenas")
        return out

    def comprobar_artefactos(self) -> Dict:
        """Imagen del Pluto del entorno, fuga en silencio y saturación del receptor en el peor caso."""
        out = {}
        desp = [self.desplazamiento(k, "entorno", T_BUF_ENTORNO) for k in range(P.N_CANALES)]
        # imagen: una emisión en el canal 0; su imagen cae entre los canales 4 y 5
        self.tx["entorno"].emitir(emisiones_a_senal([(0, "primario")], desp), self.ganancia_db["entorno"])
        self._cambio()
        psd = self.psd()[0]
        p0 = float(db(self.an.potencia_canal(psd, 0)))
        imagen = 2 * (P.LO_PLUTO_HZ + self.desfase_hz["entorno"]) - P.CANALES_HZ[0] - self.rx.centro_hz
        idx = self.an.indices(imagen)
        p_img = float(db(psd[idx].sum()))
        ruido_img = float(db((10 ** (self.piso_db[idx] / 10)).sum()))
        out["imagen_sobre_ruido_db"] = round(p_img - ruido_img, 1)
        # si la imagen no asoma sobre el ruido, solo se sabe que está por debajo de la cota
        out["imagen_dbc"] = round(p_img - p0, 1) if p_img - ruido_img >= 3 else None
        out["imagen_dbc_cota"] = round(ruido_img - p0, 1)
        # silencio: lo que queda con los dos Pluto silenciados
        self.silencio()
        ms = self._medir_canales(range(P.N_CANALES))
        out["fuga_en_silencio_db"] = round(max(m.snr_db for m in ms), 2)
        # peor caso: cinco emisiones del entorno y el secundario 15 dB por encima de la referencia (30 dBm pedidos)
        self.tx["entorno"].emitir(emisiones_a_senal([(k, "primario") for k in (0, 1, 2, 3, 4)], desp),
                                  self.ganancia_db["entorno"])
        g = limitar_ganancia(self.ganancia_db["secundario"] + 15.0)
        self.tx["secundario"].emitir(self.senal_secundario(5), g)
        self._cambio()
        m5 = self._medir_canales([5])[0]
        out["pico_adc_peor_caso"] = round(self._ultimo_pico, 3)
        out["snr_secundario_30dbm_db"] = round(m5.snr_db, 1)
        out["saturacion"] = bool(self._ultimo_pico >= 0.95)
        self.silencio()
        return out

    def calibrar(self, objetivo: Optional[Dict[str, float]] = None, ganancias: bool = True) -> Dict:
        t0 = time.time()
        cal = {"version_poc": VERSION_POC, "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "dispositivos": {p: tx.info() for p, tx in self.tx.items()}, "receptor": self.rx.info()}
        cal["ruido"] = self.medir_ruido()
        for p in self.tx:
            cal.setdefault(p, {})["frecuencia"] = self.calibrar_frecuencia(p)
        self.medir_ruido()
        if ganancias:
            g = self.calibrar_ganancias(objetivo)
            for p in self.tx:
                cal[p].update(g[p])
        cal["artefactos"] = self.comprobar_artefactos()
        for p in self.tx:
            cal[p]["desfase_hz"] = round(self.desfase_hz[p], 1)
            cal[p]["ganancia_db"] = self.ganancia_db[p]
        cal["duracion_s"] = round(time.time() - t0, 1)
        self.medir_ruido()
        return cal

    def cerrar(self) -> None:
        try:
            self.silencio()
        finally:
            self.captura.parar()
            for tx in self.tx.values():
                tx.cerrar()
            self.rx.cerrar()


# ====================================================================== simulador con sensado real
def emisiones_del_paso(guion_t) -> List[Tuple[int, str]]:
    out = []
    for c, e in enumerate(guion_t):
        if e.primario:
            out.append((c, "primario"))
        elif e.emulador:
            out.append((c, "emulador"))
        if e.interferencia:
            out.append((c, "interferencia"))
    return out


def crear_simulador_radio(Base):
    """Subclase del simulador (o de una subclase suya, como la del ataque insistente) con sensado real."""
    from spectrum_shield.entorno.simulador import TRANSMITIR

    class SimuladorRadio(Base):
        radio: Optional[ControlRadio] = None
        ultimo = None

        def reiniciar(self, escenario, semilla: int) -> None:
            super().reiniciar(escenario, semilla)
            type(self).ultimo = self
            self._preparado = None
            self.registro_radio = {"version_poc": VERSION_POC, "pasos": {}, "medidas": [], "origenes": [],
                                   "rafagas": [], "diagnosticos": []}
            if self.radio is not None:
                self.radio.estado.update(escenario=escenario.codigo, semilla=semilla, c0=self.c0, refugio=self.refugio,
                                         paso=None)
                self.registro_radio["ruido"] = self.radio.medir_ruido()
                self.radio.captura.evento("episodio", escenario=escenario.codigo, semilla=semilla, c0=self.c0,
                                          refugio=self.refugio)

        def _preparar(self) -> None:
            if self.radio is None or self.terminado or self._preparado == self.paso:
                return
            t = self.paso
            self._preparado = t
            emisiones = emisiones_del_paso(self.guion[t])
            self.radio.estado["paso"] = t
            self.radio.captura.evento("paso", paso=t, emisiones=[list(e) for e in emisiones])
            self.radio.entorno(emisiones)
            self.registro_radio["pasos"][str(t)] = {"emisiones": [list(e) for e in emisiones], "t": round(time.time(), 3)}

        def observacion(self) -> Dict:
            self._preparar()
            return super().observacion()

        def _verdad(self, canal: int) -> Dict:
            e = self.guion[min(self.paso, self.cfg.n_pasos - 1)][canal]
            return {"emision": e.hay_emision(), "firma": e.firma_primaria(), "degradado": e.degradado,
                    "tipo": ("primario" if e.primario else "emulador" if e.emulador else
                             "interferencia" if e.interferencia else "degradado" if e.degradado else "libre")}

        def informe_sensado(self, canal) -> Dict:
            inf = super().informe_sensado(canal)
            if self.radio is None or "error" in inf:
                return inf
            self._preparar()
            simulado = {k: inf[k] for k in ("energia_dbm", "deteccion_local", "firma_primario")}
            m = self.radio.medir(inf["canal"])
            inf["energia_dbm"] = round(m.energia_dbm, 1)
            inf["deteccion_local"] = "ocupado" if m.energia_dbm > self.cfg.umbral_energia_dbm else "libre"
            inf["firma_primario"] = bool(m.firma)
            self.registro_radio["medidas"].append({
                "paso": min(self.paso, self.cfg.n_pasos - 1), "canal": inf["canal"],
                "real": {k: inf[k] for k in simulado}, "simulado": simulado, "verdad": self._verdad(inf["canal"]),
                "medida": m.a_dict()})
            return inf

        def estimacion_origen(self, canal) -> Dict:
            res = super().estimacion_origen(canal)
            if self.radio is None or "error" in res:
                return res
            self._preparar()
            t = min(self.paso, self.cfg.n_pasos - 1)
            m = self.radio.medir(res["canal"])
            fila = self.medidas[t][res["canal"]]
            if m.energia_dbm <= self.cfg.umbral_energia_dbm:
                d = None
            else:
                d = fila["distancia_m"] if fila["distancia_m"] is not None else fila["distancia_falsa_m"]
            simulado = {"emisor_detectado": res["emisor_detectado"],
                        "distancia_m": res["distancia_al_primario_registrado_m"]}
            res["emisor_detectado"] = d is not None
            res["distancia_al_primario_registrado_m"] = None if d is None else round(d, 1)
            self.registro_radio["origenes"].append({
                "paso": t, "canal": res["canal"], "energia_dbm": round(m.energia_dbm, 1),
                "real": {"emisor_detectado": res["emisor_detectado"],
                         "distancia_m": res["distancia_al_primario_registrado_m"]},
                "simulado": simulado, "verdad": self._verdad(res["canal"])})
            return res

        def registrar_diagnostico(self, etiqueta, justificacion: str = "") -> Dict:
            r = super().registrar_diagnostico(etiqueta, justificacion)
            if self.radio is not None and r.get("registrado"):
                self.radio.captura.evento("diagnostico", paso=self.paso, etiqueta=r["etiqueta"])
                self.registro_radio["diagnosticos"].append({"paso": self.paso, "etiqueta": r["etiqueta"]})
            return r

        def ejecutar(self, accion, publico_extra=None) -> Dict:
            t = self.paso
            self._preparar()
            ev = super().ejecutar(accion, publico_extra)
            if self.radio is None:
                return ev
            a = ev["accion"]
            if a["accion"] == TRANSMITIR:
                r = self.radio.rafaga(a["canal"], a["potencia_dbm"], a["duracion_ms"])
            else:
                r = None
                self.radio.captura.evento("espera", paso=t)
            self.registro_radio["rafagas"].append({"paso": t, "accion": copy.deepcopy(a), "perjudicial": ev["perjudicial"],
                                                   "motivos": list(ev["motivos"]), "exito": ev["exito"], "rafaga": r})
            if self.terminado:
                self.radio.entorno([])
            return ev

    return SimuladorRadio


def crear_escudo_registrado(EscudoBase):
    class EscudoRegistrado(EscudoBase):
        """El escudo de siempre; además anota cada decisión en la grabación de la cascada."""
        captura: Optional[Captura] = None
        modo: str = "activo"           # lo fija quien ejecuta el episodio: en sombra el escudo solo anota

        def evaluar(self, accion, paso, llamadas=None):
            d = super().evaluar(accion, paso, llamadas)
            if self.captura is not None:
                self.captura.evento("escudo", paso=paso, accion=dict(accion), permitida=bool(d["permitida"]),
                                    infracciones=list(d.get("infracciones", [])), modo=self.modo)
            return d

    return EscudoRegistrado
