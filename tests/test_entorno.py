import json
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config, Simulador  # noqa: E402
from spectrum_shield.incidentes import CATALOGO, escenario  # noqa: E402
from spectrum_shield.incidentes import catalogo as cat  # noqa: E402

SEMILLAS = range(20)


def nuevo(codigo, semilla, cfg=None):
    sim = Simulador(cfg or Config())
    sim.reiniciar(escenario(codigo), semilla)
    return sim


class TestReproducibilidad(unittest.TestCase):
    def test_misma_semilla_mismo_episodio(self):
        for e in CATALOGO:
            a, b = nuevo(e.codigo, 7), nuevo(e.codigo, 7)
            self.assertEqual(a.medidas, b.medidas)
            self.assertEqual(a.estado_real(3), b.estado_real(3))
            self.assertEqual(a.inyeccion, b.inyeccion)
            self.assertEqual((a.c0, a.refugio, a.bizantinos), (b.c0, b.refugio, b.bizantinos))

    def test_semillas_distintas_dan_episodios_distintos(self):
        a, b = nuevo("JAM_S", 1), nuevo("JAM_S", 2)
        self.assertNotEqual(a.medidas, b.medidas)

    def test_las_acciones_no_cambian_lo_que_se_mide(self):
        """Episodios emparejados: lo que haga un sistema no altera las medidas que verá otro."""
        a, b = nuevo("PUE_C", 3), nuevo("PUE_C", 3)
        for t in range(a.cfg.n_pasos):
            self.assertEqual(a.informe_sensado(1), b.informe_sensado(1))
            self.assertEqual(a.estimacion_origen(2), b.estimacion_origen(2))
            a.ejecutar({"accion": "TRANSMITIR", "canal": t % 5})
            b.ejecutar({"accion": "ESPERAR"})


class TestEscenarios(unittest.TestCase):
    def test_incidente_en_el_canal_inicial_y_refugio_libre(self):
        marca = {cat.INTERFERENCIA: "interferencia", cat.EMULACION_PRIMARIO: "emulador",
                 cat.FALSIFICACION_SENSADO: "primario", cat.FALLO_CANAL: "degradado"}
        for e in CATALOGO:
            for s in SEMILLAS:
                sim = nuevo(e.codigo, s)
                t0 = sim.cfg.paso_incidente
                for t in range(sim.cfg.n_pasos):
                    c0 = sim.guion[t][sim.c0]
                    self.assertEqual(getattr(c0, marca[e.tipo]), t >= t0, (e.codigo, s, t))
                    if t < t0:
                        self.assertTrue(c0.apto())
                    self.assertTrue(sim.guion[t][sim.refugio].apto(), (e.codigo, s, t))
                self.assertIn(sim.c0, sim.cfg.canales_autorizados)
                self.assertIn(sim.refugio, sim.cfg.canales_autorizados)

    def test_inyeccion_solo_en_compuestos(self):
        for e in CATALOGO:
            for s in SEMILLAS:
                sim = nuevo(e.codigo, s)
                if not e.compuesto:
                    self.assertIsNone(sim.inyeccion)
                    continue
                iny = sim.inyeccion
                self.assertIsNotNone(iny)
                if iny["objetivo"] == "transmitir_en_canal":
                    self.assertIn(f"canal {iny['canal_objetivo']}", iny["texto"])

    def test_inyeccion_aparece_desde_el_incidente_y_en_su_campo(self):
        for e in [x for x in CATALOGO if x.compuesto]:
            for s in SEMILLAS:
                sim = nuevo(e.codigo, s)
                texto = sim.inyeccion["texto"]
                for t in range(sim.cfg.n_pasos):
                    obs = sim.observacion()
                    en_notas = obs["sensado_canal_activo"]["notas"] == texto
                    en_mensajes = texto in obs["mensajes"] and texto in sim.historial_publico()["registro"]
                    activa = t >= sim.cfg.paso_incidente
                    self.assertEqual(en_notas, activa and sim.inyeccion["ubicacion"] == "notas_sensado")
                    self.assertEqual(en_mensajes, activa and sim.inyeccion["ubicacion"] == "mensaje_controlador")
                    sim.ejecutar({"accion": "ESPERAR"})

    def test_objetivo_de_la_inyeccion(self):
        for e in [x for x in CATALOGO if x.compuesto]:
            for s in SEMILLAS:
                sim = nuevo(e.codigo, s)
                iny, t0 = sim.inyeccion, sim.cfg.paso_incidente
                obj = iny["canal_objetivo"]
                if iny["rol_canal"] == "con_primario":
                    if e.tipo == cat.FALSIFICACION_SENSADO:
                        self.assertEqual(obj, sim.c0)
                    else:
                        self.assertNotIn(obj, (sim.c0, sim.refugio))
                    for t in range(t0, sim.cfg.n_pasos):
                        self.assertTrue(sim.guion[t][obj].primario)
                elif iny["rol_canal"] == "no_autorizado":
                    self.assertNotIn(obj, sim.cfg.canales_autorizados)
                else:
                    self.assertEqual(obj, sim.c0)

    def test_bizantinos_solo_en_falsificacion(self):
        for e in CATALOGO:
            sim = nuevo(e.codigo, 0)
            esperado = sim.cfg.n_bizantinos if e.tipo == cat.FALSIFICACION_SENSADO else 0
            self.assertEqual(len(sim.bizantinos), esperado)

    def test_centro_de_fusion_cegado_en_falsificacion(self):
        libres = 0
        for s in range(50):
            sim = nuevo("SSDF_S", s)
            sim.paso = sim.cfg.paso_incidente
            libres += sim.informe_sensado(sim.c0)["decision_fusion"] == "libre"
        self.assertGreaterEqual(libres, 40)


class TestMedidas(unittest.TestCase):
    def test_estimar_origen_requiere_detectar_energia(self):
        for e in CATALOGO:
            for s in range(10):
                sim = nuevo(e.codigo, s)
                for t in range(sim.cfg.n_pasos):
                    sim.paso = t
                    for c in range(sim.cfg.n_canales):
                        origen = sim.estimacion_origen(c)
                        detecta = sim.informe_sensado(c)["deteccion_local"] == "ocupado"
                        self.assertEqual(origen["emisor_detectado"], detecta)
                        self.assertEqual(origen["distancia_al_primario_registrado_m"] is None, not detecta)

    def test_canal_no_valido(self):
        sim = nuevo("JAM_S", 0)
        for c in (-1, 6, "3", None):
            self.assertIn("error", sim.informe_sensado(c))
            self.assertIn("error", sim.estimacion_origen(c))

    def test_enlace_desaparece_al_esperar(self):
        sim = nuevo("JAM_S", 0)
        self.assertIsNotNone(sim.observacion()["enlace"])
        sim.ejecutar({"accion": "ESPERAR"})
        obs = sim.observacion()
        self.assertIsNone(obs["enlace"])
        self.assertFalse(obs["transmitiendo"])
        self.assertEqual(obs["ultima_accion"]["accion"], "ESPERAR")


class TestAcciones(unittest.TestCase):
    def setUp(self):
        self.sim = nuevo("JAM_S", 0)
        self.libre = self.sim.refugio

    def test_normalizar(self):
        n = self.sim.normalizar
        self.assertTrue(n({"accion": "esperar"})["valida"])
        self.assertEqual(n({"accion": "TRANSMITIR", "canal": 3.0})["canal"], 3)
        a = n({"accion": "transmitir", "canal": "2"})
        self.assertEqual((a["canal"], a["potencia_dbm"], a["duracion_ms"]), (2, 15.0, 500))
        for mala in (None, {}, {"accion": "SALTAR"}, {"accion": "TRANSMITIR"},
                     {"accion": "TRANSMITIR", "canal": 9}, {"accion": "TRANSMITIR", "canal": 1, "potencia_dbm": "x"},
                     {"accion": "TRANSMITIR", "canal": 1, "duracion_ms": 0},
                     {"accion": "TRANSMITIR", "canal": 2.5}, {"accion": "TRANSMITIR", "canal": float("nan")},
                     {"accion": "TRANSMITIR", "canal": 1, "potencia_dbm": float("nan")},
                     {"accion": "TRANSMITIR", "canal": 1, "duracion_ms": float("inf")}):
            r = n(mala)
            self.assertFalse(r["valida"], mala)
            self.assertEqual(r["accion"], "ESPERAR")

    def test_motivos_de_perjuicio(self):
        ev = self.sim.evaluar_accion
        self.assertEqual(ev({"accion": "TRANSMITIR", "canal": self.libre})["motivos"], [])
        self.assertTrue(ev({"accion": "TRANSMITIR", "canal": self.libre})["exito"])
        self.assertIn("banda_no_autorizada", ev({"accion": "TRANSMITIR", "canal": 5})["motivos"])
        self.assertIn("potencia_excesiva",
                      ev({"accion": "TRANSMITIR", "canal": self.libre, "potencia_dbm": 25})["motivos"])
        self.assertIn("duracion_excesiva",
                      ev({"accion": "TRANSMITIR", "canal": self.libre, "duracion_ms": 3000})["motivos"])
        self.assertFalse(ev({"accion": "ESPERAR"})["perjudicial"])

    def test_transmitir_sobre_primario_es_perjudicial(self):
        sim = nuevo("SSDF_S", 0)
        sim.paso = sim.cfg.paso_incidente
        r = sim.evaluar_accion({"accion": "TRANSMITIR", "canal": sim.c0})
        self.assertTrue(r["perjudicial"])
        self.assertIn("transmision_sobre_primario", r["motivos"])

    def test_interferencia_no_es_perjudicial_pero_no_tiene_exito(self):
        sim = nuevo("JAM_S", 0)
        sim.paso = sim.cfg.paso_incidente
        r = sim.evaluar_accion({"accion": "TRANSMITIR", "canal": sim.c0})
        self.assertFalse(r["perjudicial"])
        self.assertFalse(r["exito"])


class TestConfig(unittest.TestCase):
    def test_json_del_repositorio_coincide_con_los_valores_por_defecto(self):
        cfg = Config.desde_json(RAIZ / "config" / "experimento.json")
        self.assertEqual(cfg, Config())
        self.assertEqual(cfg.huella(), Config().huella())

    def test_parametro_desconocido(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "c.json"
            ruta.write_text(json.dumps({"n_canales": 6, "inventado": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                Config.desde_json(ruta)

    def test_validar(self):
        with self.assertRaises(ValueError):
            Simulador(Config(paso_incidente=0))
        with self.assertRaises(ValueError):
            Simulador(Config(canales_autorizados=[0, 1]))


if __name__ == "__main__":
    unittest.main()
