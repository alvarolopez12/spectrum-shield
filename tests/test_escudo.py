import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config, Simulador  # noqa: E402
from spectrum_shield.incidentes import CATALOGO, escenario  # noqa: E402
from spectrum_shield.sistemas import Escudo, Herramientas, decidir_python  # noqa: E402
from spectrum_shield.sistemas.escudo import RUTA_POLITICA, MotorOPA  # noqa: E402

HAY_OPA = shutil.which("opa") is not None
POLITICA = {"canales_autorizados": [0, 1, 2, 3, 4], "potencia_max_dbm": 20.0, "duracion_max_ms": 1000,
            "umbral_distancia_m": 100.0}


def sensado(canal, paso, propia="libre", firma=False, fusion="libre"):
    return {"tipo": "sensado", "fuente": "consultar_sensado", "paso": paso, "canal": canal,
            "deteccion_local": propia, "energia_dbm": -60.0 if propia == "ocupado" else -95.0,
            "firma_primario": firma, "decision_fusion": fusion, "votos_ocupado": 5 if fusion == "ocupado" else 0}


def origen(canal, paso, distancia):
    return {"tipo": "origen", "fuente": "estimar_origen", "paso": paso, "canal": canal,
            "emisor_detectado": distancia is not None, "distancia_m": distancia}


def tx(canal, potencia=15.0, duracion=500):
    return {"accion": "TRANSMITIR", "canal": canal, "potencia_dbm": potencia, "duracion_ms": duracion}


def decide(accion, evidencias, paso=3):
    return decidir_python({"accion": accion, "paso": paso, "politica": POLITICA, "evidencias": evidencias})


class TestPoliticaPython(unittest.TestCase):
    def test_permitida(self):
        d = decide(tx(2), [sensado(2, 3)])
        self.assertTrue(d["permitida"])
        self.assertEqual(d["infracciones"], [])

    def test_esperar(self):
        self.assertTrue(decide({"accion": "ESPERAR"}, [])["permitida"])

    def test_invariantes_de_politica(self):
        self.assertEqual(decide(tx(2, potencia=25.0), [sensado(2, 3)])["infracciones"], ["I2"])
        self.assertEqual(decide(tx(5), [sensado(5, 3)])["infracciones"], ["I3"])
        self.assertEqual(decide(tx(2, duracion=3000), [sensado(2, 3)])["infracciones"], ["I4"])
        self.assertEqual(decide(tx(2, potencia=20.0, duracion=1000), [sensado(2, 3)])["infracciones"], [])

    def test_i5(self):
        self.assertEqual(decide(tx(2), [sensado(1, 3)])["infracciones"], ["I5"])
        self.assertEqual(decide(tx(2), [sensado(2, 2)])["infracciones"], ["I5"])

    def test_i1(self):
        self.assertEqual(decide(tx(2), [sensado(2, 3, "ocupado", True, "ocupado")])["infracciones"], ["I1"])
        self.assertEqual(decide(tx(2), [sensado(2, 3, "ocupado")])["infracciones"], ["I1"])
        self.assertEqual(decide(tx(2), [sensado(2, 3, firma=True)])["infracciones"], ["I1"])
        self.assertEqual(decide(tx(2), [sensado(2, 3, fusion="ocupado")])["infracciones"], ["I1"])

    def test_i1_y_verificacion_de_origen(self):
        ocupado = sensado(2, 3, "ocupado", True, "ocupado")
        self.assertTrue(decide(tx(2), [ocupado, origen(2, 3, 450.0)])["permitida"])
        self.assertFalse(decide(tx(2), [ocupado, origen(2, 3, 25.0)])["permitida"])
        self.assertFalse(decide(tx(2), [ocupado, origen(2, 2, 450.0)])["permitida"])  # origen de otro paso
        self.assertFalse(decide(tx(2), [ocupado, origen(2, 3, None)])["permitida"])

    def test_contradiccion_fuerte(self):
        # la medida propia ve un primario (energía y firma) que el centro de fusión da por libre
        d = decide(tx(2), [sensado(0, 2, "ocupado", True, "libre"), sensado(2, 3, fusion="ocupado")])
        self.assertTrue(d["permitida"])
        self.assertFalse(d["cooperativa_fiable"])
        # la medida propia sigue mandando aunque no se confíe en el centro de fusión
        d = decide(tx(2), [sensado(0, 2, "ocupado", True, "libre"), sensado(2, 3, "ocupado", True, "libre")])
        self.assertEqual(d["infracciones"], ["I1"])

    def test_contradiccion_sistematica(self):
        debil_libre = sensado(0, 2, "ocupado", False, "libre")    # energía sin firma, la fusión dice libre
        inv_ocupado = sensado(1, 2, fusion="ocupado")             # canal libre, la fusión dice ocupado
        objetivo = sensado(2, 3, fusion="ocupado")
        d = decide(tx(2), [debil_libre, inv_ocupado, sensado(3, 2), objetivo])
        self.assertTrue(d["permitida"])
        self.assertFalse(d["cooperativa_fiable"])
        # discrepancias aisladas no bastan
        d = decide(tx(2), [debil_libre, sensado(1, 2), sensado(3, 2), sensado(4, 2), objetivo])
        self.assertEqual(d["infracciones"], ["I1"])
        self.assertTrue(d["cooperativa_fiable"])
        # hacen falta al menos tres lecturas
        self.assertTrue(decide(tx(2), [debil_libre, objetivo])["cooperativa_fiable"])
        # una lectura repetida cuenta una vez
        self.assertTrue(decide(tx(1), [debil_libre] * 3 + [sensado(1, 2), sensado(3, 2)])["cooperativa_fiable"])

    def test_la_desconfianza_se_mantiene_en_el_episodio(self):
        # en el paso 2 se compromete la fuente; las lecturas concordantes del paso 3 no la rehabilitan
        evidencias = [sensado(0, 2, "ocupado", False, "libre"), sensado(1, 2, fusion="ocupado"), sensado(3, 2)]
        evidencias += [sensado(c, 3) for c in (0, 1, 3, 4)] + [sensado(2, 3, fusion="ocupado")]
        self.assertFalse(decide(tx(2), evidencias)["cooperativa_fiable"])

    def test_todas_las_infracciones_ordenadas(self):
        self.assertEqual(decide(tx(5, 30.0, 3000), [])["infracciones"], ["I2", "I3", "I4", "I5"])


class TestEscudoEnEpisodio(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.sim = Simulador(self.cfg)
        self.sim.reiniciar(escenario("JAM_C"), 0)
        self.h = Herramientas(self.sim)
        self.escudo = Escudo(self.cfg, motor="python")

    def test_la_observacion_cuenta_como_sensado_del_canal_activo(self):
        obs = self.sim.observacion()
        self.escudo.observar(obs)
        accion = self.sim.normalizar({"accion": "TRANSMITIR", "canal": self.sim.canal_activo})
        d = self.escudo.evaluar(accion, self.sim.paso, self.h.llamadas)
        self.assertNotIn("I5", d["infracciones"])

    def test_sensado_pedido_en_el_paso(self):
        self.escudo.observar(self.sim.observacion())
        accion = self.sim.normalizar({"accion": "TRANSMITIR", "canal": self.sim.refugio})
        self.assertIn("I5", self.escudo.evaluar(accion, self.sim.paso, self.h.llamadas)["infracciones"])
        self.h.consultar_sensado(self.sim.refugio)
        self.assertTrue(self.escudo.evaluar(accion, self.sim.paso, self.h.llamadas)["permitida"])
        # en el paso siguiente el sensado anterior ya no vale
        self.sim.ejecutar({"accion": "ESPERAR"})
        self.escudo.observar(self.sim.observacion())
        self.assertIn("I5", self.escudo.evaluar(accion, self.sim.paso, self.h.llamadas)["infracciones"])

    def test_el_texto_libre_no_entra_en_la_politica(self):
        for e in [x for x in CATALOGO if x.compuesto]:
            for s in range(10):
                sim = Simulador(self.cfg)
                sim.reiniciar(e, s)
                h = Herramientas(sim)
                escudo = Escudo(self.cfg, motor="python")
                sim.paso = self.cfg.paso_incidente
                escudo.observar(sim.observacion())
                for c in range(self.cfg.n_canales):
                    h.consultar_sensado(c)
                    h.estimar_origen(c)
                h.consultar_historial()
                escudo._sincronizar(h.llamadas)
                entrada = escudo.entrada(sim.normalizar({"accion": "TRANSMITIR", "canal": 1}), sim.paso)
                texto = json.dumps(entrada, ensure_ascii=False)
                self.assertNotIn(sim.inyeccion["texto"], texto)
                self.assertNotIn("notas", texto)
                self.assertNotIn("mensajes", texto)

    def test_motivo_legible(self):
        self.escudo.observar(self.sim.observacion())
        accion = self.sim.normalizar({"accion": "TRANSMITIR", "canal": 5, "potencia_dbm": 30})
        d = self.escudo.evaluar(accion, self.sim.paso, self.h.llamadas)
        self.assertFalse(d["permitida"])
        for i in d["infracciones"]:
            self.assertIn(i, d["motivo"])


def casos_aleatorios(n_semillas=10, por_paso=8):
    """Entradas de la política a partir de episodios reales, con acciones y evidencias aleatorias."""
    cfg = Config()
    rng = random.Random(2026)
    casos = []
    for e in CATALOGO:
        for s in range(n_semillas):
            sim = Simulador(cfg)
            sim.reiniciar(e, s)
            h = Herramientas(sim)
            escudo = Escudo(cfg, motor="python")
            while not sim.terminado:
                escudo.observar(sim.observacion())
                for c in rng.sample(range(cfg.n_canales), rng.randint(0, cfg.n_canales)):
                    h.consultar_sensado(c)
                    if rng.random() < 0.5:
                        h.estimar_origen(c)
                escudo._sincronizar(h.llamadas)
                for _ in range(por_paso):
                    if rng.random() < 0.1:
                        accion = {"accion": "ESPERAR"}
                    else:
                        accion = {"accion": "TRANSMITIR", "canal": rng.randrange(cfg.n_canales),
                                  "potencia_dbm": rng.choice([10.0, 15.0, 20.0, 20.5, 25.0, 30.0]),
                                  "duracion_ms": rng.choice([100, 500, 1000, 1001, 3000])}
                    entrada = escudo.entrada(accion, sim.paso)
                    # se quitan evidencias al azar para cubrir más combinaciones
                    entrada["evidencias"] = [x for x in entrada["evidencias"] if rng.random() < 0.8]
                    casos.append(entrada)
                sim.ejecutar({"accion": "TRANSMITIR", "canal": rng.randrange(5)})
    return casos


@unittest.skipUnless(HAY_OPA, "OPA no está instalado (brew install opa)")
class TestPoliticaRego(unittest.TestCase):
    def test_pruebas_de_la_politica(self):
        r = subprocess.run(["opa", "test", str(RUTA_POLITICA.parent)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_rego_y_python_deciden_igual(self):
        casos = casos_aleatorios()
        lote = ("package spectrum_shield.lote\n\nimport rego.v1\n\n"
                "resultados := [d | some caso in input.casos\n"
                "\td := data.spectrum_shield.escudo.decision with input as caso]\n")
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "lote.rego"
            ruta.write_text(lote, encoding="utf-8")
            motor = MotorOPA(timeout=300)
            rego = motor.evaluar("data.spectrum_shield.lote.resultados", {"casos": casos}, datos=[ruta])
        self.assertEqual(len(rego), len(casos))
        cobertura = {}
        for caso, dr in zip(casos, rego):
            dp = decidir_python(caso)
            self.assertEqual(dr, dp, json.dumps(caso)[:2000])
            for i in dp["infracciones"]:
                cobertura[i] = cobertura.get(i, 0) + 1
            cobertura[f"fiable={dp['cooperativa_fiable']}"] = cobertura.get(f"fiable={dp['cooperativa_fiable']}", 0) + 1
            cobertura[f"permitida={dp['permitida']}"] = cobertura.get(f"permitida={dp['permitida']}", 0) + 1
        for clave in ("I1", "I2", "I3", "I4", "I5", "fiable=True", "fiable=False", "permitida=True",
                      "permitida=False"):
            self.assertGreater(cobertura.get(clave, 0), 10, clave)

    def test_motor_opa_en_episodio(self):
        cfg = Config()
        sim = Simulador(cfg)
        sim.reiniciar(escenario("SSDF_C"), 1)
        h = Herramientas(sim)
        e_opa, e_py = Escudo(cfg, motor="opa"), Escudo(cfg, motor="python")
        while not sim.terminado:
            obs = sim.observacion()
            e_opa.observar(obs)
            e_py.observar(obs)
            for c in range(cfg.n_canales):
                h.consultar_sensado(c)
            for c in range(cfg.n_canales):
                a = sim.normalizar({"accion": "TRANSMITIR", "canal": c})
                self.assertEqual(e_opa.evaluar(a, sim.paso, h.llamadas)["infracciones"],
                                 e_py.evaluar(a, sim.paso, h.llamadas)["infracciones"])
            sim.ejecutar({"accion": "ESPERAR"})


if __name__ == "__main__":
    unittest.main()
