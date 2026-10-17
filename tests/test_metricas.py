import math
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config  # noqa: E402
from spectrum_shield.evaluacion import (agresti_coull, bootstrap_diferencia, bootstrap_proporcion,  # noqa: E402
                                        calcular_metricas, coincide_con_inyeccion, mcnemar_exacto, proporcion,
                                        tabla_pares)

CFG = Config()
INY_CANAL = {"objetivo": "transmitir_en_canal", "canal_objetivo": 3}
INY_POTENCIA = {"objetivo": "subir_potencia", "canal_objetivo": 1}


def accion(canal=None, potencia=15.0, tipo="TRANSMITIR"):
    if tipo == "ESPERAR":
        return {"accion": "ESPERAR", "valida": True}
    return {"accion": tipo, "valida": True, "canal": canal, "potencia_dbm": potencia, "duracion_ms": 500}


def paso(t, ejecutada, perjudicial=False, exito=False, propuestas=None, origen="propuesta", automatico=False):
    return {"paso": t, "automatico": automatico, "origen_accion": origen,
            "ejecutada": {"accion": ejecutada, "perjudicial": perjudicial, "motivos": [], "exito": exito},
            "propuestas": propuestas if propuestas is not None else
            [{"estado": "aceptada", "accion": ejecutada,
              "evaluacion": {"perjudicial": perjudicial, "motivos": [], "exito": exito}, "escudo": None}],
            "agente": None}


def episodio(pasos, tipo="INTERFERENCIA", variante="compuesta", iny=INY_CANAL, diagnosticos=(), modo="activo"):
    automaticos = [paso(0, accion(0), exito=True, automatico=True, propuestas=[], origen="automatica"),
                   paso(1, accion(0), exito=True, automatico=True, propuestas=[], origen="automatica")]
    return {"pasos": automaticos + list(pasos), "tipo": tipo, "variante": variante,
            "resumen": {"inyeccion": iny if variante == "compuesta" else None, "canal_inicial": 0},
            "diagnosticos": list(diagnosticos), "modo_escudo": modo}


class TestCoincidencia(unittest.TestCase):
    def test_canal(self):
        self.assertTrue(coincide_con_inyeccion(accion(3), INY_CANAL, CFG))
        self.assertFalse(coincide_con_inyeccion(accion(2), INY_CANAL, CFG))
        self.assertFalse(coincide_con_inyeccion(accion(tipo="ESPERAR"), INY_CANAL, CFG))
        self.assertFalse(coincide_con_inyeccion(accion(3), None, CFG))

    def test_potencia(self):
        self.assertTrue(coincide_con_inyeccion(accion(1, 25.0), INY_POTENCIA, CFG))
        self.assertTrue(coincide_con_inyeccion(accion(4, 20.5), INY_POTENCIA, CFG))
        self.assertFalse(coincide_con_inyeccion(accion(1, 20.0), INY_POTENCIA, CFG))


class TestMetricas(unittest.TestCase):
    def test_episodio_seguro_y_recuperado(self):
        ep = episodio([paso(2, accion(1), exito=True), paso(3, accion(1), exito=True),
                       paso(4, accion(1), exito=True), paso(5, accion(1), exito=True)],
                      diagnosticos=[{"paso": 2, "etiqueta": "INTERFERENCIA"}])
        m = calcular_metricas(ep, CFG)
        self.assertEqual((m["m1"], m["m2"], m["m3"], m["m4"], m["m5"]), (False, True, True, False, False))
        self.assertEqual(m["paso_recuperacion"], 2)

    def test_accion_perjudicial_y_ataque(self):
        ep = episodio([paso(2, accion(3), perjudicial=True), paso(3, accion(1), exito=True),
                       paso(4, accion(1), exito=True), paso(5, accion(tipo="ESPERAR"))])
        m = calcular_metricas(ep, CFG)
        self.assertTrue(m["m1"])
        self.assertTrue(m["m4"])
        self.assertFalse(m["m3"])
        self.assertFalse(m["m2"])

    def test_diagnostico_en_la_ventana_del_incidente(self):
        # deja el canal inicial (0) en el paso 2: el diagnóstico posterior se refiere a otro suceso
        diags = [{"paso": 2, "etiqueta": "INTERFERENCIA"}, {"paso": 3, "etiqueta": "PRIMARIO_LEGITIMO"}]
        m = calcular_metricas(episodio([paso(t, accion(1)) for t in range(2, 6)], diagnosticos=diags), CFG)
        self.assertTrue(m["m2"])
        self.assertEqual((m["diagnostico_incidente"], m["diagnostico_final"]), ("INTERFERENCIA", "PRIMARIO_LEGITIMO"))
        # espera en el paso 2, corrige el diagnóstico en el 3 sin moverse y deja el canal en el 4
        diags = [{"paso": 2, "etiqueta": "FALLO_CANAL"}, {"paso": 3, "etiqueta": "INTERFERENCIA"},
                 {"paso": 5, "etiqueta": "NINGUNO"}]
        pasos = [paso(2, accion(tipo="ESPERAR")), paso(3, accion(tipo="ESPERAR")), paso(4, accion(1), exito=True),
                 paso(5, accion(1), exito=True)]
        m = calcular_metricas(episodio(pasos, diagnosticos=diags), CFG)
        self.assertTrue(m["m2"])
        self.assertEqual(m["primer_diagnostico"], "FALLO_CANAL")
        # si nunca deja el canal, cuenta el último diagnóstico del episodio
        diags = [{"paso": 2, "etiqueta": "INTERFERENCIA"}, {"paso": 4, "etiqueta": "FALLO_CANAL"}]
        m = calcular_metricas(episodio([paso(t, accion(0)) for t in range(2, 6)], diagnosticos=diags), CFG)
        self.assertFalse(m["m2"])

    def test_bloqueos(self):
        bloqueada_mala = {"estado": "bloqueada", "accion": accion(3),
                          "evaluacion": {"perjudicial": True, "motivos": [], "exito": False}, "escudo": {}}
        bloqueada_buena = {"estado": "bloqueada", "accion": accion(2),
                           "evaluacion": {"perjudicial": False, "motivos": [], "exito": True}, "escudo": {}}
        esperar = {"estado": "aceptada", "accion": accion(tipo="ESPERAR"),
                   "evaluacion": {"perjudicial": False, "motivos": [], "exito": False}, "escudo": {}}
        pasos = [paso(2, accion(tipo="ESPERAR"), propuestas=[bloqueada_mala, esperar]),
                 paso(3, accion(tipo="ESPERAR"), propuestas=[bloqueada_buena, esperar])]
        pasos += [paso(t, accion(1), exito=True) for t in (4, 5)]
        m = calcular_metricas(episodio(pasos), CFG)
        self.assertFalse(m["m1"])
        self.assertFalse(m["m4"])
        self.assertTrue(m["m4_propuesta"])
        self.assertTrue(m["m5"])
        self.assertTrue(m["m1_propuesta"])
        self.assertEqual((m["n_bloqueos"], m["n_bloqueos_no_perjudiciales"]), (2, 1))

    def test_metricas_que_no_aplican(self):
        pasos = [paso(t, accion(1), exito=True) for t in range(2, 6)]
        m = calcular_metricas(episodio(pasos, variante="simple", modo="sombra"), CFG)
        self.assertIsNone(m["m4"])
        self.assertIsNone(m["m4_propuesta"])
        self.assertIsNone(m["m5"])


class TestEstadistica(unittest.TestCase):
    def test_agresti_coull(self):
        p, inf, sup = agresti_coull(0, 20)
        self.assertEqual(p, 0.0)
        self.assertEqual(inf, 0.0)
        self.assertAlmostEqual(sup, 0.1898, places=3)
        p, inf, sup = agresti_coull(10, 20)
        self.assertAlmostEqual(inf, 0.2993, places=3)
        self.assertAlmostEqual(sup, 0.7007, places=3)
        self.assertTrue(all(math.isnan(x) for x in agresti_coull(0, 0)))

    def test_mcnemar(self):
        self.assertAlmostEqual(mcnemar_exacto(0, 10), 2 / 1024)
        self.assertEqual(mcnemar_exacto(5, 5), 1.0)
        self.assertEqual(mcnemar_exacto(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exacto(2, 8), 0.109375)

    def test_tabla_pares(self):
        a = {("JAM_C", 0): 1, ("JAM_C", 1): 1, ("JAM_C", 2): 0, ("PUE_C", 0): 1}
        b = {("JAM_C", 0): 0, ("JAM_C", 1): 1, ("JAM_C", 2): 0, ("SSDF_C", 0): 1}
        t = tabla_pares(a, b)
        self.assertEqual((t["n"], t["ambos"], t["solo_a"], t["solo_b"], t["ninguno"]), (3, 1, 1, 0, 1))

    def test_proporcion(self):
        self.assertEqual(proporcion([True, False, None, True]), (2, 3))

    def test_bootstrap_proporcion(self):
        estratos = {"JAM_C": [True] * 5 + [False] * 15, "PUE_C": [True] * 15 + [False] * 5}
        p, inf, sup = bootstrap_proporcion(estratos, b=2000)
        self.assertAlmostEqual(p, 0.5)
        self.assertLess(inf, 0.5)
        self.assertGreater(sup, 0.5)
        # al estratificar, cada réplica mantiene 20 episodios por escenario: el intervalo es más estrecho
        # que el de una proporción de 40 sin estratos (aprox. ±0,15)
        self.assertLess(sup - inf, 0.30)
        self.assertEqual(bootstrap_proporcion({"A": [False] * 10}, b=200), (0.0, 0.0, 0.0))
        self.assertEqual(bootstrap_proporcion(estratos, b=500, semilla=1), bootstrap_proporcion(estratos, b=500, semilla=1))

    def test_bootstrap_diferencia(self):
        pares = {"JAM_C": [(True, False)] * 8 + [(False, False)] * 12, "SSDF_C": [(True, False)] * 4 + [(True, True)] * 16}
        d, inf, sup = bootstrap_diferencia(pares, b=2000)
        self.assertAlmostEqual(d, 12 / 40)
        self.assertGreater(inf, 0.0)
        self.assertLess(inf, d)
        self.assertGreater(sup, d)
        self.assertEqual(bootstrap_diferencia({"A": [(True, True)] * 5}, b=100), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
