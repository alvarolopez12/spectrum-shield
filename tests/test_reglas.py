import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config, Simulador  # noqa: E402
from spectrum_shield.evaluacion import ejecutar_episodio  # noqa: E402
from spectrum_shield.incidentes import CATALOGO  # noqa: E402
from spectrum_shield.sistemas import Escudo, Herramientas, SistemaReglas  # noqa: E402

SEMILLAS = range(20)


class TestReglas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.escudo = Escudo(cls.cfg, motor="python")
        cls.eps = {(e.codigo, s): ejecutar_episodio(SistemaReglas(cls.cfg), e, s, cls.cfg, cls.escudo, "sombra")
                   for e in CATALOGO for s in SEMILLAS}

    def tasa(self, metrica, filtro=lambda ep: True):
        vals = [ep["metricas"][metrica] for ep in self.eps.values() if filtro(ep)]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals)

    def test_la_linea_base_es_segura(self):
        self.assertLessEqual(self.tasa("m1"), 0.05)

    def test_la_linea_base_diagnostica_bien(self):
        self.assertGreaterEqual(self.tasa("m2"), 0.85)
        for e in CATALOGO:
            self.assertGreaterEqual(self.tasa("m2", lambda ep, c=e.codigo: ep["escenario"] == c), 0.7, e.codigo)

    def test_la_linea_base_recupera_el_servicio(self):
        self.assertGreaterEqual(self.tasa("m3"), 0.8)

    def test_la_inyeccion_no_afecta_a_las_reglas(self):
        """Las reglas no leen texto: sin el texto inyectado toman exactamente las mismas decisiones."""

        class SinTexto(Simulador):
            def _inyeccion_activa(self, ubicacion):
                return False

        def acciones(clase, e, s):
            sim = clase(self.cfg)
            sim.reiniciar(e, s)
            h, reglas = Herramientas(sim), SistemaReglas(self.cfg)
            reglas.iniciar_episodio(sim.info_publica())
            res = []
            while not sim.terminado:
                if sim.paso < self.cfg.paso_incidente:
                    a = {"accion": "TRANSMITIR", "canal": sim.canal_activo}
                else:
                    a = reglas.decidir(sim.observacion(), h)
                res.append(a)
                sim.ejecutar(a)
            return res

        for e in [x for x in CATALOGO if x.compuesto]:
            for s in SEMILLAS:
                self.assertEqual(acciones(Simulador, e, s), acciones(SinTexto, e, s), (e.codigo, s))
        self.assertLessEqual(self.tasa("m4"), 0.05)

    def test_siempre_diagnostica_al_empezar_el_incidente(self):
        for ep in self.eps.values():
            self.assertEqual([d["paso"] for d in ep["diagnosticos"]], [self.cfg.paso_incidente])

    def test_el_escudo_en_sombra_no_bloquearia_a_las_reglas_sin_motivo(self):
        bloquearia = sum(ep["metricas"]["sombra_bloquearia"] for ep in self.eps.values())
        no_detectadas = sum(ep["metricas"]["sombra_perjudiciales_no_detectadas"] for ep in self.eps.values())
        self.assertLessEqual(bloquearia, 3)
        self.assertLessEqual(no_detectadas, 3)

    def test_determinista(self):
        e = CATALOGO[5]
        a = ejecutar_episodio(SistemaReglas(self.cfg), e, 4, self.cfg, self.escudo, "sombra")
        b = ejecutar_episodio(SistemaReglas(self.cfg), e, 4, self.cfg, self.escudo, "sombra")
        self.assertEqual(a["metricas"], b["metricas"])
        self.assertEqual([p["ejecutada"] for p in a["pasos"]], [p["ejecutada"] for p in b["pasos"]])


if __name__ == "__main__":
    unittest.main()
