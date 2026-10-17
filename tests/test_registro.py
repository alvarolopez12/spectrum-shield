import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config  # noqa: E402
from spectrum_shield.evaluacion import Campana, ejecutar_episodio, fila, id_sistema  # noqa: E402
from spectrum_shield.evaluacion.registro import COLUMNAS  # noqa: E402
from spectrum_shield.incidentes import escenario  # noqa: E402
from spectrum_shield.sistemas import Escudo, SistemaReglas  # noqa: E402


class TestCampana(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.campana = Campana(self.tmp.name, "prueba")

    def tearDown(self):
        self.tmp.cleanup()

    def episodio(self, codigo, semilla):
        return ejecutar_episodio(SistemaReglas(self.cfg), escenario(codigo), semilla, self.cfg,
                                 Escudo(self.cfg, motor="python"), "sombra", campana="prueba")

    def test_guardar_existe_y_tabla(self):
        for codigo, s in (("JAM_C", 0), ("PUE_S", 1)):
            ep = self.episodio(codigo, s)
            ruta = self.campana.guardar(ep)
            self.assertTrue(ruta.exists())
            self.campana.anadir_fila(ep)
        self.assertTrue(self.campana.existe("reglas", "JAM_C", 0))
        self.assertFalse(self.campana.existe("reglas", "JAM_C", 1))
        self.assertEqual(self.campana.reconstruir_csv(), 2)
        with self.campana.ruta_csv.open(newline="", encoding="utf-8") as fh:
            filas = list(csv.DictReader(fh))
        self.assertEqual(list(filas[0].keys()), COLUMNAS)
        self.assertEqual({f["escenario"] for f in filas}, {"JAM_C", "PUE_S"})
        jam = next(f for f in filas if f["escenario"] == "JAM_C")
        self.assertIn(jam["m4"], ("0", "1"))
        self.assertEqual(next(f for f in filas if f["escenario"] == "PUE_S")["m4"], "")

    def test_fila_de_un_episodio(self):
        f = fila(self.episodio("SSDF_C", 2))
        self.assertEqual(f["sistema"], "reglas")
        self.assertEqual(f["modelo"], "")
        self.assertIn(f["m1"], (0, 1))

    def test_manifiesto_y_reanudacion(self):
        self.campana.preparar_manifiesto(self.cfg, {"sistemas": ["reglas"]})
        self.campana.preparar_manifiesto(self.cfg, {"sistemas": ["reglas"]})
        man = json.loads(self.campana.ruta_manifiesto.read_text(encoding="utf-8"))
        self.assertEqual(len(man["ejecuciones"]), 2)
        otra = Config(umbral_rssi_dbm=-60.0)
        with self.assertRaises(RuntimeError):
            self.campana.preparar_manifiesto(otra, {})
        self.campana.preparar_manifiesto(otra, {}, forzar=True)

    def test_reanudar_una_campana_del_registro_anterior(self):
        import json as _json
        self.campana.dir.mkdir(parents=True, exist_ok=True)
        antiguo = {"campana": "prueba", "creada": "x", "config_huella": self.cfg.huella(), "codigo_huella": "abc",
                   "version": "0.1.0", "config": self.cfg.a_dict(), "ejecuciones": []}
        self.campana.ruta_manifiesto.write_text(_json.dumps(antiguo), encoding="utf-8")
        man = self.campana.preparar_manifiesto(self.cfg, {})
        self.assertIn("huella_trayectorias", man)
        self.campana.preparar_manifiesto(self.cfg, {})   # la segunda vez ya compara la huella añadida

    def test_huella_solo_del_codigo_que_genera_trayectorias(self):
        from spectrum_shield.evaluacion.registro import archivos_trayectorias
        arch = archivos_trayectorias()
        for f in ("entorno/simulador.py", "sistemas/escudo.py", "sistemas/agente.py", "politicas/escudo.rego",
                  "evaluacion/ejecutor.py", "__init__.py"):
            self.assertIn(f, arch)
        for f in ("evaluacion/metricas.py", "evaluacion/estadistica.py", "evaluacion/registro.py",
                  "politicas/escudo_test.rego"):
            self.assertNotIn(f, arch)

    def test_la_tabla_recalcula_las_metricas(self):
        ep = self.episodio("JAM_S", 0)
        ep["metricas"]["m2"] = "valor antiguo"
        self.campana.guardar(ep)
        self.campana.reconstruir_csv()
        with self.campana.ruta_csv.open(newline="", encoding="utf-8") as fh:
            f = next(csv.DictReader(fh))
        self.assertIn(f["m2"], ("0", "1"))

    def test_id_sistema(self):
        self.assertEqual(id_sistema("reglas", None), "reglas")
        self.assertEqual(id_sistema("agente_escudo", "qwen3.8:27b"), "agente_escudo__qwen3.8-27b")


class TestScripts(unittest.TestCase):
    def test_campana_de_reglas_y_resumen(self):
        with tempfile.TemporaryDirectory() as d:
            orden = [sys.executable, str(RAIZ / "scripts" / "ejecutar.py"), "--campana", "t", "--sistemas",
                     "reglas", "--semillas", "0-1", "--salida", d, "--motor-escudo", "python"]
            r = subprocess.run(orden, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("16 pendientes", r.stdout)
            r = subprocess.run(orden, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("0 pendientes", r.stdout)
            with (Path(d) / "t" / "episodios.csv").open(newline="", encoding="utf-8") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 16)
            r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "resumen.py"), "--campana", "t",
                                "--salida", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("M1 perjudicial", r.stdout)
            self.assertTrue((Path(d) / "t" / "tasas.csv").exists())

    def test_sistema_desconocido(self):
        r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "ejecutar.py"), "--sistemas", "magia"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
