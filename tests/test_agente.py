import json
import re
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config, Simulador  # noqa: E402
from spectrum_shield.evaluacion import ejecutar_episodio  # noqa: E402
from spectrum_shield.incidentes import escenario  # noqa: E402
from spectrum_shield.sistemas import AgenteLLM, ClienteOllama, ErrorOllama, Escudo, Herramientas  # noqa: E402


def llamada(nombre, **args):
    return {"function": {"name": nombre, "arguments": args}}


def respuesta(*llamadas, texto=""):
    msg = {"role": "assistant", "content": texto}
    if llamadas:
        msg["tool_calls"] = list(llamadas)
    return {"message": msg, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20,
            "prompt_eval_duration": 10 ** 8, "eval_duration": 10 ** 9, "load_duration": 0}


class ClienteGuion:
    """Devuelve respuestas fijadas de antemano, en orden."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.peticiones = []

    def chat(self, modelo, mensajes, herramientas=None, **kw):
        self.peticiones.append({"mensajes": json.loads(json.dumps(mensajes)), "kw": kw})
        return self.respuestas.pop(0)


class ClienteIngenuo:
    """Agente falso que obedece cualquier instrucción con un número de canal que aparezca en los datos.

    Si no encuentra ninguna, sensa todos los canales a la vez y transmite en el que tenga menos
    energía de los que parecen libres. Si le bloquean una acción, espera.
    """

    def chat(self, modelo, mensajes, herramientas=None, **kw):
        idx = max(i for i, m in enumerate(mensajes) if m["role"] == "user" and m["content"].startswith("Paso"))
        obs = json.loads(mensajes[idx]["content"].split("Observación:\n", 1)[1].rsplit("\nDecide", 1)[0])
        despues = mensajes[idx + 1:]
        resultados = [json.loads(m["content"]) for m in despues if m["role"] == "tool"]
        if any(r.get("resultado") == "bloqueada" for r in resultados):
            return respuesta(llamada("ejecutar_accion", accion="ESPERAR"))
        textos = list(obs["mensajes"]) + [obs["sensado_canal_activo"]["notas"]]
        for t in textos:
            m = re.search(r"canal (\d+)", t)
            if m:
                return respuesta(llamada("ejecutar_accion", accion="TRANSMITIR", canal=int(m.group(1))))
        sensados = [r for r in resultados if "decision_fusion" in r]
        if not sensados:
            return respuesta(*[llamada("consultar_sensado", canal=c) for c in range(5)])
        libres = [r for r in sensados if r["decision_fusion"] == "libre" and r["deteccion_local"] == "libre"
                  and not r["firma_primario"]]
        if not libres:
            return respuesta(llamada("ejecutar_accion", accion="ESPERAR"))
        mejor = min(libres, key=lambda r: r["energia_dbm"])
        return respuesta(llamada("ejecutar_accion", accion="TRANSMITIR", canal=mejor["canal"]))


def preparar(codigo="JAM_S", semilla=0):
    cfg = Config()
    sim = Simulador(cfg)
    sim.reiniciar(escenario(codigo), semilla)
    sim.paso = cfg.paso_incidente
    return cfg, sim, Herramientas(sim)


def aceptar_todo(sim):
    def validar(p):
        n = sim.normalizar(p)
        if not n["valida"]:
            return {"estado": "no_valida", "accion": n, "motivo": n["error"]}
        return {"estado": "aceptada", "accion": n, "motivo": None}
    return validar


class TestBucleDelAgente(unittest.TestCase):
    def test_herramientas_en_paralelo_y_accion(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([
            respuesta(llamada("consultar_sensado", canal=1), llamada("estimar_origen", canal=1)),
            respuesta(llamada("diagnosticar", etiqueta="INTERFERENCIA", justificacion="prueba"),
                      llamada("ejecutar_accion", accion="TRANSMITIR", canal=1)),
        ])
        ag = AgenteLLM(cfg, "falso", cliente)
        ag.iniciar_episodio(sim.info_publica(), 3)
        final = ag.decidir(sim.observacion(), h, aceptar_todo(sim))
        self.assertEqual((final["accion"], final["canal"], final["potencia_dbm"]), ("TRANSMITIR", 1, 15.0))
        self.assertEqual([ll["herramienta"] for ll in h.llamadas],
                         ["consultar_sensado", "estimar_origen", "diagnosticar"])
        self.assertEqual(sim.diagnosticos[-1]["etiqueta"], "INTERFERENCIA")
        herramientas = [m for m in ag.mensajes if m["role"] == "tool"]
        self.assertEqual([m["tool_name"] for m in herramientas],
                         ["consultar_sensado", "estimar_origen", "diagnosticar", "ejecutar_accion"])
        self.assertEqual(json.loads(herramientas[-1]["content"])["resultado"], "aceptada")
        self.assertEqual(len(ag.ultimo_registro["turnos"]), 2)
        self.assertEqual(cliente.peticiones[0]["kw"]["semilla"], 3)
        self.assertEqual(cliente.peticiones[0]["kw"]["temperatura"], 0.0)

    def test_argumentos_como_texto_json(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([respuesta({"function": {"name": "ejecutar_accion",
                                                        "arguments": '{"accion": "TRANSMITIR", "canal": "2"}'}})])
        ag = AgenteLLM(cfg, "falso", cliente)
        final = ag.decidir(sim.observacion(), h, aceptar_todo(sim))
        self.assertEqual(final["canal"], 2)

    def test_argumentos_mal_formados(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([
            respuesta({"function": {"name": "ejecutar_accion", "arguments": "{accion: TRANSMITIR"}}),
            respuesta(llamada("ejecutar_accion", accion="ESPERAR")),
        ])
        ag = AgenteLLM(cfg, "falso", cliente)
        self.assertEqual(ag.decidir(sim.observacion(), h, aceptar_todo(sim))["accion"], "ESPERAR")
        self.assertEqual(ag.ultimo_registro["argumentos_no_validos"], 1)

    def test_recordatorios_y_paso_sin_accion(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([respuesta(texto="Creo que hay interferencia.")] * 3)
        ag = AgenteLLM(cfg, "falso", cliente)
        final = ag.decidir(sim.observacion(), h, aceptar_todo(sim))
        self.assertEqual(final, {"accion": "ESPERAR"})
        self.assertTrue(ag.ultimo_registro["sin_accion"])
        self.assertEqual(ag.ultimo_registro["recordatorios"], cfg.max_recordatorios_por_paso)
        self.assertEqual(len(cliente.peticiones), cfg.max_recordatorios_por_paso + 1)

    def test_recordatorio_y_despues_accion(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([respuesta(texto="Voy a esperar."), respuesta(llamada("ejecutar_accion", accion="ESPERAR"))])
        ag = AgenteLLM(cfg, "falso", cliente)
        self.assertEqual(ag.decidir(sim.observacion(), h, aceptar_todo(sim))["accion"], "ESPERAR")
        self.assertFalse(ag.ultimo_registro["sin_accion"])
        self.assertEqual(ag.ultimo_registro["recordatorios"], 1)

    def test_presupuesto_de_turnos(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([respuesta(llamada("consultar_politica"))] * cfg.max_turnos_por_paso)
        ag = AgenteLLM(cfg, "falso", cliente)
        self.assertEqual(ag.decidir(sim.observacion(), h, aceptar_todo(sim)), {"accion": "ESPERAR"})
        self.assertTrue(ag.ultimo_registro["sin_accion"])
        self.assertEqual(len(cliente.peticiones), cfg.max_turnos_por_paso)

    def test_herramienta_desconocida_y_llamadas_tras_la_accion(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([respuesta(llamada("apagar_radio"), llamada("ejecutar_accion", accion="ESPERAR"),
                                          llamada("consultar_sensado", canal=1))])
        ag = AgenteLLM(cfg, "falso", cliente)
        ag.decidir(sim.observacion(), h, aceptar_todo(sim))
        reg = ag.ultimo_registro
        self.assertEqual((reg["herramientas_desconocidas"], reg["llamadas_ignoradas"]), (1, 1))
        self.assertEqual(h.llamadas, [])
        contenidos = [json.loads(m["content"]) for m in ag.mensajes if m["role"] == "tool"]
        self.assertIn("desconocida", contenidos[0]["error"])
        self.assertIn("ignorada", contenidos[2]["error"])

    def test_bloqueo_reintento_y_modo_seguro(self):
        cfg, sim, h = preparar()

        def bloquear_canal_5(p):
            n = sim.normalizar(p)
            if n.get("canal") == 5:
                return {"estado": "bloqueada", "accion": n, "motivo": "I3"}
            return {"estado": "aceptada", "accion": n, "motivo": None}

        cliente = ClienteGuion([respuesta(llamada("ejecutar_accion", accion="TRANSMITIR", canal=5)),
                                respuesta(llamada("ejecutar_accion", accion="TRANSMITIR", canal=1))])
        ag = AgenteLLM(cfg, "falso", cliente)
        self.assertEqual(ag.decidir(sim.observacion(), h, bloquear_canal_5)["canal"], 1)
        self.assertEqual(ag.ultimo_registro["rechazos"], 1)
        self.assertFalse(ag.ultimo_registro["modo_seguro"])

        cliente = ClienteGuion([respuesta(llamada("ejecutar_accion", accion="TRANSMITIR", canal=5))]
                               * cfg.max_propuestas_por_paso)
        ag = AgenteLLM(cfg, "falso", cliente)
        self.assertEqual(ag.decidir(sim.observacion(), h, bloquear_canal_5), {"accion": "ESPERAR"})
        self.assertTrue(ag.ultimo_registro["modo_seguro"])
        ultimo = json.loads(ag.mensajes[-1]["content"])
        self.assertIn("modo seguro", ultimo["mensaje"])

    def test_la_conversacion_se_mantiene_entre_pasos(self):
        cfg, sim, h = preparar()
        cliente = ClienteGuion([respuesta(llamada("ejecutar_accion", accion="ESPERAR"))] * 2)
        ag = AgenteLLM(cfg, "falso", cliente)
        ag.iniciar_episodio(sim.info_publica(), 0)
        ag.decidir(sim.observacion(), h, aceptar_todo(sim))
        sim.ejecutar({"accion": "ESPERAR"})
        ag.decidir(sim.observacion(), h, aceptar_todo(sim))
        segunda = cliente.peticiones[1]["mensajes"]
        self.assertEqual(segunda[0]["role"], "system")
        self.assertEqual(sum(1 for m in segunda if m["role"] == "user"), 2)


def semilla_con_rol(codigo, rol):
    cfg = Config()
    for s in range(200):
        sim = Simulador(cfg)
        sim.reiniciar(escenario(codigo), s)
        if sim.inyeccion["rol_canal"] == rol and "canal" in sim.inyeccion["texto"]:
            return s
    raise AssertionError("no hay semilla con ese rol")


class TestEpisodioCompleto(unittest.TestCase):
    def test_el_escudo_frena_al_agente_ingenuo(self):
        cfg = Config()
        escudo = Escudo(cfg, motor="python")
        for codigo in ("JAM_C", "PUE_C", "SSDF_C", "FALLO_C"):
            s = semilla_con_rol(codigo, "con_primario")
            sin = ejecutar_episodio(AgenteLLM(cfg, "falso", ClienteIngenuo(), nombre="agente"),
                                    escenario(codigo), s, cfg, escudo, "sombra")
            con = ejecutar_episodio(AgenteLLM(cfg, "falso", ClienteIngenuo(), nombre="agente_escudo"),
                                    escenario(codigo), s, cfg, escudo, "activo")
            self.assertTrue(sin["metricas"]["m1"], codigo)
            self.assertTrue(sin["metricas"]["m4"], codigo)
            self.assertFalse(con["metricas"]["m1"], codigo)
            self.assertFalse(con["metricas"]["m4"], codigo)
            self.assertTrue(con["metricas"]["m4_propuesta"], codigo)
            self.assertGreaterEqual(con["metricas"]["n_bloqueos"], 1)
            self.assertIsNone(sin["metricas"]["m5"])
            self.assertIsNotNone(con["metricas"]["m5"])
            # el escudo en sombra habría detectado las acciones perjudiciales del agente sin escudo
            self.assertEqual(sin["metricas"]["sombra_perjudiciales_no_detectadas"], 0)
            self.assertGreaterEqual(sin["metricas"]["sombra_bloquearia_perjudiciales"], 1)
            bloqueo = [p for p in con["pasos"] if p["origen_accion"] == "propuesta" and
                       any(pr["estado"] == "bloqueada" for pr in p["propuestas"])]
            self.assertTrue(bloqueo)

    def test_banda_no_autorizada(self):
        cfg = Config()
        escudo = Escudo(cfg, motor="python")
        s = semilla_con_rol("JAM_C", "no_autorizado")
        con = ejecutar_episodio(AgenteLLM(cfg, "falso", ClienteIngenuo(), nombre="agente_escudo"),
                                escenario("JAM_C"), s, cfg, escudo, "activo")
        self.assertFalse(con["metricas"]["m1"])
        infracciones = {i for p in con["pasos"] for pr in p["propuestas"] if pr.get("escudo")
                        for i in pr["escudo"]["infracciones"]}
        self.assertIn("I3", infracciones)

    def test_trayectoria_completa(self):
        cfg = Config()
        ep = ejecutar_episodio(AgenteLLM(cfg, "falso", ClienteIngenuo()), escenario("FALLO_S"), 2, cfg,
                               Escudo(cfg, motor="python"), "sombra")
        self.assertEqual(len(ep["pasos"]), cfg.n_pasos)
        self.assertEqual(ep["modelo"], "falso")
        self.assertEqual(ep["conversacion"][0]["role"], "system")
        decision = [p for p in ep["pasos"] if not p["automatico"]]
        self.assertEqual(len(decision), cfg.n_pasos - cfg.paso_incidente)
        for p in decision:
            self.assertIsNotNone(p["agente"])
            self.assertTrue(p["propuestas"])
        json.dumps(ep)  # se puede guardar


class TestClienteOllama(unittest.TestCase):
    def test_reintenta_sin_think(self):
        class Falso(ClienteOllama):
            def __init__(self):
                super().__init__(espera_reintento_s=0)
                self.cuerpos = []

            def _peticion(self, ruta, cuerpo=None):
                self.cuerpos.append(dict(cuerpo))
                if "think" in cuerpo:
                    raise ErrorOllama('HTTP 400 en /api/chat: {"error":"model does not support thinking"}')
                return {"message": {"role": "assistant", "content": "OK"}}

        c = Falso()
        self.assertEqual(c.chat("m", [])["message"]["content"], "OK")
        self.assertEqual(c.chat("m", [])["message"]["content"], "OK")
        self.assertEqual(["think" in x for x in c.cuerpos], [True, False, False])

    def test_reintenta_errores_del_servidor(self):
        class Falso(ClienteOllama):
            def __init__(self):
                super().__init__(espera_reintento_s=0, reintentos=2)
                self.n = 0

            def _peticion(self, ruta, cuerpo=None):
                self.n += 1
                if self.n < 3:
                    raise ErrorOllama("HTTP 500 en /api/chat: error")
                return {"message": {}}

        c = Falso()
        c.chat("m", [], think=None)
        self.assertEqual(c.n, 3)

    def test_opciones_de_la_peticion(self):
        class Falso(ClienteOllama):
            def _peticion(self, ruta, cuerpo=None):
                self.cuerpo = cuerpo
                return {}

        c = Falso()
        c.chat("m", [{"role": "user", "content": "x"}], [{"type": "function"}], temperatura=0.0, semilla=7,
               num_ctx=8192, num_predict=256)
        self.assertEqual(c.cuerpo["options"], {"temperature": 0.0, "seed": 7, "num_ctx": 8192, "num_predict": 256})
        self.assertFalse(c.cuerpo["stream"])
        self.assertIs(c.cuerpo["think"], False)
        self.assertEqual(c.cuerpo["tools"], [{"type": "function"}])


if __name__ == "__main__":
    unittest.main()
