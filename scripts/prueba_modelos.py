#!/usr/bin/env python3
"""Prueba rápida de los modelos antes del piloto.

Para cada modelo comprueba que está descargado, lo carga, y ejecuta tres
episodios cortos (JAM_S y JAM_C sin escudo, JAM_C con escudo) con la semilla 0.
Mide si el modelo termina cada paso con una llamada válida a ejecutar_accion,
la latencia por turno, la velocidad de generación y el tamaño del contexto, y
estima cuánto durarían el piloto y la campaña.

  python3 scripts/prueba_modelos.py
  python3 scripts/prueba_modelos.py --modelos granite4.1:8b
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from spectrum_shield.entorno import Config  # noqa: E402
from spectrum_shield.evaluacion import Campana, ejecutar_episodio  # noqa: E402
from spectrum_shield.incidentes import escenario  # noqa: E402
from spectrum_shield.sistemas import (AgenteLLM, ClienteOllama, ErrorConexionOllama,  # noqa: E402
                                      ErrorOllama, Escudo)

PRUEBAS = [("agente", "JAM_S"), ("agente", "JAM_C"), ("agente_escudo", "JAM_C")]
EPISODIOS_PILOTO_POR_MODELO = 2 * 8 * 5      # agente y agente_escudo, 8 escenarios, 5 semillas
EPISODIOS_CAMPANA_POR_MODELO = 2 * 8 * 20    # ídem con 20 semillas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelos", default="qwen3.8:27b,granite4.1:8b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--config", default=str(RAIZ / "config" / "experimento.json"))
    p.add_argument("--motor-escudo", default="auto", choices=("auto", "opa", "python"))
    args = p.parse_args()

    ruta_cfg = Path(args.config)
    cfg = Config.desde_json(ruta_cfg) if ruta_cfg.exists() else Config()
    cliente = ClienteOllama(args.url)
    try:
        print(f"Ollama {cliente.version()}")
        locales = cliente.modelos_locales()
    except ErrorConexionOllama as e:
        print(e)
        return 2
    escudo = Escudo(cfg, motor=args.motor_escudo)
    print(f"Escudo: {escudo.version_motor()}")
    campana = Campana(RAIZ / "resultados", "prueba_modelos_" + datetime.now().strftime("%Y%m%d_%H%M"))

    informe = []
    for modelo in [m.strip() for m in args.modelos.split(",") if m.strip()]:
        print(f"\n=== {modelo}")
        if modelo not in locales:
            print(f"No está descargado. Ejecuta: ollama pull {modelo}")
            informe.append((modelo, "no descargado", None))
            continue
        inicio = time.time()
        try:
            r = cliente.chat(modelo, [{"role": "user", "content": "Responde solo OK"}], semilla=0,
                             num_ctx=cfg.num_ctx, num_predict=16)
        except ErrorOllama as e:
            print(f"Error al cargar el modelo: {e}")
            informe.append((modelo, "error de carga", None))
            continue
        print(f"Carga y primera respuesta: {time.time() - inicio:.1f} s "
              f"(respuesta: {((r.get('message') or {}).get('content') or '').strip()[:40]!r})")

        datos = {"pasos": 0, "con_accion": 0, "sin_accion": 0, "modo_seguro": 0, "turnos": 0,
                 "recordatorios": 0, "no_validas": 0, "desconocidas": 0, "args_mal": 0, "truncadas": 0,
                 "tokens_salida": 0, "salida_s": 0.0, "tokens_entrada": 0, "entrada_s": 0.0,
                 "duracion_turnos": 0.0, "max_contexto": 0, "episodios_s": []}
        for sis, cod in PRUEBAS:
            agente = AgenteLLM(cfg, modelo, cliente, nombre=sis)
            modo = "activo" if sis == "agente_escudo" else "sombra"
            try:
                ep = ejecutar_episodio(agente, escenario(cod), 0, cfg, escudo, modo, campana=campana.nombre)
            except ErrorOllama as e:
                print(f"{sis} {cod}: ERROR {e}")
                continue
            campana.guardar(ep)
            m = ep["metricas"]
            datos["episodios_s"].append(ep["duracion_s"])
            for paso in ep["pasos"]:
                if paso["automatico"]:
                    continue
                a = paso["agente"]
                datos["pasos"] += 1
                datos["sin_accion"] += int(a["sin_accion"])
                datos["modo_seguro"] += int(a["modo_seguro"])
                datos["con_accion"] += int(not a["sin_accion"])
                datos["recordatorios"] += a["recordatorios"]
                datos["desconocidas"] += a["herramientas_desconocidas"]
                datos["args_mal"] += a["argumentos_no_validos"]
                datos["no_validas"] += sum(1 for pr in paso["propuestas"] if pr["estado"] == "no_valida")
                for t in a["turnos"]:
                    datos["turnos"] += 1
                    datos["duracion_turnos"] += t["duracion_s"]
                    datos["tokens_salida"] += t["tokens_salida"] or 0
                    datos["salida_s"] += t["salida_s"]
                    datos["tokens_entrada"] += t["tokens_entrada"] or 0
                    datos["entrada_s"] += t["entrada_s"]
                    datos["truncadas"] += int(t["fin"] == "length")
            contexto = sum(len(str(msg.get("content", ""))) + len(str(msg.get("tool_calls", "")))
                           for msg in ep["conversacion"])
            datos["max_contexto"] = max(datos["max_contexto"], contexto)
            print(f"{sis:14s} {cod}: {ep['duracion_s']:6.1f} s | M1={m['m1']:d} M2={m['m2']:d} "
                  f"M3={m['m3']:d} M4={'-' if m['m4'] is None else int(m['m4'])} "
                  f"M5={'-' if m['m5'] is None else int(m['m5'])} | turnos {m['turnos_llm']} | "
                  f"diagnóstico {m['diagnostico_final']}")

        if not datos["pasos"]:
            informe.append((modelo, "sin datos", None))
            continue
        vel_salida = datos["tokens_salida"] / datos["salida_s"] if datos["salida_s"] else 0.0
        vel_entrada = datos["tokens_entrada"] / datos["entrada_s"] if datos["entrada_s"] else 0.0
        media_ep = sum(datos["episodios_s"]) / len(datos["episodios_s"])
        apto = (datos["sin_accion"] == 0 and datos["desconocidas"] == 0 and datos["args_mal"] == 0
                and datos["truncadas"] == 0)
        print(f"Pasos de decisión: {datos['pasos']}, terminados con ejecutar_accion: {datos['con_accion']}, "
              f"sin acción: {datos['sin_accion']}, en modo seguro: {datos['modo_seguro']}")
        print(f"Recordatorios: {datos['recordatorios']}, acciones no válidas: {datos['no_validas']}, "
              f"herramientas desconocidas: {datos['desconocidas']}, argumentos mal formados: {datos['args_mal']}, "
              f"respuestas truncadas: {datos['truncadas']}")
        print(f"Latencia media por turno: {datos['duracion_turnos'] / max(1, datos['turnos']):.1f} s | "
              f"generación {vel_salida:.1f} tokens/s | lectura del contexto {vel_entrada:.0f} tokens/s | "
              f"contexto máximo ~{datos['max_contexto'] // 3} tokens (de {cfg.num_ctx})")
        print(f"Episodio medio: {media_ep:.0f} s -> piloto (5 semillas) ~"
              f"{media_ep * EPISODIOS_PILOTO_POR_MODELO / 3600:.1f} h, campaña (20 semillas) ~"
              f"{media_ep * EPISODIOS_CAMPANA_POR_MODELO / 3600:.1f} h con este modelo")
        informe.append((modelo, "apto" if apto else "revisar", media_ep))

    print("\n=== Resumen")
    for modelo, estado, media in informe:
        print(f"{modelo:20s} {estado:14s}" + (f" episodio medio {media:.0f} s" if media else ""))
    print(f"Trayectorias guardadas en {campana.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
