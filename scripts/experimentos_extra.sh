#!/bin/bash
# Experimentos exploratorios sobre las mismas semillas de la campaña (fuera de los criterios fijados).
#
#   caffeinate -i bash scripts/experimentos_extra.sh
#
# Se puede lanzar mientras corre la campaña principal, en otra pestaña del Terminal:
#   - los pasos 1 a 3 solo usan granite, que es otro modelo y otro proceso de Ollama, así que no
#     alteran las respuestas de qwen; comparten la GPU, de modo que las dos cosas van algo más lentas;
#   - el paso 4 usa qwen como atacante y por eso espera a que termine la campaña principal: si se
#     enviaran peticiones a qwen mientras la campaña lo usa, Ollama podría agruparlas y cambiar
#     ligeramente sus respuestas;
#   - el paso 5 rehace los análisis con la campaña completa.
# Cada paso se puede interrumpir con Ctrl+C y, al relanzar el script, sigue donde se quedó.
set -u
cd "$(dirname "$0")/.."
MODELO="${MODELO:-granite4.1:8b}"
ATACANTE="${ATACANTE:-qwen3.8:27b}"
COMPUESTOS="JAM_C,PUE_C,SSDF_C,FALLO_C"
LOG="resultados/experimentos_extra.log"
mkdir -p resultados
echo "== experimentos extra, $(date) ==" | tee -a "$LOG"

echo "-- 1. ataque insistente ($MODELO)" | tee -a "$LOG"
python3 scripts/ataque_adaptativo.py --variante insistente --modelo "$MODELO" 2>&1 | tee -a "$LOG"
python3 scripts/ataque_adaptativo.py --variante insistente --modelo "$MODELO" --resumen 2>&1 | tee -a "$LOG"

echo "-- 2. agente endurecido ($MODELO)" | tee -a "$LOG"
python3 scripts/ejecutar.py --campana endurecido --sistemas agente_endurecido --modelos "$MODELO" \
    --escenarios "$COMPUESTOS" --semillas 20-59 2>&1 | tee -a "$LOG"
python3 scripts/comparar_campanas.py sistemas --modelo "$MODELO" --a endurecido:agente_endurecido \
    --b campana:agente --escenarios "$COMPUESTOS" 2>&1 | tee -a "$LOG"

echo "-- 3. repeticiones ($MODELO, semillas 20 a 29)" | tee -a "$LOG"
for r in 2 3; do
    python3 scripts/ejecutar.py --campana "repeticion_$r" --sistemas agente,agente_escudo --modelos "$MODELO" \
        --semillas 20-29 2>&1 | tee -a "$LOG"
done
for s in agente agente_escudo; do
    python3 scripts/comparar_campanas.py repeticiones --modelo "$MODELO" --sistema "$s" \
        --campanas campana,repeticion_2,repeticion_3 2>&1 | tee -a "$LOG"
done

# el atacante es qwen: se espera a que no quede ningún proceso de la campaña principal
while pgrep -f -- "--campana campana( |$)" > /dev/null; do
    echo "   esperando a que termine la campaña principal para usar $ATACANTE como atacante ($(date +%H:%M))" | tee -a "$LOG"
    sleep 600
done

echo "-- 4. ataque reescrito ($MODELO, atacante $ATACANTE)" | tee -a "$LOG"
python3 scripts/ataque_adaptativo.py --variante reescrita --modelo "$MODELO" --atacante "$ATACANTE" 2>&1 | tee -a "$LOG"
python3 scripts/ataque_adaptativo.py --variante reescrita --modelo "$MODELO" --resumen 2>&1 | tee -a "$LOG"

echo "-- 5. escudo sin agente y contrafactual con la campaña completa" | tee -a "$LOG"
python3 scripts/escudo_extra.py --campana campana 2>&1 | tee -a "$LOG"
echo "== fin, $(date) ==" | tee -a "$LOG"
