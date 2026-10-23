#!/bin/bash
# Experimento 7: incidentes que las reglas no previeron (ATEN_S y FUSION_S), para H3.
#
#   caffeinate -i bash scripts/experimentos_incidentes.sh
#
# Unas 8 o 9 horas: las reglas y las reglas ampliadas en segundos (también en los ocho escenarios del catálogo),
# granite con el agente con escudo y con el sistema final (160 episodios, algo más de 1 hora) y qwen con los mismos
# (160 episodios, unas 7 horas). Cada paso se puede interrumpir con Ctrl+C y, al relanzar el script, sigue donde se
# quedó.
set -u
cd "$(dirname "$0")/.."
LOG="resultados/experimentos_incidentes.log"
mkdir -p resultados
echo "== experimento 7, $(date) ==" | tee -a "$LOG"
echo "-- reglas y reglas ampliadas" | tee -a "$LOG"
python3 scripts/incidentes_nuevos.py --parte reglas 2>&1 | tee -a "$LOG"
for M in granite4.1:8b qwen3.8:27b; do
    echo "-- agente con escudo y sistema final ($M)" | tee -a "$LOG"
    python3 scripts/incidentes_nuevos.py --parte agente --modelo "$M" 2>&1 | tee -a "$LOG"
done
python3 scripts/incidentes_nuevos.py --resumen 2>&1 | tee -a "$LOG"
echo "== fin, $(date) ==" | tee -a "$LOG"
