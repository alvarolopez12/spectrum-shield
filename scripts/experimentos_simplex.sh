#!/bin/bash
# Escudo con respaldo Simplex (exploratorio, fuera de los criterios fijados).
#
#   caffeinate -i bash scripts/experimentos_simplex.sh
#
# Se puede lanzar ya, en otra pestaña del Terminal: espera a que termine scripts/experimentos_extra.sh,
# porque los dos usan granite y, si corrieran a la vez, Ollama podría agrupar sus peticiones y cambiar
# ligeramente las respuestas (y las repeticiones de experimentos_extra.sh miden justo esa variabilidad).
# Primero corre con los textos del ataque insistente, que es el caso en que el escudo deja sin servicio,
# y después con los textos de la campaña en los ocho escenarios.
# Cada paso se puede interrumpir con Ctrl+C y, al relanzar el script, sigue donde se quedó.
set -u
cd "$(dirname "$0")/.."
MODELO="${MODELO:-granite4.1:8b}"
LOG="resultados/experimentos_simplex.log"
mkdir -p resultados
echo "== respaldo Simplex, $(date) ==" | tee -a "$LOG"

while pgrep -f "experimentos_extra\.sh" > /dev/null; do
    echo "   esperando a que termine experimentos_extra.sh ($(date +%H:%M))" | tee -a "$LOG"
    sleep 600
done

echo "-- 1. respaldo con el ataque insistente ($MODELO)" | tee -a "$LOG"
python3 scripts/respaldo_simplex.py --texto insistente --modelo "$MODELO" 2>&1 | tee -a "$LOG"
python3 scripts/respaldo_simplex.py --texto insistente --modelo "$MODELO" --resumen 2>&1 | tee -a "$LOG"

echo "-- 2. respaldo con los textos de la campaña, ocho escenarios ($MODELO)" | tee -a "$LOG"
python3 scripts/respaldo_simplex.py --texto original --modelo "$MODELO" --escenarios todos 2>&1 | tee -a "$LOG"
python3 scripts/respaldo_simplex.py --texto original --modelo "$MODELO" --escenarios todos --resumen 2>&1 | tee -a "$LOG"
echo "== fin, $(date) ==" | tee -a "$LOG"
