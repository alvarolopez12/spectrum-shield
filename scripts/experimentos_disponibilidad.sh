#!/bin/bash
# Ataque a la disponibilidad («seguro pero inútil») y Simplex con vigilancia de la vivacidad (exploratorio).
#
#   caffeinate -i bash scripts/experimentos_disponibilidad.sh
#
# Unas 7 horas con granite (1.280 episodios a unos 20 s):
#   1. ataque «esperar»: agente_escudo, agente_escudo_simplex y agente_escudo_vivo en los cuatro escenarios
#      compuestos, semillas 20 a 59 (480 episodios);
#   2. ataque «quedarse»: lo mismo (480 episodios);
#   3. agente_escudo_vivo con los textos de la campaña en los ocho escenarios (320 episodios), para compararlo
#      con campana y simplex_original.
# Cada paso se puede interrumpir con Ctrl+C y, al relanzar el script, sigue donde se quedó.
set -u
cd "$(dirname "$0")/.."
MODELO="${MODELO:-granite4.1:8b}"
LOG="resultados/experimentos_disponibilidad.log"
mkdir -p resultados
echo "== disponibilidad y vivacidad, $(date) ==" | tee -a "$LOG"

for T in esperar quedarse; do
    echo "-- ataque «seguro pero inútil»: $T ($MODELO)" | tee -a "$LOG"
    python3 scripts/disponibilidad.py --texto "$T" --modelo "$MODELO" 2>&1 | tee -a "$LOG"
done

echo "-- vigilancia de la vivacidad con los textos de la campaña, ocho escenarios ($MODELO)" | tee -a "$LOG"
python3 scripts/disponibilidad.py --texto original --modelo "$MODELO" 2>&1 | tee -a "$LOG"

for T in esperar quedarse original; do
    python3 scripts/disponibilidad.py --texto "$T" --modelo "$MODELO" --resumen 2>&1 | tee -a "$LOG"
done
echo "== fin, $(date) ==" | tee -a "$LOG"
