#!/bin/bash
# Experimento 6: ataques adaptativos a la evidencia del escudo y del monitor de vivacidad (exploratorio).
#
#   caffeinate -i bash scripts/experimentos_evidencia.sh
#
# Unas 3,5 horas con granite (586 episodios con modelo a unos 20 s, más 160 de reglas en segundos):
#   1. origen: el atacante falsea la estimación de origen del canal objetivo; agente_escudo y la variante sin
#      desbloqueo por origen (142 + 142 episodios con un canal autorizado con primario);
#   2. bizantinos: tres nodos mienten solo sobre el canal objetivo; agente_escudo (142);
#   3. enlace: «quedarse» con la tasa de entrega del canal activo falseada; vigilancia de la vivacidad (160) y
#      reglas (160).
# Cada paso se puede interrumpir con Ctrl+C y, al relanzar el script, sigue donde se quedó.
set -u
cd "$(dirname "$0")/.."
MODELO="${MODELO:-granite4.1:8b}"
LOG="resultados/experimentos_evidencia.log"
mkdir -p resultados
echo "== experimento 6, $(date) ==" | tee -a "$LOG"

for A in origen bizantinos enlace; do
    echo "-- ataque a la evidencia: $A ($MODELO)" | tee -a "$LOG"
    python3 scripts/ataque_evidencia.py --ataque "$A" --modelo "$MODELO" 2>&1 | tee -a "$LOG"
done

for A in origen bizantinos enlace; do
    python3 scripts/ataque_evidencia.py --ataque "$A" --modelo "$MODELO" --resumen 2>&1 | tee -a "$LOG"
done
echo "== fin, $(date) ==" | tee -a "$LOG"
