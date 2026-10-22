#!/bin/bash
# Experimento 6, complemento: el ataque al origen contra el sistema final y el coste de quitar el desbloqueo.
#
#   caffeinate -i bash scripts/experimentos_evidencia_vivo.sh
#
# Se puede lanzar mientras termina experimentos_evidencia.sh: espera a que su registro acabe en «== fin» y entonces
# arranca. Unas 3 horas con granite (364 episodios):
#   1. origen: agente_escudo_vivo y agente_escudo_vivo_sin_origen con el ataque al origen (142 + 142);
#   2. original: agente_escudo_vivo_sin_origen sin ataque en PUE_S y PUE_C (80).
# Cada paso se puede interrumpir con Ctrl+C y, al relanzar el script, sigue donde se quedó.
set -u
cd "$(dirname "$0")/.."
MODELO="${MODELO:-granite4.1:8b}"
PREVIO="resultados/experimentos_evidencia.log"
LOG="resultados/experimentos_evidencia_vivo.log"
mkdir -p resultados

ultima_marca() { grep '^== ' "$PREVIO" 2>/dev/null | tail -1 | cut -c1-6; }
if [ -f "$PREVIO" ] && [ "$(ultima_marca)" != "== fin" ]; then
    echo "Esperando a que termine experimentos_evidencia.sh (se mira su registro cada minuto)..."
    while [ "$(ultima_marca)" != "== fin" ]; do sleep 60; done
fi

echo "== experimento 6 con vivacidad, $(date) ==" | tee -a "$LOG"
for P in origen original; do
    echo "-- $P ($MODELO)" | tee -a "$LOG"
    python3 scripts/ataque_evidencia_vivo.py --parte "$P" --modelo "$MODELO" 2>&1 | tee -a "$LOG"
done

for P in origen original; do
    python3 scripts/ataque_evidencia_vivo.py --parte "$P" --modelo "$MODELO" --resumen 2>&1 | tee -a "$LOG"
done
echo "== fin, $(date) ==" | tee -a "$LOG"
