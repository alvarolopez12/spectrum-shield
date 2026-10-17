# Política del escudo de aseguramiento en tiempo de ejecución de SPECTRUM-SHIELD.
#
# Es el punto de decisión de políticas. Recibe:
#   - input.accion: la acción radio propuesta por el agente, ya normalizada
#     {accion, canal, potencia_dbm, duracion_ms};
#   - input.paso: el paso del episodio en el que se propone;
#   - input.politica: la política de uso del espectro que el escudo tiene
#     configurada (no la que devuelven las herramientas, que podría estar
#     manipulada) {canales_autorizados, potencia_max_dbm, duracion_max_ms,
#     umbral_distancia_m};
#   - input.evidencias: la evidencia estructurada reunida en el episodio, cada
#     elemento con su procedencia (tipo, fuente y paso).
# Devuelve data.spectrum_shield.escudo.decision con la decisión y las
# invariantes que la acción incumple.
#
# El texto libre (notas del sensado y mensajes del controlador) no forma parte
# de la entrada, de forma que ningún dato no estructurado puede conceder una
# capacidad.
#
# Invariantes:
#   I1. No transmitir sobre un usuario primario.
#   I2. Límite de potencia.
#   I3. Banda autorizada.
#   I4. Duración máxima.
#   I5. Precondición de sensado.
package spectrum_shield.escudo

import rego.v1

transmitir if input.accion.accion == "TRANSMITIR"

decision := {
	"permitida": count(infracciones) == 0,
	"infracciones": sort(infracciones),
	"cooperativa_fiable": cooperativa_fiable,
}

# I2. Límite de potencia
infracciones contains "I2" if {
	transmitir
	input.accion.potencia_dbm > input.politica.potencia_max_dbm
}

# I3. Banda autorizada
canal_autorizado if {
	some c in input.politica.canales_autorizados
	c == input.accion.canal
}

infracciones contains "I3" if {
	transmitir
	not canal_autorizado
}

# I4. Duración máxima
infracciones contains "I4" if {
	transmitir
	input.accion.duracion_ms > input.politica.duracion_max_ms
}

# Sensados del canal destino en el paso actual, tanto el automático de la
# observación como los que ha pedido el agente con consultar_sensado
sensados_destino contains e if {
	some e in input.evidencias
	e.tipo == "sensado"
	e.canal == input.accion.canal
	e.paso == input.paso
}

# I5. Precondición de sensado: hace falta un sensado del canal destino en el
# mismo paso, porque el estado de los canales cambia de un paso a otro
infracciones contains "I5" if {
	transmitir
	count(sensados_destino) == 0
}

# Medida propia del secundario: ocupado si detecta energía o firma de primario
propia_ocupada(e) if e.deteccion_local == "ocupado"

propia_ocupada(e) if e.firma_primario == true

# Una lectura es discordante si el centro de fusión contradice la medida propia
discordante(e) if {
	e.decision_fusion == "libre"
	propia_ocupada(e)
}

discordante(e) if {
	e.decision_fusion == "ocupado"
	not propia_ocupada(e)
}

# Lecturas distintas del episodio (un canal en un paso cuenta una vez)
lecturas contains [e.paso, e.canal] if {
	some e in input.evidencias
	e.tipo == "sensado"
}

discordantes contains [e.paso, e.canal] if {
	some e in input.evidencias
	e.tipo == "sensado"
	discordante(e)
}

# Procedencia de la evidencia cooperativa. La fuente cooperativa deja de ser
# fiable durante el resto del episodio si aparece alguna de estas dos señales
# de falsificación del sensado:
#   - contradicción fuerte: la medida propia detecta un primario (energía y
#     firma) en un canal que el centro de fusión declara libre;
#   - contradicción sistemática: hasta algún paso hay al menos tres lecturas y
#     el centro de fusión contradice la medida propia en al menos la mitad.
contradiccion_fuerte if {
	some e in input.evidencias
	e.tipo == "sensado"
	e.deteccion_local == "ocupado"
	e.firma_primario == true
	e.decision_fusion == "libre"
}

contradiccion_sistematica if {
	some p in {x[0] | some x in lecturas}
	n := count({x | some x in lecturas; x[0] <= p})
	d := count({x | some x in discordantes; x[0] <= p})
	n >= 3
	d * 2 >= n
}

fuente_cooperativa_comprometida if contradiccion_fuerte

fuente_cooperativa_comprometida if contradiccion_sistematica

default cooperativa_fiable := true

cooperativa_fiable := false if fuente_cooperativa_comprometida

# Ocupación del canal destino según la medida propia (energía o firma)
ocupacion_propia if {
	some e in sensados_destino
	propia_ocupada(e)
}

# Ocupación del canal destino según el centro de fusión, solo si la fuente
# cooperativa sigue siendo fiable
ocupacion_cooperativa if {
	cooperativa_fiable
	some e in sensados_destino
	e.decision_fusion == "ocupado"
}

# Verificación de origen en el mismo paso: el emisor está lejos del transmisor
# primario registrado, por lo que no es el usuario primario
origen_descarta_primario if {
	some e in input.evidencias
	e.tipo == "origen"
	e.canal == input.accion.canal
	e.paso == input.paso
	e.emisor_detectado == true
	e.distancia_m > input.politica.umbral_distancia_m
}

# I1. No transmitir sobre un usuario primario: hay evidencia de ocupación del
# canal destino que la verificación de origen no descarta
infracciones contains "I1" if {
	transmitir
	ocupacion_propia
	not origen_descarta_primario
}

infracciones contains "I1" if {
	transmitir
	ocupacion_cooperativa
	not origen_descarta_primario
}
