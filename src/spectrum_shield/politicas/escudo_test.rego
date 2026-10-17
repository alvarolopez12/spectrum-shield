# Pruebas de la política del escudo: opa test src/spectrum_shield/politicas -v
package spectrum_shield.escudo_test

import rego.v1

import data.spectrum_shield.escudo

politica := {
	"canales_autorizados": [0, 1, 2, 3, 4],
	"potencia_max_dbm": 20.0,
	"duracion_max_ms": 1000,
	"umbral_distancia_m": 100.0,
}

libre(canal, paso) := {
	"tipo": "sensado", "fuente": "consultar_sensado", "paso": paso, "canal": canal,
	"deteccion_local": "libre", "energia_dbm": -95.0, "firma_primario": false,
	"decision_fusion": "libre", "votos_ocupado": 0,
}

con_primario(canal, paso) := {
	"tipo": "sensado", "fuente": "consultar_sensado", "paso": paso, "canal": canal,
	"deteccion_local": "ocupado", "energia_dbm": -60.0, "firma_primario": true,
	"decision_fusion": "ocupado", "votos_ocupado": 5,
}

transmitir(canal, potencia, duracion) := {"accion": "TRANSMITIR", "canal": canal, "potencia_dbm": potencia, "duracion_ms": duracion}

entrada(accion, evidencias) := {"accion": accion, "paso": 3, "politica": politica, "evidencias": evidencias}

test_transmision_correcta_permitida if {
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [libre(2, 3)])
	d.permitida
	d.infracciones == []
}

test_esperar_siempre_permitida if {
	d := escudo.decision with input as entrada({"accion": "ESPERAR"}, [])
	d.permitida
}

test_i2_potencia if {
	d := escudo.decision with input as entrada(transmitir(2, 25.0, 500), [libre(2, 3)])
	d.infracciones == ["I2"]
}

test_i3_banda if {
	d := escudo.decision with input as entrada(transmitir(5, 15.0, 500), [libre(5, 3)])
	d.infracciones == ["I3"]
}

test_i4_duracion if {
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 3000), [libre(2, 3)])
	d.infracciones == ["I4"]
}

test_i5_sin_sensado if {
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [libre(1, 3)])
	d.infracciones == ["I5"]
}

test_i5_sensado_de_otro_paso if {
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [libre(2, 2)])
	d.infracciones == ["I5"]
}

test_i1_primario if {
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [con_primario(2, 3)])
	d.infracciones == ["I1"]
}

test_i1_origen_lejano_descarta_primario if {
	origen := {
		"tipo": "origen", "fuente": "estimar_origen", "paso": 3, "canal": 2,
		"emisor_detectado": true, "distancia_m": 450.0,
	}
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [con_primario(2, 3), origen])
	d.permitida
}

test_i1_origen_cercano_no_descarta if {
	origen := {
		"tipo": "origen", "fuente": "estimar_origen", "paso": 3, "canal": 2,
		"emisor_detectado": true, "distancia_m": 25.0,
	}
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [con_primario(2, 3), origen])
	d.infracciones == ["I1"]
}

test_i1_fusion_ocupada_bloquea_si_es_fiable if {
	e := object.union(libre(2, 3), {"decision_fusion": "ocupado", "votos_ocupado": 4})
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), [e])
	d.infracciones == ["I1"]
	d.cooperativa_fiable
}

# primario que el centro de fusión da por libre (contradicción fuerte: energía y firma)
invertido_libre(canal, paso) := object.union(con_primario(canal, paso), {"decision_fusion": "libre", "votos_ocupado": 2})

# energía sin firma que el centro de fusión da por libre (contradicción débil)
debil_libre(canal, paso) := object.union(libre(canal, paso), {"deteccion_local": "ocupado", "energia_dbm": -70.0})

# canal libre que el centro de fusión da por ocupado (contradicción débil)
invertido_ocupado(canal, paso) := object.union(libre(canal, paso), {"decision_fusion": "ocupado", "votos_ocupado": 3})

test_contradiccion_fuerte_basta if {
	evidencias := [invertido_libre(0, 2), invertido_ocupado(2, 3)]
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), evidencias)
	d.permitida
	not d.cooperativa_fiable
}

test_fusion_contradicha_en_la_mitad_de_las_lecturas if {
	# en el paso 2 el centro de fusión contradice la medida propia en dos de tres canales
	evidencias := [debil_libre(0, 2), invertido_ocupado(1, 2), libre(3, 2), invertido_ocupado(2, 3)]
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), evidencias)
	d.permitida
	not d.cooperativa_fiable
}

test_discrepancias_aisladas_no_bastan if {
	evidencias := [debil_libre(0, 2), libre(1, 2), libre(3, 2), libre(4, 2), invertido_ocupado(2, 3)]
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), evidencias)
	d.infracciones == ["I1"]
	d.cooperativa_fiable
}

test_hacen_falta_tres_lecturas if {
	evidencias := [debil_libre(0, 2), invertido_ocupado(2, 3)]
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), evidencias)
	d.infracciones == ["I1"]
	d.cooperativa_fiable
}

test_la_medida_propia_manda_aunque_no_se_confie_en_la_fusion if {
	evidencias := [invertido_libre(0, 2), invertido_libre(2, 3)]
	d := escudo.decision with input as entrada(transmitir(2, 15.0, 500), evidencias)
	d.infracciones == ["I1"]
	not d.cooperativa_fiable
}

test_una_lectura_repetida_cuenta_una_vez if {
	evidencias := [debil_libre(0, 2), debil_libre(0, 2), debil_libre(0, 2), libre(1, 2), libre(3, 2)]
	d := escudo.decision with input as entrada(transmitir(1, 15.0, 500), evidencias)
	d.cooperativa_fiable
}

test_varias_infracciones_ordenadas if {
	d := escudo.decision with input as entrada(transmitir(5, 30.0, 3000), [])
	d.infracciones == ["I2", "I3", "I4", "I5"]
}
