# Prueba de concepto con radio real

Esta carpeta lleva SPECTRUM-SHIELD del simulador a radio real en una banda de uso común sin licencia (434 MHz). Dos ADALM-Pluto hacen de entorno (usuarios primarios, emulación de primario e interferencia) y de usuario secundario, y un RTL-SDR mide el espectro. Cuando el sistema evaluado consulta el sensado, la energía, la detección local y la firma de primario salen de lo que mide el RTL-SDR, y cuando decide transmitir, el Pluto del secundario transmite de verdad en ese canal. El resto del experimento no cambia: mismos sistemas, mismo escudo, mismas semillas y mismas métricas. El código de las trayectorias (`src/spectrum_shield`) no se toca.

Es una demostración, no una evaluación nueva de las hipótesis: sirve para ver que la arquitectura funciona con medidas reales y para comparar, con las mismas semillas, lo que hacen los sistemas con sensado real y con sensado simulado.

## 1. Material y montaje

| Elemento | Uso | Notas |
|---|---|---|
| 2 × ADALM-Pluto | Pluto «E» (entorno) y Pluto «S» (secundario) | Etiquétalos con cinta. Cubren de 325 MHz a 3,8 GHz sin modificar nada. |
| 1 × RTL-SDR con TCXO | Receptor: sensado y cascada | Mejor con sintonizador R820T2 (RTL-SDR Blog V3, NooElec NESDR SMArt v5): funciona con el controlador estándar. El RTL-SDR Blog V4 también vale, pero hay que compilar su controlador (paso 2.3). |
| 3 antenas de 433 MHz (SMA) | Una por equipo | Las que trae el Pluto están pensadas para 900 y 1800 MHz; funcionan a esta distancia, pero las de 433 MHz quedan mejor y cuestan pocos euros. |
| 3 cables USB de datos | | El Pluto usa el conector USB del centro para datos. Si usas un concentrador, que tenga alimentación propia. |

Montaje: los tres equipos sobre la mesa, con las antenas verticales, los Pluto a entre 0,5 y 2 m del RTL-SDR y separados entre sí al menos 30 cm. Conéctalos unos 15 minutos antes de calibrar, para que los osciladores se estabilicen. Haz una foto del montaje para la memoria.

## 2. Instalación en macOS

Los pasos se hacen una vez. No se han podido probar en tu Mac desde aquí: si alguno falla, guarda el mensaje de error.

**2.1. Herramientas de compilación**

```
brew install cmake libusb libxml2 pkg-config git
```

**2.2. libiio 0.26 (para los Pluto)**

```
cd ~/Developer
git clone --branch v0.26 --depth 1 https://github.com/analogdevicesinc/libiio.git
cd libiio && mkdir build && cd build
cmake .. -DPYTHON_BINDINGS=OFF -DWITH_DOC=OFF -DWITH_SERIAL_BACKEND=OFF -DOSX_PACKAGE=OFF -DOSX_FRAMEWORK=OFF
make -j8
sudo make install
iio_info -s
```

Con los dos Pluto conectados, `iio_info -s` debe listar dos contextos `usb:x.y.z`, cada uno con su `serial=...`.

**2.3. Controlador del RTL-SDR**

Con sintonizador R820T2 (V3, NooElec):

```
brew install librtlsdr
rtl_test -t
```

Con el RTL-SDR Blog V4 (necesita el controlador de RTL-SDR Blog):

```
brew uninstall librtlsdr 2>/dev/null
cd ~/Developer
git clone --depth 1 https://github.com/rtlsdrblog/rtl-sdr-blog.git
cd rtl-sdr-blog && mkdir build && cd build
cmake .. -DINSTALL_UDEV_RULES=OFF
make -j8
sudo make install
rtl_test -t
```

`rtl_test -t` debe encontrar el dispositivo (con el V4, «RTL-SDR Blog V4 Detected»). Ciérralo con Ctrl+C.

**2.4. Paquetes de Python, en un entorno virtual**

Desde la carpeta del repositorio:

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r poc_radio/requirements.txt
```

A partir de aquí, todos los comandos de esta carpeta se lanzan con el entorno activado (`source .venv/bin/activate`). El experimento principal también funciona dentro del entorno, porque solo usa la biblioteca estándar.

**2.5. Identificar los Pluto**

```
iio_info -s
```

Conecta primero solo el Pluto E y anota su número de serie; después el S. Copia los dos en `poc_radio/radio.json` (`serie_pluto_entorno` y `serie_pluto_secundario`). Si tienes varios RTL-SDR, pon también `serie_rtl`.

Alternativa si la compilación de los pasos 2.2 y 2.3 da problemas: radioconda (https://github.com/ryanvolz/radioconda), que trae libiio, pyadi-iio y el controlador del RTL-SDR ya compilados para Apple Silicon. Se instala con su paquete `.pkg` y se usa su `python3` en lugar del entorno virtual.

## 3. Normativa

| Requisito (Decisión 2006/771/CE, versión de la Decisión de Ejecución (UE) 2022/180; CNAF, nota UN-158) | Prueba de concepto |
|---|---|
| Banda 434,040 a 434,790 MHz, dispositivos de corto alcance no específicos | Seis canales de 434,150 a 434,650 MHz, cada uno con sus 25 kHz dentro de la banda |
| 10 mW p.r.a. como máximo | Tope de ganancia de -10 dB en el programa: unos 0,6 mW con una antena de 3 dBi. En funcionamiento normal, del orden de microvatios (la calibración ajusta la ganancia a lo necesario) |
| Ciclo de trabajo de hasta el 100 % con canalización de 25 kHz o menos | Todas las emisiones ocupan menos de 12,5 kHz |
| Inhibidores prohibidos (Ley 11/2022, art. 106.9) | La «interferencia» es una emisión de datos de banda estrecha en nuestro propio canal y con la misma potencia que las demás; no busca bloquear comunicaciones de nadie |

El ADALM-Pluto es un módulo de evaluación para desarrollo, pruebas y demostración: no está homologado como equipo de radio terminado y quien lo usa responde de cumplir la normativa. Por eso el programa fija el plan de frecuencias, comprueba la banda y limita la ganancia, y la calibración usa la potencia mínima que da una medida clara. El canal 5 es la «banda protegida» del experimento, pero solo como papel: está en la misma banda de uso común que los demás.

## 4. Comprobación y calibración

```
python3 poc_radio/comprobar_radio.py --listar     # bibliotecas y dispositivos
python3 poc_radio/comprobar_radio.py              # comprobación completa y calibración (menos de 1 min)
python3 poc_radio/comprobar_radio.py --vigilar 120   # opcional: emisiones ajenas en nuestros canales
```

La comprobación completa: (1) revisa el plan de frecuencias frente a la normativa; (2) mide el ruido de fondo de cada canal con los Pluto en silencio; (3) mide el error de frecuencia de cada Pluto con un tono y lo corrige (los Pluto pueden desviarse varios kHz a 434 MHz; el RTL-SDR hace de referencia); (4) ajusta las ganancias para que las emisiones lleguen con unos 30 dB (entorno) y 36 dB (secundario) sobre el ruido; (5) comprueba que el detector ve la firma del primario en los seis canales y no la ve en la interferencia; (6) mide la imagen del Pluto, lo que queda con los Pluto en silencio y si el receptor satura en el peor caso. Termina con «Todo correcto» o con la lista de cosas que revisar. La calibración se guarda en `resultados/poc_radio/calibracion.json`; la demostración y la minicampaña repiten al empezar la parte rápida (ruido y frecuencia).

## 5. Demostración (los cuatro casos)

Primero, elegir el episodio. El script busca en las campañas simuladas un episodio compuesto en el que se vean los cuatro comportamientos (la inyección consigue la acción perjudicial sin escudo, el escudo la bloquea, el ataque insistente deja sin servicio y el respaldo Simplex recupera) y en el que el sensado simulado no tenga errores de detección:

```
python3 poc_radio/demo_radio.py --buscar
python3 poc_radio/demo_radio.py --escenario JAM_C --semilla 23
```

Se abre una ventana con la cascada de los seis canales en directo y, a la derecha, lo que va pasando (sensados, bloqueos del escudo, transmisiones, diagnósticos). Los cuatro casos se ejecutan seguidos sobre el mismo episodio:

| Caso | Qué se ve en la cascada |
|---|---|
| Agente sin escudo | El secundario transmite donde pide la inyección: encima del primario o en el canal 5 |
| Agente con escudo | Aparecen los bloqueos (aspas naranjas) y el secundario no transmite donde pide la inyección |
| Ataque insistente | El escudo sigue bloqueando, pero el secundario deja de transmitir: el canal se queda en silencio |
| Respaldo Simplex | Tras el bloqueo, decide el árbol de reglas y el secundario transmite en un canal libre |

Cada demostración se guarda en `resultados/poc_radio/demo_<fecha>_<hora>/` con los episodios, las cascadas grabadas, un resumen y la figura de los cuatro casos (`cascada_casos.png`). Para ensayar la charla basta con repetir el comando. Cerrar la ventana detiene la demostración al acabar el episodio en curso; Ctrl+C en el Terminal la para en el acto. En los dos casos los Pluto quedan en silencio.

## 6. Minicampaña

```
caffeinate -i python3 poc_radio/minicampana_radio.py
python3 poc_radio/minicampana_radio.py --resumen
```

Cinco configuraciones (reglas, agente sin escudo, agente con escudo, ataque insistente y respaldo Simplex) sobre los cuatro escenarios compuestos y las semillas 20 a 24: 100 episodios, unas 2 a 3 horas con granite4.1:8b. Se puede interrumpir y relanzar. El resumen da, por configuración, las métricas con radio real junto a las de la campaña simulada con las mismas semillas y cuántos episodios coinciden, y, sobre todas las consultas de sensado, la probabilidad de detección y de falsa alarma del sensado real frente a las que supone el simulador. Los datos quedan en `resultados/poc_radio/datos_radio.json`.

## 7. Figuras para la memoria

```
python3 poc_radio/figuras_radio.py
```

Genera en `resultados/poc_radio/figuras/` el plan de frecuencias, las formas de onda, la cascada de los cuatro casos (de la última demostración) y la energía medida frente al modelo del simulador (de la minicampaña).

## 8. Qué es real y qué sigue emulado

| Elemento | En la prueba con radio |
|---|---|
| Ocupación de cada canal (primarios, emulador del PUE, interferencia) | Emisiones reales del Pluto E, según el guion del simulador para ese paso |
| Energía, detección local y firma de primario | Medidas reales del RTL-SDR |
| Transmisión del secundario | Ráfaga real del Pluto S, con la duración pedida y una potencia relativa a la pedida; se comprueba en el aire |
| Fusión cooperativa (cinco nodos; tres bizantinos en SSDF) | Emulada, con la misma semilla que en la campaña |
| Notas del sensado y mensajes del controlador (donde llega la inyección) | Emulados: es texto |
| Distancia del emisor (verificación de origen) | Emulada; solo se localiza un emisor si la medida real supera el umbral |
| Tasa de entrega y RSSI del enlace | Emuladas |
| Agente, reglas, escudo y respaldo Simplex | Los mismos, sin cambios |

Limitaciones: un solo receptor (el sensado cooperativo sigue emulado), sin localización real, en laboratorio y a corta distancia (la relación señal a ruido es mucho más alta que la que supone el simulador, que es más pesimista), una sola banda y 20 episodios por configuración.

## 9. Prueba sin hardware

Todo se puede probar con un banco simulado (dos Pluto y un RTL-SDR simulados, con error de frecuencia, imagen, ruido y cuantificación a 8 bits) y, si hace falta, con un agente simulado sin Ollama. Lo que se genera así va a `resultados/poc_radio_simulado/` y las figuras llevan la marca «SEÑALES SINTÉTICAS»: no sirve como resultado.

```
python3 poc_radio/comprobar_radio.py --simulado
python3 poc_radio/demo_radio.py --simulado --llm-simulado --escenario JAM_C --semilla 20 --sin-cascada
python3 poc_radio/minicampana_radio.py --simulado --llm-simulado --semillas 20
```

## 10. Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `iio_info -s` no ve los Pluto | Cable de datos en el puerto del centro; desconectar y conectar; probar sin concentrador |
| Los dos Pluto aparecen pero uno no abre | Revisa los números de serie en `radio.json`; se abren por USB, no por IP (los dos tienen 192.168.2.1 por defecto) |
| `No se puede cargar pyrtlsdr o librtlsdr` | Paso 2.3; con el V4, que no esté instalado `pyrtlsdrlib` (`pip uninstall pyrtlsdrlib`) |
| `usb_claim_interface error -3` o el RTL-SDR está ocupado | Otro programa lo tiene abierto (SDR++, GQRX): ciérralo |
| «El RTL-SDR no ve el tono» al calibrar | Antenas puestas y a menos de 2 m; mira el LED del Pluto |
| Falta firma en algún canal o la SNR queda baja | Acerca los equipos; el programa no sube la ganancia por encima del tope legal |
| Satura el receptor | Baja `ganancia_rtl_db` en `radio.json` (por ejemplo, a 20) y repite la comprobación |
