# AIllu · Let's work together

*Ayllu: la comunidad andina que trabaja junta.*

**Small AI for Development Hackathon · World Bank × Hack-Nation · Desafío de agricultura (Anexo B)**

Herramienta Small AI para cooperativas cafetaleras. El promotor de calidad fotografía en el acopio una muestra de 100 granos de café trillado. Un modelo pequeño, que funciona sin conexión, cuenta los defectos, estima la categoría SCA y el precio de referencia. Noor recibe el resultado por SMS y audio en quechua, en su teléfono básico. Además, Noor avisa por SMS cuánto café tendrá y cuándo. La cooperativa suma esos avisos, pronostica el volumen y, con la confirmación del gerente, publica un lote completo en la app del comprador, en el idioma del comprador.

**Problema (una frase, formato del reto).** Gracias a AIllu, Noor conocerá la categoría y el precio de referencia de su café, y la causa probable de la caída de su rendimiento (la broca), **antes** de que llegue el intermediario; sin ella vendería al precio que él le ponga y esperaría la visita del extensionista. Lo sabemos porque solo **3,8 % de los productores agrarios del Perú recibió asistencia técnica en 2022** (9,2 % en 2014; ENA 2022, INEI) y **223 mil familias** viven del café en el Perú (MIDAGRI, 2026).

**Idiomas.** La herramienta completa se puede usar en **español, quechua (runasimi) e inglés** (botón arriba a la derecha). El inglés es para que el jurado entienda cada función; los socios reciben SMS y audio en quechua o español, y cada mensaje tiene su traducción al inglés (botón EN).

---

## 1. Recorrido del usuario

| Paso | Quién | Qué pasa | Tecnología |
|---|---|---|---|
| 1 | Promotor (celular de la cooperativa) | Foto de 100 granos sobre una hoja blanca. Si hace falta, suma otra foto de la misma muestra | Segmentación clásica + CNN pequeña en el dispositivo |
| 2 | Sistema | Defectos → equivalencias SCA → distribución de probabilidad de la categoría → rango de precio | Reglas + simulación bayesiana |
| 3 | Promotor | Sigue 4 pasos guiados (socio → foto → revisar → enviar) y ve el SMS exacto antes de enviarlo. Si el sistema no está seguro, el mensaje deriva al catador | Humano en el circuito |
| 4 | Noor (teléfono básico) | SMS y audio en quechua o español: categoría, precio de referencia ("no es una oferta") y un consejo para la próxima cosecha | Audio ya grabado: MMS-TTS (quechua) y ElevenLabs (español) |
| 5 | Noor | Responde por SMS, por ejemplo "pusaq qq pergamino pa noviembre" | Lector de SMS (n-gramas + reglas para números en quechua) |
| 6 | Gerente | 4 pasos guiados: (1) atiende los SMS que el sistema no entendió, (2) ve el pronóstico del lote (P10–P90), (3) publica cuando hay 80 % de probabilidad de juntar 470 qq, (4) responde al comprador con plantillas | Simulación con el historial de cumplimiento de cada socio |
| 7 | Comprador | Ve el lote (volumen, perfil de calidad, origen) en inglés o español. Sus pedidos llegan a la cooperativa en español | Plantillas, sin traducción libre |

## 2. Dónde está la IA y por qué una herramienta más simple no basta

- **Visión por computadora.** Clasifica cada grano en 11 clases de defecto. Esto no se puede hacer con un SMS, una hoja de cálculo ni una búsqueda: hoy lo hace un catador, contando a mano.
- **Lenguaje.** El lector de SMS entiende mensajes libres en español, quechua o mezcla, con faltas, abreviaturas y números en quechua. Así nace un padrón de oferta sin formularios.
- **Voz.** La respuesta se escucha en quechua, una barrera de alfabetización que el SMS solo no resuelve.
- **Sin IA, a propósito:**
  - la tabla SCA;
  - el precio (Pink Sheet + diferenciales que fija la cooperativa);
  - los consejos, que son textos revisados, no generados en el momento;
  - el pronóstico del lote, que es estadística simple.

## 3. Restricciones del reto (sección 06)

| Regla | Cómo se cumple |
|---|---|
| Corre en un dispositivo que el usuario ya tiene | El celular de la cooperativa (navegador) y el teléfono básico de Noor (SMS y llamada) |
| Su función principal funciona sin conexión | Segmentación, CNN, grado, precio y audio corren en el navegador, sin servidor |
| Modelo pequeño | CNN de **123 mil parámetros, 0,49 MB** (float32); el motor de inferencia es JavaScript propio, sin librerías |
| Al menos una interacción en lengua local | **Quechua cusqueño (quz)**: SMS de entrada y salida, y audio |

## 4. Resultados

### Visión (clasificador de granos)

- Granos de prueba aislados (176): **98,9 %** sano vs. defecto; **98,3 %** de los defectos detectados; **83 %** de acierto en la clase exacta (11 clases).
- De punta a punta, en 60 fotos sintéticas de unos 85 granos:
  - segmentación: 99,7 % de los granos detectados;
  - error medio: **0,7 defectos por muestra** (sesgo −0,08);
  - categoría correcta: **90 %**;
  - en las 27 respuestas marcadas como "resultado confiable", **0 errores**;
  - 33 de 60 muestras se derivaron al catador o a una segunda foto, casi siempre porque 100 granos dan poca certeza estadística.
- Velocidad: unos 60 ms por grano en JavaScript, unos 6 s por muestra.

### Lenguaje (SMS)

Medido en un set nuevo de 305 SMS generados por otro modelo (Claude Sonnet) y no usado para ajustar reglas:

- intención: **90,5 %**;
- en avisos de cosecha: cantidad **95,3 %**, unidad **98,8 %**;
- se pide aclaración en 24 % de los casos;
- errores silenciosos (dato mal leído sin pedir aclaración): **3 de 172**.

## 5. Datos

| Dataset | Uso | Fuente y licencia | Tamaño |
|---|---|---|---|
| Green Coffee Beans Dataset | Entrenar y probar el clasificador | J4ckDev/GreenCoffeeBeansDataset (Colombia), **CC BY-NC-SA 4.0** | 582 granos, 11 clases |
| Muestras de 100 granos | Evaluación de punta a punta y demo | **Sintéticas**: montajes de granos de prueba sobre fondo de hoja con luz, sombra, desenfoque y JPEG variables | 60 + 2 fotos |
| SMS de oferta | Entrenar y probar el lector de SMS | **Sintéticos**, generados con Claude Haiku 4.5 (entrenamiento) y Claude Sonnet 4.6 (prueba) | 1 199 + 308 + 305 |
| Padrón y avisos de la cooperativa | Demo del acopio | **Sintéticos**, 50 socios | 50 + 34 |
| Consejos por defecto | Asesoría | Borradores de Claude a partir de pautas técnicas dadas por el equipo. **Pendiente: revisión de un agrónomo** | 10 defectos × 3 idiomas |
| Traducción al quechua | SMS y audio | Borradores de Claude. **Pendiente: revisión de un hablante** | — |
| Voz en quechua | Audio | facebook/mms-tts-quz (Meta MMS), **CC BY-NC 4.0** | 25 mensajes |
| Voz en español e inglés | Audio (el inglés es para el jurado) | ElevenLabs, eleven_v3 (voz natural) | 25 + 25 mensajes |
| Precio del arábica | Precio de referencia | World Bank Pink Sheet (agosto de 2026): arábica de julio 2026 = 7,91 US$/kg | 1 valor |

**Lo que estos datos no cubren**

- **No hay granos peruanos ni fotos reales de campo.** Los granos son de Colombia, de una sola sesión de fotos por clase. El modelo pudo aprender algo de esa luz, y las fotos de muestra son montajes. Antes de usarlo hace falta validar con fotos reales del acopio.
- **Hay clases con muy pocos ejemplos:** hongo (12), cereza seca (9) e inmaduro (16). Su detección es débil, sobre todo hongo e inmaduro.
- **El vinagre del dataset mezcla total y parcial.** Se usa la equivalencia SCA más conservadora (1:1).
- **La foto ve solo una cara del grano.** No mide humedad, olor ni taza, y el grado Especial requiere catación.
- **Los SMS sintéticos tienen un quechua probablemente imperfecto.** Generados por un LLM, no reemplazan mensajes reales.
- **Supuestos editables:**
  - 2 060 granos en 350 g;
  - rendimiento oro/pergamino de 0,80;
  - tipo de cambio de 3,45;
  - participación del productor de 85 %;
  - diferenciales por categoría;
  - corrección de la frecuencia de granos sanos (+1 logit), calibrada con las mismas muestras sintéticas.

## 6. Salvaguardas, privacidad y sesgo

- **No adivina.** Si la categoría más probable tiene menos de 70 %, o más de 8 % de los granos son dudosos, responde "no estamos seguros: lleve su café al catador" o pide otra foto.
- **Especial nunca se confirma por foto.**
- **Personas en el circuito:** el promotor revisa antes de enviar, el gerente confirma antes de publicar y Noor decide a quién vende. El precio siempre se dice como "referencia, no oferta".
- **Sin alucinaciones en tiempo real.** Los consejos y las plantillas son textos fijos y revisables. No hay un LLM respondiendo a los productores.
- **Privacidad.** El comprador no ve nombres ni teléfonos, solo códigos anónimos, y solo de socios que dieron su consentimiento. El padrón queda en la cooperativa.
- **Sesgo y lengua.** El TTS es quechua de Cusco. Para Puno o Ayacucho la calidad baja, y el plan es grabar las frases fijas con hablantes reales. Common Voice tiene quechua de Puno para evaluar el reconocimiento de voz.

## 7. Escalabilidad

- **Otras cooperativas y cultivos.** Cambiando la tabla de defectos y el dataset, sirve para otras cooperativas de café (Perú, Colombia, Etiopía, Côte d'Ivoire) y para cacao (prueba de corte).
- **Registro de productores.** Cada SMS de oferta alimenta un registro de productores, la precondición que el Anexo B identifica como cuello de botella.
- **Próximos pasos:**
  1. 300 fotos reales en una cooperativa de La Convención, etiquetadas por un catador.
  2. Revisión de consejos y quechua.
  3. Pasarela SMS real.
  4. Cuantización int8 (unos 0,13 MB).

## 8. Herramientas y costo de construcción

- **Claude** (este chat y API, unos US$ 2): código, datos sintéticos, borradores de consejos y traducción.
- **ElevenLabs** (unos 3 300 caracteres): voz en español.
- **Meta MMS:** voz en quechua.
- **PyTorch:** entrenamiento con CPU.
- **Prototipo:** HTML + JavaScript sin dependencias.

## 9. Fuentes de la evidencia del problema

- INEI–MIDAGRI, Encuesta Nacional Agropecuaria 2022: 3,8 % de productores recibió asistencia técnica y 6,6 % capacitación (citado en [Agraria.pe](https://agraria.pe/columna/asistencia-tecnica-y-capacitacion-en-la-agricultura-peruana-33242); publicación: [INEI, ENA 2022](https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1912/libro.pdf)).
- MIDAGRI (2026): exportaciones récord de café de US$ 1 796 millones que benefician a 223 mil familias ([gob.pe](https://www.gob.pe/institucion/midagri/noticias/1435801-peru-alcanza-record-de-us-1-796-millones-en-exportaciones-de-cafe-y-beneficia-a-223-mil-familias-productoras)).
- World Bank Pink Sheet, agosto 2026: arábica jul-2026 = 7,91 US$/kg.

## 10. Estructura del código

```
app/      prototipo (index.html, engine.js = motor offline, ui.json = textos en es/qu/en, pesos, audio2/ = voz es/qu/en, datos)
vision/   recorte de granos, generador de muestras, entrenamiento, evaluación
sms/      generador de SMS sintéticos (API de Claude), lector y clasificador de intención
logic/    tabla SCA, grado con incertidumbre, precio
content/  biblioteca de consejos (es/qu/en)
coop/     padrón y avisos sintéticos
audio/    generación de audio (ElevenLabs, MMS-TTS)
```
