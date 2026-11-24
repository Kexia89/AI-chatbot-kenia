# Model Card

## Descripción

Clasificador multiclase de intenciones en español integrado con una capa de
estado conversacional. El artefacto utiliza 33 etiquetas estadísticas y deriva
afirmación y negación mediante reglas dependientes del historial.

## Arquitectura

- TF-IDF de palabras con n-gramas `(1, 2)`.
- TF-IDF de caracteres con n-gramas `(3, 5)`.
- Regresión logística multiclase con regularización L2.
- Umbral de rechazo calibrado en validación.
- Reglas deterministas para respuestas cortas, repetición y acciones no conectadas.

## Evaluación

El modelo 0.2 obtuvo 81.50% de accuracy y 80.75% de F1 macro en una partición
interna de 200 ejemplos. Con rechazo selectivo alcanzó 91.97% de exactitud sobre
68.50% de cobertura. Estos resultados no sustituyen una evaluación externa.

## Uso previsto

- Prototipos conversacionales ligeros en español.
- Clasificación de intención con baja latencia.
- Experimentación con rechazo por incertidumbre.

## Limitaciones

- La prueba proviene del mismo corpus base que el entrenamiento.
- Las clases `limitaciones`, `aclaracion` y estados emocionales necesitan más datos.
- No ejecuta acciones externas ni sustituye atención médica o servicios de emergencia.
- El archivo Joblib debe cargarse únicamente desde una fuente confiable.

## Datos

El repositorio no distribuye el corpus de entrenamiento. El artefacto incluye
solo el vocabulario aprendido y los recursos mínimos requeridos durante inferencia.
