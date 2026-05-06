# Estructura Lógica del Backend - Proyecto Anemia (NFNet)

Este documento detalla el flujo de procesamiento de imágenes y cálculo de métricas de explicabilidad implementadas en el sistema.

## 1. Etapas del Procesamiento de Imágenes (Individual)

### Paso A: Recepción y Filtrado
- **Función**: `evaluar_imagen_anemia` (views.py)
- **Lógica Detallada**:
  - **Recepción**: La imagen ingresa en formato RAW (base64 o archivo) desde el front-end o lote.
  - **Conversión**: Se convierte a espacio de color RGB y escala de grises para el análisis de calidad.
  - **Filtros de Calidad (Descarte de imágenes no viables)**:
    1. **Brillo (Brightness)**: Se calcula la media de los píxeles. Si la imagen es excesivamente oscura (subexpuesta) o demasiado brillante (sobreexpuesta) fuera de los umbrales configurados (`MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`), se rechaza.
    2. **Contraste (RMS Contrast)**: Se mide la desviación estándar de la intensidad. Se descartan imágenes "planas" sin detalles (`MIN_CONTRAST`).
    3. **Nitidez (Laplacian Variance)**: Se aplica un filtro Laplaciano para medir bordes. Si la varianza es muy baja, la imagen está borrosa y se rechaza (`MIN_BLUR`).
- **Objetivo**: Asegurar que solo imágenes nítidas y bien iluminadas pasen a la segmentación, ahorrando procesamiento inútil.

### Paso B: Segmentación de Conjuntiva
- **Función Principal**: `extraer_conjuntiva`
- **Flujo Detallado de Detección y Extracción**:
  1. **Ancla Principal (AI - Mediapipe)**: 
     - Se utiliza la malla facial de `mediapipe` para detectar 478 puntos de referencia en alta precisión.
     - Se extraen las coordenadas exactas de los ojos (puntos del contorno ocular y pupila).
  2. **Validación de "Ojo Abierto"**:
     - Se calcula el **Eye Aspect Ratio (EAR)**: La relación entre la altura y el ancho del ojo.
     - Si el ojo está demasiado cerrado, el pipeline se detiene para evitar recortar un párpado cerrado en lugar de la conjuntiva.
  3. **Mecanismos de Respaldo (Fallbacks)**:
     - Si Mediapipe falla (ej. rostro incompleto en la foto), se activa un detector secundario basado en **Haar Cascades** (visión computacional clásica).
     - Si Haar también falla, se realiza un recorte dinámico central como último recurso, asegurando que ninguna imagen se pierda por error del sistema.
  4. **Aislamiento de la Conjuntiva (Recorte y Forma)**:
     - Basado en los puntos inferiores del ojo detectado, se calcula una región de interés (ROI).
     - **Validación de Forma ("Media Luna")**: Se aplica lógica morfológica para confirmar que la región recortada tiene la curvatura característica de la conjuntiva palpebral. Si es muy cuadrada o rectangular, se ajustan los factores de recorte (Factores X/Y) hasta obtener el área correcta.
- **Objetivo Final**: Obtener una imagen limpia de `224x224` píxeles que contiene *exclusivamente* la conjuntiva, lista para el diagnóstico de anemia.

### Paso C: Clasificación de Anemia
- **Función**: `modelo_nfnet`
- **Lógica**: La imagen segmentada pasa por la red neuronal para obtener la probabilidad de anemia.
- **Técnica**: **NFNet-F0** (Normalizer-Free Network).
- **Objetivo**: Predicción binaria (Con Anemia / Sin Anemia) con alta precisión sin capas de normalización.

### Paso D: Generación de Explicabilidad
- **Función**: `generar_smoothgrad` (explicabilidad.py)
- **Lógica**: Se generan múltiples versiones con ruido de la imagen para promediar los gradientes y obtener un mapa de calor estable.
- **Técnica**: **SmoothGrad**.
- **Objetivo**: Visualizar qué píxeles de la conjuntiva determinaron el diagnóstico.

---

## 2. Cálculo de Indicadores de Explicabilidad

Se implementan 5 indicadores clave para validar la calidad de la IA en el ámbito clínico:

### 1. Nivel de Detalle (RCAP)
- **Lógica**: Relación entre la importancia visual y la confianza del modelo.
- **Objetivo**: Validar si el modelo "mira" lo que un médico miraría.

### 2. Exactitud de Áreas (P)
- **Lógica**: Intersección entre el mapa de calor y la máscara real de la conjuntiva.
- **Objetivo**: Medir qué tanto del mapa de calor se sale de la zona de interés (falsos positivos visuales).

### 3. Robustez de los Resultados (RG)
- **Función**: `calcular_robustez_imagen` (robustez.py)
- **Lógica**: Mide la distancia mínima al límite de decisión entre clases.
- **Objetivo**: Evaluar la estabilidad de la predicción ante pequeñas variaciones en la imagen.

### 4. Visibilidad de Características Claras (NT)
- **Función**: `calcular_transparencia_diagnostico` (transparencia.py)
- **Lógica**: Compara el mapa de **SmoothGrad** con un mapa de **SHAP** (Kernel Explainer).
- **Objetivo**: Validar el consenso entre dos técnicas matemáticas diferentes de explicabilidad.

### 5. Sensibilidad de la Explicabilidad (S)
- **Función**: `calcular_sensibilidad_explicabilidad` (sensibilidad.py)
- **Lógica**: Calcula el gradiente de la explicación respecto a perturbaciones mínimas ($\epsilon$).
- **Objetivo**: Medir qué tan resistente es el mapa de calor frente a ruido técnico en la captura.

---

## 3. Evaluación Grupal (Batch Processing)

### Flujo de Lote:
- **Función**: `evaluar_indicadores` (views.py)
- **Proceso**:
    1. Carga el dataset de validación completo.
    2. Ejecuta los pasos A, B, C y D para cada imagen.
    3. Calcula los 5 indicadores individuales para cada prueba.
    4. **Agregación**: Obtiene el promedio global de cada métrica (ej. el valor **D** global).
- **Resultado Final**: Un reporte estadístico que valida la confiabilidad del modelo NFNet en condiciones reales de laboratorio.
