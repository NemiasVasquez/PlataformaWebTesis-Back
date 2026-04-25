# Skill: Organización Modular del Backend (Django/CV)

Para mantener la cordura y que las piedras no se caigan sobre la cabeza, el proyecto sigue estas reglas de organización:

1.  **Archivos Cortos**: Ningún archivo `.py` debe superar las 150 líneas si es posible. Si crece, partir en trozos.
2.  **Capa de Servicios**: Las vistas (`views.py`) solo reciben piedras y las pasan a `services.py`. La lógica pesada NO vive en la vista.
3.  **Tareas Modularizadas**: En `tasks/preprocesamiento/`, dividir en:
    *   `core/`: Los motores pesados (clases de extracción).
    *   `validations/`: Juicios sobre la imagen (calidad, anatomía).
    *   `utils/`: Herramientas de cueva (limpiar carpetas, rutas).
    *   `logic/`: Orquestadores que usan todo lo anterior.
4.  **Scripts de Prueba**: Todo lo que sea para debug o validar debe vivir en `scripts_debug/`. No ensuciar la entrada de la cueva principal.
5.  **Lenguaje Claro**: Las funciones deben tener una descripción en lenguaje natural estándar (humano moderno) para que cualquiera entienda qué hace la piedra sin ser un chamán del código.

Sigue estas reglas y el mamut será fácil de cazar.
