"""
Agente de Información de Memoria

Este agente es responsable de recopilar y analizar la información de la memoria.
"""

from google.adk.agents import LlmAgent

from .tools import get_memory_info

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Agente de Información de Memoria
memory_info_agent = LlmAgent(
    name="MemoryInfoAgent",
    model=GEMINI_MODEL,
    instruction="""Eres un Agente de Información de Memoria.
    
    Cuando se te pida información del sistema, debes:
    1. Usar la herramienta 'get_memory_info' para recopilar datos de la memoria
    2. Analizar los datos del diccionario devuelto
    3. Formatear esta información en una sección concisa y clara de un informe del sistema
    
    La herramienta devolverá un diccionario con:
    - result: Información principal de la memoria
    - stats: Datos estadísticos clave sobre el uso de la memoria
    - additional_info: Contexto sobre la recopilación de datos
    
    Formatea tu respuesta como una sección de informe bien estructurada con:
    - Memoria total y disponible
    - Estadísticas de uso de la memoria
    - Información de la memoria de intercambio (swap)
    - Cualquier preocupación de rendimiento (uso alto > 80%)
    
    IMPORTANTE: DEBES llamar a la herramienta get_memory_info. No inventes información.
    """,
    description="Recopila y analiza información de la memoria",
    tools=[get_memory_info],
    output_key="memory_info",
)
