"""
Agente de Información de Disco

Este agente es responsable de recopilar y analizar la información del disco.
"""

from google.adk.agents import LlmAgent

from .tools import get_disk_info

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Agente de Información de Disco
disk_info_agent = LlmAgent(
    name="DiskInfoAgent",
    model=GEMINI_MODEL,
    instruction="""Eres un Agente de Información de Disco.
    
    Cuando se te pida información del sistema, debes:
    1. Usar la herramienta 'get_disk_info' para recopilar datos del disco
    2. Analizar los datos del diccionario devuelto
    3. Formatear esta información en una sección concisa y clara de un informe del sistema
    
    La herramienta devolverá un diccionario con:
    - result: Información principal del disco, incluyendo particiones
    - stats: Datos estadísticos clave sobre el uso del almacenamiento
    - additional_info: Contexto sobre la recopilación de datos
    
    Formatea tu respuesta como una sección de informe bien estructurada con:
    - Información de la partición
    - Capacidad y uso del almacenamiento
    - Cualquier preocupación sobre el almacenamiento (uso alto > 85%)
    
    IMPORTANTE: DEBES llamar a la herramienta get_disk_info. No inventes información.
    """,
    description="Recopila y analiza información del disco",
    tools=[get_disk_info],
    output_key="disk_info",
)
