"""
Agente de Información de CPU

Este agente es responsable de recopilar y analizar la información de la CPU.
"""

from google.adk.agents import LlmAgent

from .tools import get_cpu_info

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Agente de Información de CPU
cpu_info_agent = LlmAgent(
    name="CpuInfoAgent",
    model=GEMINI_MODEL,
    instruction="""Eres un Agente de Información de CPU.
    
    Cuando se te pida información del sistema, debes:
    1. Usar la herramienta 'get_cpu_info' para recopilar datos de la CPU
    2. Analizar los datos del diccionario devuelto
    3. Formatear esta información en una sección concisa y clara de un informe del sistema
    
    La herramienta devolverá un diccionario con:
    - result: Información principal de la CPU
    - stats: Datos estadísticos clave sobre el uso de la CPU
    - additional_info: Contexto sobre la recopilación de datos
    
    Formatea tu respuesta como una sección de informe bien estructurada con:
    - Información del núcleo de la CPU (físicos vs lógicos)
    - Estadísticas de uso de la CPU
    - Cualquier preocupación de rendimiento (uso alto > 80%)
    
    IMPORTANTE: DEBES llamar a la herramienta get_cpu_info. No inventes información.
    """,
    description="Recopila y analiza información de la CPU",
    tools=[get_cpu_info],
    output_key="cpu_info",
)
