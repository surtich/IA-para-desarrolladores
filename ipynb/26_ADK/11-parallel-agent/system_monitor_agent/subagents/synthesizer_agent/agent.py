"""
Agente Sintetizador de Informes del Sistema

Este agente es responsable de sintetizar información de otros agentes
para crear un informe completo de la salud del sistema.
"""

from google.adk.agents import LlmAgent

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Agente Sintetizador de Informes del Sistema
system_report_synthesizer = LlmAgent(
    name="SystemReportSynthesizer",
    model=GEMINI_MODEL,
    instruction="""Eres un Sintetizador de Informes del Sistema.
    
    Tu tarea es crear un informe completo de la salud del sistema combinando información de:
    - Información de la CPU: {cpu_info}
    - Información de la memoria: {memory_info}
    - Información del disco: {disk_info}
    
    Crea un informe bien formateado con:
    1. Un resumen ejecutivo en la parte superior con el estado general de salud del sistema
    2. Secciones para cada componente con su información respectiva
    3. Recomendaciones basadas en cualquier métrica preocupante
    
    Usa formato Markdown para hacer el informe legible y profesional.
    Resalta cualquier valor preocupante y proporciona recomendaciones prácticas.
    """,
    description="Sintetiza toda la información del sistema en un informe completo",
)
