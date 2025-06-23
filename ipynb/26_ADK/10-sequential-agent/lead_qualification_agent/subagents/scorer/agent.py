"""
Agente Calificador de Leads

Este agente es responsable de calificar el nivel de cualificación de un lead
basado en varios criterios.
"""

from google.adk.agents import LlmAgent

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Crear el agente calificador
lead_scorer_agent = LlmAgent(
    name="LeadScorerAgent",
    model=GEMINI_MODEL,
    instruction="""Eres una IA de Calificación de Leads.
    
    Analiza la información del lead y asigna una puntuación de calificación del 1 al 10 basada en:
    - Necesidad expresada (urgencia/claridad del problema)
    - Autoridad para tomar decisiones
    - Indicadores de presupuesto
    - Indicadores de cronograma
    
    Genera SOLAMENTE una puntuación numérica y UNA frase de justificación.
    
    Ejemplo de salida: '8: Tomador de decisiones con presupuesto claro y necesidad inmediata'
    Ejemplo de salida: '3: Interés vago sin mención de cronograma o presupuesto'
    """,
    description="Califica leads cualificados en una escala del 1 al 10.",
    output_key="lead_score",
)
