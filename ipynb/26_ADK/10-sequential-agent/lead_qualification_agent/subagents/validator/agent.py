"""
Agente Validador de Leads

Este agente es responsable de validar si un lead tiene toda la información necesaria
para la calificación.
"""

from google.adk.agents import LlmAgent

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Crear el agente validador
lead_validator_agent = LlmAgent(
    name="LeadValidatorAgent",
    model=GEMINI_MODEL,
    instruction="""Eres una IA de Validación de Leads.
    
    Examina la información del lead proporcionada por el usuario y determina si está lo suficientemente completa para la calificación.
    Un lead completo debe incluir:
    - Información de contacto (nombre, correo electrónico o teléfono)
    - Alguna indicación de interés o necesidad
    - Información de la empresa o contexto si aplica
    
    Genera SOLAMENTE 'válido' o 'inválido' con una única razón si es inválido.
    
    Ejemplo de salida válida: 'válido'
    Ejemplo de salida inválida: 'inválido: falta información de contacto'
    """,
    description="Valida la información del lead para verificar su completitud.",
    output_key="validation_status",
)
