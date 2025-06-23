"""
Agente Recomendador de Acciones

Este agente es responsable de recomendar las siguientes acciones apropiadas
basadas en la validación del lead y los resultados de la puntuación.
"""

from google.adk.agents import LlmAgent

# --- Constantes ---
GEMINI_MODEL = "gemini-2.0-flash"

# Crear el agente recomendador
action_recommender_agent = LlmAgent(
    name="ActionRecommenderAgent",
    model=GEMINI_MODEL,
    instruction="""Eres una IA de Recomendación de Acciones.
    
    Basado en la información y puntuación del lead:
    
    - Para leads inválidos: Sugiere qué información adicional se necesita
    - Para leads con puntuación de 1-3: Sugiere acciones de fomento (contenido educativo, etc.)
    - Para leads con puntuación de 4-7: Sugiere acciones de calificación (llamada de descubrimiento, evaluación de necesidades)
    - Para leads con puntuación de 8-10: Sugiere acciones de venta (demostración, propuesta, etc.)
    
    Formatea tu respuesta como una recomendación completa para el equipo de ventas.
    
    Puntuación del Lead:
    {lead_score}

    Estado de Validación del Lead:
    {validation_status}
    """,
    description="Recomienda las siguientes acciones basadas en la calificación del lead.",
    output_key="action_recommendation",
)
