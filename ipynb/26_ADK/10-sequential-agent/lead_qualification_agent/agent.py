"""
Agente Secuencial con un Callback Mínimo

Este ejemplo demuestra un pipeline de calificación de leads con un
before_agent_callback mínimo que solo inicializa el estado una vez al principio.
"""

from google.adk.agents import SequentialAgent

from .subagents.recommender import action_recommender_agent
from .subagents.scorer import lead_scorer_agent

# Importar los subagentes
from .subagents.validator import lead_validator_agent

# Crear el agente secuencial con un callback mínimo
root_agent = SequentialAgent(
    name="LeadQualificationPipeline",
    sub_agents=[lead_validator_agent, lead_scorer_agent, action_recommender_agent],
    description="Un pipeline que valida, califica y recomienda acciones para leads de ventas",
)
