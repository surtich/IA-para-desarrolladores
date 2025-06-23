import os
import random

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# https://docs.litellm.ai/docs/providers/cohere
model = LiteLlm(
    model="cohere/command-a-03-2025",
    api_key=os.getenv("COHERE_API_KEY")
)

def get_dad_joke():
    """Obtiene un chiste de papá aleatorio.

    Returns:
        str: Un chiste de papá.
    """
    jokes = [
        "¿Por qué el pollo cruzó la carretera? ¡Para llegar al otro lado!",
        "¿Cómo llamas a un cinturón hecho de relojes? Una pérdida de tiempo.",
        "¿Cómo llamas a los espaguetis falsos? ¡Un impasta!",
        "¿Por qué el espantapájaros ganó un premio? ¡Porque era sobresaliente en su campo!",
    ]
    return random.choice(jokes)


root_agent = Agent(
    name="dad_joke_agent",
    model=model,
    description="Agente de chistes de papá",
    instruction="""
    Eres un asistente útil que puede contar chistes de papá. 
    Solo usa la herramienta `get_dad_joke` para contar chistes.
    """,
    tools=[get_dad_joke],
)
