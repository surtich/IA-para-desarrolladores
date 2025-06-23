from google.adk.agents import Agent

root_agent = Agent(
    name="greeting_agent",
    # https://ai.google.dev/gemini-api/docs/models
    model="gemini-2.0-flash",
    description="Agente de saludo",
    instruction="""
    Eres un asistente útil que saluda al usuario. 
    Pregunta el nombre del usuario y salúdalo por su nombre.
    """,
)
