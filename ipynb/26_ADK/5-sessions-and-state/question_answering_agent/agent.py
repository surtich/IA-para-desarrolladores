from google.adk.agents import Agent

# Create the root agent
question_answering_agent = Agent(
    name="question_answering_agent",
    model="gemini-2.0-flash",
    description="Agente de respuesta a preguntas",
    instruction="""
    Eres un asistente útil que responde preguntas sobre las preferencias del usuario.

    Aquí tienes información sobre el usuario:
    Nombre: 
    {user_name}
    Preferencias: 
    {user_preferences}
    """,
)
