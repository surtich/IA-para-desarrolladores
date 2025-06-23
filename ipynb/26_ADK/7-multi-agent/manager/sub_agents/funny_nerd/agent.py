from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext


def get_nerd_joke(topic: str, tool_context: ToolContext) -> dict:
    """Obtiene un chiste friki sobre un tema específico."""
    print(f"--- Herramienta: get_nerd_joke llamada para el tema: {topic} ---")

    # Chistes de ejemplo - en una implementación real, podrías querer usar una API
    jokes = {
        "python": "¿Por qué a los programadores de Python no les gusta usar la herencia? ¡Porque no les gusta heredar nada!",
        "javascript": "¿Por qué el desarrollador de JavaScript se arruinó? ¡Porque agotó toda su caché!",
        "java": "¿Por qué los desarrolladores de Java usan gafas? ¡Porque no pueden ver C#!",
        "programming": "¿Por qué los programadores prefieren el modo oscuro? ¡Porque la luz atrae a los bichos!",
        "math": "¿Por qué el signo igual era tan humilde? ¡Porque sabía que no era ni menos que ni mayor que nadie más!",
        "physics": "¿Por qué el fotón se registró en un hotel? ¡Porque viajaba ligero!",
        "chemistry": "¿Por qué el ácido fue al gimnasio? ¡Para convertirse en una solución tampón!",
        "biology": "¿Por qué la célula fue a terapia? ¡Porque tenía demasiados problemas!",
        "default": "¿Por qué la computadora fue al médico? ¡Porque tenía un virus!",
    }

    joke = jokes.get(topic.lower(), jokes["default"])

    # Actualizar el estado con el último tema del chiste
    tool_context.state["last_joke_topic"] = topic

    return {"status": "éxito", "joke": joke, "topic": topic}


# Crear el agente friki divertido
funny_nerd = Agent(
    name="funny_nerd",
    model="gemini-2.0-flash",
    description="Un agente que cuenta chistes frikis sobre varios temas.",
    instruction="""
    Eres un agente friki divertido que cuenta chistes frikis sobre varios temas.
    
    Cuando se te pida que cuentes un chiste:
    1. Usa la herramienta get_nerd_joke para obtener un chiste sobre el tema solicitado
    2. Si no se menciona un tema específico, pregunta al usuario qué tipo de chiste friki le gustaría escuchar
    3. Formatea la respuesta para incluir tanto el chiste como una breve explicación si es necesario
    
    Los temas disponibles incluyen:
    - python
    - javascript
    - java
    - programming
    - math
    - physics
    - chemistry
    - biology
    
    Formato de respuesta de ejemplo:
    "Aquí tienes un chiste friki sobre <TEMA>:
    <CHISTE>
    
    Explicación: {breve explicación si es necesaria}"

    Si el usuario pregunta sobre cualquier otra cosa,
    debes delegar la tarea al agente gestor.
    """,
    tools=[get_nerd_joke],
)
