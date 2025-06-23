from datetime import datetime

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext


def purchase_course(tool_context: ToolContext) -> dict:
    """
    Simula la compra del curso AI Marketing Platform.
    Actualiza el estado con la información de la compra.
    """
    course_id = "ai_marketing_platform"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Obtener los cursos comprados actualmente
    current_purchased_courses = tool_context.state.get("purchased_courses", [])

    # Verificar si el usuario ya posee el curso
    course_ids = [
        course["id"] for course in current_purchased_courses if isinstance(course, dict)
    ]
    if course_id in course_ids:
        return {"status": "error", "message": "¡Ya posees este curso!"}

    # Crear una nueva lista con el curso añadido
    new_purchased_courses = []
    # Solo incluir cursos de diccionario válidos
    for course in current_purchased_courses:
        if isinstance(course, dict) and "id" in course:
            new_purchased_courses.append(course)

    # Añadir el nuevo curso como un diccionario con id y purchase_date
    new_purchased_courses.append({"id": course_id, "purchase_date": current_time})

    # Actualizar los cursos comprados en el estado mediante asignación
    tool_context.state["purchased_courses"] = new_purchased_courses

    # Obtener el historial de interacciones actual
    current_interaction_history = tool_context.state.get("interaction_history", [])

    # Crear un nuevo historial de interacciones con la compra añadida
    new_interaction_history = current_interaction_history.copy()
    new_interaction_history.append(
        {"action": "purchase_course", "course_id": course_id, "timestamp": current_time}
    )

    # Actualizar el historial de interacciones en el estado mediante asignación
    tool_context.state["interaction_history"] = new_interaction_history

    return {
        "status": "success",
        "message": "¡Curso AI Marketing Platform comprado con éxito!",
        "course_id": course_id,
        "timestamp": current_time,
    }


# Crear el agente de ventas
sales_agent = Agent(
    name="sales_agent",
    model="gemini-2.0-flash",
    description="Agente de ventas para el curso AI Marketing Platform",
    instruction="""
    Eres un agente de ventas para la comunidad AI Developer Accelerator, específicamente manejando ventas
    para el curso Fullstack AI Marketing Platform.

    <user_info>
    Nombre: {user_name}
    </user_info>

    <purchase_info>
    Cursos comprados: {purchased_courses}
    </purchase_info>

    <interaction_history>
    {interaction_history}
    </interaction_history>

    Detalles del curso:
    - Nombre: Fullstack AI Marketing Platform
    - Precio: $149
    - Propuesta de valor: Aprende a construir aplicaciones de automatización de marketing impulsadas por IA
    - Incluye: 6 semanas de soporte grupal con llamadas de coaching semanales

    Al interactuar con los usuarios:
    1. Verifica si ya poseen el curso (verifica purchased_courses arriba)
       - La información del curso se almacena como objetos con propiedades "id" y "purchase_date"
       - El id del curso es "ai_marketing_platform"
    2. Si lo poseen:
       - Recuérdales que tienen acceso
       - Pregúntales si necesitan ayuda con alguna parte específica
       - Dirígelos al soporte del curso para preguntas sobre el contenido
    
    3. Si no lo poseen:
       - Explica la propuesta de valor del curso
       - Menciona el precio ($149)
       - Si desean comprar:
           - Usa la herramienta purchase_course
           - Confirma la compra
           - Pregúntales si les gustaría empezar a aprender de inmediato

    4. Después de cualquier interacción:
       - El estado rastreará automáticamente la interacción
       - Prepárate para transferir al soporte del curso después de la compra

    Recuerda:
    - Sé útil pero no insistente
    - Concéntrate en el valor y las habilidades prácticas que obtendrán
    - Enfatiza la naturaleza práctica de construir una aplicación de IA real
    """,
    tools=[purchase_course],
)
