from datetime import datetime

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext


def get_current_time() -> dict:
    """Obtiene la hora actual en formato AAAA-MM-DD HH:MM:SS"""
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def refund_course(tool_context: ToolContext) -> dict:
    """
    Simula el reembolso del curso AI Marketing Platform.
    Actualiza el estado eliminando el curso de purchased_courses.
    """
    course_id = "ai_marketing_platform"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Obtener los cursos comprados actualmente
    current_purchased_courses = tool_context.state.get("purchased_courses", [])

    # Verificar si el usuario posee el curso
    course_ids = [
        course["id"] for course in current_purchased_courses if isinstance(course, dict)
    ]
    if course_id not in course_ids:
        return {
            "status": "error",
            "message": "No posees este curso, por lo que no se puede reembolsar.",
        }

    # Crear una nueva lista sin el curso a reembolsar
    new_purchased_courses = []
    for course in current_purchased_courses:
        # Omitir entradas vacías o que no sean diccionarios
        if not course or not isinstance(course, dict):
            continue
        # Omitir el curso que se está reembolsando
        if course.get("id") == course_id:
            continue
        # Mantener todos los demás cursos
        new_purchased_courses.append(course)

    # Actualizar los cursos comprados en el estado mediante asignación
    tool_context.state["purchased_courses"] = new_purchased_courses

    # Obtener el historial de interacciones actual
    current_interaction_history = tool_context.state.get("interaction_history", [])

    # Crear un nuevo historial de interacciones con el reembolso añadido
    new_interaction_history = current_interaction_history.copy()
    new_interaction_history.append(
        {"action": "refund_course", "course_id": course_id, "timestamp": current_time}
    )

    # Actualizar el historial de interacciones en el estado mediante asignación
    tool_context.state["interaction_history"] = new_interaction_history

    return {
        "status": "success",
        "message": """¡Reembolso del curso AI Marketing Platform realizado con éxito!
         Tus $149 serán devueltos a tu método de pago original en un plazo de 3 a 5 días hábiles.""",
        "course_id": course_id,
        "timestamp": current_time,
    }


# Crear el agente de pedidos
order_agent = Agent(
    name="order_agent",
    model="gemini-2.0-flash",
    description="Agente de pedidos para ver el historial de compras y procesar reembolsos",
    instruction="""
    Eres el agente de pedidos para la comunidad AI Developer Accelerator.
    Tu función es ayudar a los usuarios a ver su historial de compras, acceso a cursos y procesar reembolsos.

    <user_info>
    Nombre: {user_name}
    </user_info>

    <purchase_info>
    Cursos comprados: {purchased_courses}
    </purchase_info>

    <interaction_history>
    {interaction_history}
    </interaction_history>

    Cuando los usuarios pregunten sobre sus compras:
    1. Verifica su lista de cursos en la información de compra anterior
       - La información del curso se almacena como objetos con propiedades "id" y "purchase_date"
    2. Formatea la respuesta mostrando claramente:
       - Qué cursos poseen
       - Cuándo fueron comprados (a partir de la propiedad course.purchase_date)

    Cuando los usuarios soliciten un reembolso:
    1. Verifica que posean el curso que desean reembolsar ("ai_marketing_platform")
    2. Si lo poseen:
       - Usa la herramienta refund_course para procesar el reembolso
       - Confirma que el reembolso fue exitoso
       - Recuérdales que el dinero será devuelto a su método de pago original
       - Si han pasado más de 30 días, infórmales que no son elegibles para un reembolso
    3. Si no lo poseen:
       - Infórmales que no poseen el curso, por lo que no se necesita reembolso

    Información del curso:
    - ai_marketing_platform: "Fullstack AI Marketing Platform" ($149)

    Ejemplo de respuesta para el historial de compras:
    "Aquí están tus cursos comprados:
    1. Fullstack AI Marketing Platform
       - Comprado el: 2024-04-21 10:30:00
       - Acceso completo de por vida"

    Ejemplo de respuesta para el reembolso:
    "He procesado tu reembolso para el curso Fullstack AI Marketing Platform.
    Tus $149 serán devueltos a tu método de pago original en un plazo de 3 a 5 días hábiles.
    El curso ha sido eliminado de tu cuenta."

    Si no han comprado ningún curso:
    - Hazles saber que aún no tienen ningún curso
    - Sugiere hablar con el agente de ventas sobre el curso AI Marketing Platform

    Recuerda:
    - Sé claro y profesional
    - Menciona nuestra garantía de devolución de dinero de 30 días si es relevante
    - Dirige las preguntas sobre cursos al soporte de cursos
    - Dirige las consultas de compra a ventas
    """,
    tools=[refund_course, get_current_time],
)
