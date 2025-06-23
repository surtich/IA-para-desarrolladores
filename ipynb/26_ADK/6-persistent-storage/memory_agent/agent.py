from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext


def add_reminder(reminder: str, tool_context: ToolContext) -> dict:
    """Añade un nuevo recordatorio a la lista de recordatorios del usuario.

    Args:
        reminder: El texto del recordatorio a añadir
        tool_context: Contexto para acceder y actualizar el estado de la sesión

    Returns:
        Un mensaje de confirmación
    """
    print(f"--- Herramienta: add_reminder llamada para '{reminder}' ---")

    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])

    # Add the new reminder
    reminders.append(reminder)

    # Update state with the new list of reminders
    tool_context.state["reminders"] = reminders

    return {
        "action": "add_reminder",
        "reminder": reminder,
        "message": f"Recordatorio añadido: {reminder}",
    }


def view_reminders(tool_context: ToolContext) -> dict:
    """Ver todos los recordatorios actuales.

    Args:
        tool_context: Contexto para acceder al estado de la sesión

    Returns:
        La lista de recordatorios
    """
    print("--- Herramienta: view_reminders llamada ---")

    # Get reminders from state
    reminders = tool_context.state.get("reminders", [])

    return {"action": "view_reminders", "reminders": reminders, "count": len(reminders)}


def update_reminder(index: int, updated_text: str, tool_context: ToolContext) -> dict:
    """Actualiza un recordatorio existente.

    Args:
        index: El índice basado en 1 del recordatorio a actualizar
        updated_text: El nuevo texto para el recordatorio
        tool_context: Contexto para acceder y actualizar el estado de la sesión

    Returns:
        Un mensaje de confirmación
    """
    print(
        f"--- Herramienta: update_reminder llamada para el índice {index} con '{updated_text}' ---"
    )

    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])

    # Check if the index is valid
    if not reminders or index < 1 or index > len(reminders):
        return {
            "action": "update_reminder",
            "status": "error",
            "message": f"No se pudo encontrar el recordatorio en la posición {index}. Actualmente hay {len(reminders)} recordatorios.",
        }

    # Update the reminder (adjusting for 0-based indices)
    old_reminder = reminders[index - 1]
    reminders[index - 1] = updated_text

    # Update state with the modified list
    tool_context.state["reminders"] = reminders

    return {
        "action": "update_reminder",
        "index": index,
        "old_text": old_reminder,
        "updated_text": updated_text,
        "message": f"Recordatorio {index} actualizado de '{old_reminder}' a '{updated_text}'",
    }


def delete_reminder(index: int, tool_context: ToolContext) -> dict:
    """Elimina un recordatorio.

    Args:
        index: El índice basado en 1 del recordatorio a eliminar
        tool_context: Contexto para acceder y actualizar el estado de la sesión

    Returns:
        Un mensaje de confirmación
    """
    print(f"--- Herramienta: delete_reminder llamada para el índice {index} ---")

    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])

    # Check if the index is valid
    if not reminders or index < 1 or index > len(reminders):
        return {
            "action": "delete_reminder",
            "status": "error",
            "message": f"No se pudo encontrar el recordatorio en la posición {index}. Actualmente hay {len(reminders)} recordatorios.",
        }

    # Remove the reminder (adjusting for 0-based indices)
    deleted_reminder = reminders.pop(index - 1)

    # Update state with the modified list
    tool_context.state["reminders"] = reminders

    return {
        "action": "delete_reminder",
        "index": index,
        "deleted_reminder": deleted_reminder,
        "message": f"Recordatorio {index} eliminado: '{deleted_reminder}'",
    }


def update_user_name(name: str, tool_context: ToolContext) -> dict:
    """Actualiza el nombre del usuario.

    Args:
        name: El nuevo nombre para el usuario
        tool_context: Contexto para acceder y actualizar el estado de la sesión

    Returns:
        Un mensaje de confirmación
    """
    print(f"--- Herramienta: update_user_name llamada con '{name}' ---")

    # Get current name from state
    old_name = tool_context.state.get("user_name", "")

    # Update the name in state
    tool_context.state["user_name"] = name

    return {
        "action": "update_user_name",
        "old_name": old_name,
        "new_name": name,
        "message": f"Tu nombre ha sido actualizado a: {name}",
    }


# Create a simple persistent agent
memory_agent = Agent(
    name="memory_agent",
    model="gemini-2.0-flash",
    description="Un agente de recordatorios inteligente con memoria persistente",
    instruction="""
    Eres un amigable asistente de recordatorios que recuerda a los usuarios a través de las conversaciones.

    La información del usuario se almacena en el estado:
    - Nombre del usuario: {user_name}
    - Recordatorios: {reminders}

    Puedes ayudar a los usuarios a gestionar sus recordatorios con las siguientes capacidades:
    1. Añadir nuevos recordatorios
    2. Ver recordatorios existentes
    3. Actualizar recordatorios
    4. Eliminar recordatorios
    5. Actualizar el nombre del usuario

    Sé siempre amigable y dirígete al usuario por su nombre. Si aún no conoces su nombre,
    usa la herramienta update_user_name para almacenarlo cuando se presenten.

    **DIRECTRICES DE GESTIÓN DE RECORDATORIOS:**

    Al tratar con recordatorios, debes ser inteligente para encontrar el recordatorio correcto:

    1. Cuando el usuario pide actualizar o eliminar un recordatorio pero no proporciona un índice:
       - Si mencionan el contenido del recordatorio (ej. "eliminar mi recordatorio de reunión"), 
         busca en los recordatorios para encontrar una coincidencia
       - Si encuentras una coincidencia exacta o cercana, usa ese índice
       - Nunca aclares a qué recordatorio se refiere el usuario, simplemente usa la primera coincidencia
       - Si no se encuentra ninguna coincidencia, lista todos los recordatorios y pide al usuario que especifique

    2. Cuando el usuario menciona un número o posición:
       - Usa eso como el índice (ej. "eliminar recordatorio 2" significa índice=2)
       - Recuerda que la indexación comienza en 1 para el usuario

    3. Para posiciones relativas:
       - Maneja "primero", "último", "segundo", etc. apropiadamente
       - "Primer recordatorio" = índice 1
       - "Último recordatorio" = el índice más alto
       - "Segundo recordatorio" = índice 2, y así sucesivamente

    4. Para ver:
       - Siempre usa la herramienta view_reminders cuando el usuario pida ver sus recordatorios
       - Formatea la respuesta en una lista numerada para mayor claridad
       - Si no hay recordatorios, sugiere añadir algunos

    5. Para añadir:
       - Extrae el texto real del recordatorio de la solicitud del usuario
       - Elimina frases como "añadir un recordatorio para" o "recuérdame que"
       - Concéntrate en la tarea en sí (ej. "añadir un recordatorio para comprar leche" → add_reminder("comprar leche"))

    6. Para actualizaciones:
       - Identifica tanto qué recordatorio actualizar como cuál debe ser el nuevo texto
       - Por ejemplo, "cambiar mi segundo recordatorio a recoger la compra" → update_reminder(2, "recoger la compra")

    7. Para eliminaciones:
       - Confirma la eliminación cuando esté completa y menciona qué recordatorio se eliminó
       - Por ejemplo, "He eliminado tu recordatorio para 'comprar leche'"

    Recuerda explicar que puedes recordar su información a través de las conversaciones.

    IMPORTANTE:
    - usa tu mejor juicio para determinar a qué recordatorio se refiere el usuario. 
    - No tienes que ser 100% correcto, pero intenta ser lo más cercano posible.
    - Nunca le pidas al usuario que aclare a qué recordatorio se refiere.
    """,
    tools=[
        add_reminder,
        view_reminders,
        update_reminder,
        delete_reminder,
        update_user_name,
    ],
)
