from datetime import datetime

from google.genai import types


# Códigos de color ANSI para la salida de la terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Colores de primer plano
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Colores de fondo
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


async def update_interaction_history(session_service, app_name, user_id, session_id, entry):
    """Añade una entrada al historial de interacciones en el estado.

    Args:
        session_service: La instancia del servicio de sesión
        app_name: El nombre de la aplicación
        user_id: El ID de usuario
        session_id: El ID de sesión
        entry: Un diccionario que contiene los datos de la interacción
            - requiere la clave 'action' (ej., 'user_query', 'agent_response')
            - otras claves son flexibles dependiendo del tipo de acción
    """
    try:
        # Obtener la sesión actual
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        # Obtener el historial de interacciones actual
        interaction_history = session.state.get("interaction_history", [])

        # Añadir marca de tiempo si no está presente
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Añadir la entrada al historial de interacciones
        interaction_history.append(entry)

        # Crear estado actualizado
        updated_state = session.state.copy()
        updated_state["interaction_history"] = interaction_history

        # Crear una nueva sesión con el estado actualizado
        session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=updated_state,
        )
    except Exception as e:
        print(f"Error al actualizar el historial de interacciones: {e}")


async def add_user_query_to_history(session_service, app_name, user_id, session_id, query):
    """Añade una consulta de usuario al historial de interacciones."""
    await update_interaction_history(
        session_service,
        app_name,
        user_id,
        session_id,
        {
            "action": "user_query",
            "query": query,
        },
    )


async def add_agent_response_to_history(
    session_service, app_name, user_id, session_id, agent_name, response
):
    """Añade una respuesta del agente al historial de interacciones."""
    await update_interaction_history(
        session_service,
        app_name,
        user_id,
        session_id,
        {
            "action": "agent_response",
            "agent": agent_name,
            "response": response,
        },
    )


async def display_state(
    session_service, app_name, user_id, session_id, label="Current State"
):
    """Muestra el estado actual de la sesión de forma formateada."""
    try:
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        # Formatear la salida con secciones claras
        print(f"\n{'-' * 10} {label} {'-' * 10}")

        # Manejar el nombre de usuario
        user_name = session.state.get("user_name", "Desconocido")
        print(f"👤 Usuario: {user_name}")

        # Manejar los cursos comprados
        purchased_courses = session.state.get("purchased_courses", [])
        if purchased_courses and any(purchased_courses):
            print("📚 Cursos:")
            for course in purchased_courses:
                if isinstance(course, dict):
                    course_id = course.get("id", "Desconocido")
                    purchase_date = course.get("purchase_date", "Fecha desconocida")
                    print(f"  - {course_id} (comprado el {purchase_date})")
                elif course:  # Manejar formato de cadena para compatibilidad con versiones anteriores
                    print(f"  - {course}")
        else:
            print("📚 Cursos: Ninguno")

        # Manejar el historial de interacciones de una manera más legible
        interaction_history = session.state.get("interaction_history", [])
        if interaction_history:
            print("📝 Historial de interacciones:")
            for idx, interaction in enumerate(interaction_history, 1):
                # Formatear entradas de diccionario, o simplemente mostrar cadenas
                if isinstance(interaction, dict):
                    action = interaction.get("action", "interacción")
                    timestamp = interaction.get("timestamp", "hora desconocida")

                    if action == "user_query":
                        query = interaction.get("query", "")
                        print(f'  {idx}. Consulta del usuario a las {timestamp}: "{query}"')
                    elif action == "agent_response":
                        agent = interaction.get("agent", "desconocido")
                        response = interaction.get("response", "")
                        # Truncar respuestas muy largas para mostrar
                        if len(response) > 100:
                            response = response[:97] + "..."
                        print(f'  {idx}. Respuesta de {agent} a las {timestamp}: "{response}"')
                    else:
                        details = ", ".join(
                            f"{k}: {v}"
                            for k, v in interaction.items()
                            if k not in ["action", "timestamp"]
                        )
                        print(
                            f"  {idx}. {action} a las {timestamp}"
                            + (f" ({details})" if details else "")
                        )
                else:
                    print(f"  {idx}. {interaction}")
        else:
            print("📝 Historial de interacciones: Ninguno")

        # Mostrar cualquier clave de estado adicional que pueda existir
        other_keys = [
            k
            for k in session.state.keys()
            if k not in ["user_name", "purchased_courses", "interaction_history"]
        ]
        if other_keys:
            print("🔑 Estado adicional:")
            for key in other_keys:
                print(f"  {key}: {session.state[key]}")

        print("-" * (22 + len(label)))
    except Exception as e:
        print(f"Error al mostrar el estado: {e}")


async def process_agent_response(event):
    """Procesa y muestra los eventos de respuesta del agente."""
    print(f"ID del evento: {event.id}, Autor: {event.author}")

    # Primero, verificar partes específicas
    has_specific_part = False
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "text") and part.text and not part.text.isspace():
                print(f"  Texto: '{part.text.strip()}'")

    # Verificar la respuesta final después de las partes específicas
    final_response = None
    if not has_specific_part and event.is_final_response():
        if (
            event.content
            and event.content.parts
            and hasattr(event.content.parts[0], "text")
            and event.content.parts[0].text
        ):
            final_response = event.content.parts[0].text.strip()
            # Usar colores y formato para que la respuesta final se destaque
            print(
                f"\n{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}╔══ RESPUESTA DEL AGENTE ═════════════════════════════════════{Colors.RESET}"
            )
            print(f"{Colors.CYAN}{Colors.BOLD}{final_response}{Colors.RESET}")
            print(
                f"{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}╚═════════════════════════════════════════════════════════════{Colors.RESET}\n"
            )
        else:
            print(
                f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}==> Respuesta final del agente: [No hay contenido de texto en el evento final]{Colors.RESET}\n"
            )

    return final_response


async def call_agent_async(runner, user_id, session_id, query):
    """Llama al agente asincrónicamente con la consulta del usuario."""
    content = types.Content(role="user", parts=[types.Part(text=query)])
    print(
        f"\n{Colors.BG_GREEN}{Colors.BLACK}{Colors.BOLD}--- Ejecutando consulta: {query} ---{Colors.RESET}"
    )
    final_response_text = None
    agent_name = None

    # Mostrar el estado antes de procesar el mensaje
    await display_state(
        runner.session_service,
        runner.app_name,
        user_id,
        session_id,
        "Estado ANTES de procesar",
    )

    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            # Capturar el nombre del agente del evento si está disponible
            if event.author:
                agent_name = event.author

            response = await process_agent_response(event)
            if response:
                final_response_text = response
    except Exception as e:
        print(f"{Colors.BG_RED}{Colors.WHITE}ERROR durante la ejecución del agente: {e}{Colors.RESET}")

    # Añadir la respuesta del agente al historial de interacciones si obtuvimos una respuesta final
    if final_response_text and agent_name:
        await add_agent_response_to_history(
            runner.session_service,
            runner.app_name,
            user_id,
            session_id,
            agent_name,
            final_response_text,
        )

    # Mostrar el estado después de procesar el mensaje
    await display_state(
        runner.session_service,
        runner.app_name,
        user_id,
        session_id,
        "Estado DESPUÉS de procesar",
    )

    print(f"{Colors.YELLOW}{'-' * 30}{Colors.RESET}")
    return final_response_text
