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


async def display_state(
    session_service, app_name, user_id, session_id, label="Estado Actual"
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

        # Manejar recordatorios
        reminders = session.state.get("reminders", [])
        if reminders:
            print("📝 Recordatorios:")
            for idx, reminder in enumerate(reminders, 1):
                print(f"  {idx}. {reminder}")
        else:
            print("📝 Recordatorios: Ninguno")

        print("-" * (22 + len(label)))
    except Exception as e:
        print(f"Error al mostrar el estado: {e}")


async def process_agent_response(event):
    """Procesa y muestra los eventos de respuesta del agente."""
    # Registrar información básica del evento
    print(f"ID del Evento: {event.id}, Autor: {event.author}")

    # Comprobar partes específicas primero
    has_specific_part = False
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "executable_code") and part.executable_code:
                # Acceder a la cadena de código real a través de .code
                print(
                    f"  Depuración: Agente generó código:\n``````"
                )
                has_specific_part = True

            elif hasattr(part, "code_execution_result") and part.code_execution_result:
                # Acceder al resultado y la salida correctamente
                print(
                    f"  Depuración: Resultado de Ejecución de Código: {part.code_execution_result.outcome} - Salida:\n{part.code_execution_result.output}"
                )
                has_specific_part = True

            elif hasattr(part, "tool_response") and part.tool_response:
                # Imprimir información de respuesta de la herramienta
                print(f"  Respuesta de la Herramienta: {part.tool_response.output}")
                has_specific_part = True

            elif hasattr(part, "function_call") and part.function_call:
                # Procesar la llamada a función
                func = part.function_call
                print(f"  Llamada a función detectada:")
                print(f"    Nombre de la función: {func.name}")
                print(f"    Argumentos: {func.args}")
                # Aquí puedes ejecutar la función correspondiente en tu código si lo deseas
                has_specific_part = True

            # También imprimir cualquier parte de texto encontrada en cualquier evento para depuración
            elif hasattr(part, "text") and part.text and not part.text.isspace():
                print(f"  Texto: '{part.text.strip()}'")


    # Comprobar la respuesta final después de las partes específicas
    final_response = None
    if event.is_final_response():
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
                f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}==> Respuesta Final del Agente: [No hay contenido de texto en el evento final]{Colors.RESET}\n"
            )

    return final_response


async def call_agent_async(runner, user_id, session_id, query):
    """Llama al agente asincrónicamente con la consulta del usuario."""
    content = types.Content(role="user", parts=[types.Part(text=query)])
    print(
        f"\n{Colors.BG_GREEN}{Colors.BLACK}{Colors.BOLD}--- Ejecutando Consulta: {query} ---{Colors.RESET}"
    )
    final_response_text = None

    # Mostrar estado antes de procesar
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
            # Procesar cada evento y obtener la respuesta final si está disponible
            response = await process_agent_response(event)
            if response:
                final_response_text = response
    except Exception as e:
        print(f"Error durante la llamada al agente: {e}")

    # Mostrar estado después de procesar el mensaje
    await display_state(
        runner.session_service,
        runner.app_name,
        user_id,
        session_id,
        "Estado DESPUÉS de procesar",
    )

    return final_response_text
