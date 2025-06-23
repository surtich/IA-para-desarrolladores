"""
Ejemplo de Callbacks Antes y Después del Agente

Este ejemplo demuestra cómo usar tanto before_agent_callback como after_agent_callback
para propósitos de registro.
"""

from datetime import datetime
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types


def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Callback simple que registra cuando el agente comienza a procesar una solicitud.

    Args:
        callback_context: Contiene información de estado y contexto.

    Returns:
        None para continuar con el procesamiento normal del agente.
    """
    # Obtener el estado de la sesión
    state = callback_context.state

    # Registrar la marca de tiempo
    timestamp = datetime.now()

    # Establecer el nombre del agente si no está presente
    if "agent_name" not in state:
        state["agent_name"] = "SimpleChatBot"

    # Inicializar el contador de solicitudes
    if "request_counter" not in state:
        state["request_counter"] = 1
    else:
        state["request_counter"] += 1

    # Almacenar la hora de inicio para el cálculo de la duración en after_agent_callback
    state["request_start_time"] = timestamp

    # Registrar la solicitud
    print("=== EJECUCIÓN DEL AGENTE INICIADA ===")
    print(f"Solicitud #: {state['request_counter']}")
    print(f"Marca de tiempo: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Imprimir en consola
    print(f"\n[CALLBACK ANTES] Agente procesando solicitud #{state['request_counter']}")

    return None


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Callback simple que registra cuando el agente termina de procesar una solicitud.

    Args:
        callback_context: Contiene información de estado y contexto.

    Returns:
        None para continuar con el procesamiento normal del agente.
    """
    # Obtener el estado de la sesión
    state = callback_context.state

    # Calcular la duración de la solicitud si el tiempo de inicio está disponible
    timestamp = datetime.now()
    duration = None
    if "request_start_time" in state:
        duration = (timestamp - state["request_start_time"]).total_seconds()

    # Registrar la finalización
    print("=== EJECUCIÓN DEL AGENTE COMPLETADA ===")
    print(f"Solicitud #: {state.get('request_counter', 'Desconocido')}")
    if duration is not None:
        print(f"Duración: {duration:.2f} segundos")

    # Imprimir en consola
    print(
        f"[CALLBACK DESPUÉS] Agente completó la solicitud #{state.get('request_counter', 'Desconocido')}"
    )
    if duration is not None:
        print(f"[CALLBACK DESPUÉS] El procesamiento tomó {duration:.2f} segundos")

    return None


# Crear el Agente
root_agent = LlmAgent(
    name="before_after_agent",
    model="gemini-2.0-flash",
    description="Un agente básico que demuestra callbacks antes y después del agente",
    instruction="""
    Eres un agente de saludo amigable. Tu nombre es {agent_name}.
    
    Tu trabajo es:
    - Saludar a los usuarios cortésmente
    - Responder a preguntas básicas
    - Mantener tus respuestas amigables y concisas
    """,
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
)
