"""
Ejemplo de Callbacks Antes y Después del Modelo

Este ejemplo demuestra el uso de callbacks de modelo
para filtrar contenido y registrar interacciones del modelo.
"""

import copy
from datetime import datetime
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types


def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Este callback se ejecuta antes de que el modelo procese una solicitud.
    Filtra contenido inapropiado y registra información de la solicitud.

    Args:
        callback_context: Contiene información de estado y contexto.
        llm_request: La solicitud LLM que se está enviando.

    Returns:
        LlmResponse opcional para anular la respuesta del modelo.
    """
    # Obtener el estado y el nombre del agente
    state = callback_context.state
    agent_name = callback_context.agent_name

    # Extraer el último mensaje del usuario
    last_user_message = ""
    if llm_request.contents and len(llm_request.contents) > 0:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts and len(content.parts) > 0:
                if hasattr(content.parts[0], "text") and content.parts[0].text:
                    last_user_message = content.parts[0].text
                    break

    # Registrar la solicitud
    print("=== SOLICITUD DEL MODELO INICIADA ===")
    print(f"Agente: {agent_name}")
    if last_user_message:
        print(f"Mensaje del usuario: {last_user_message[:100]}...")
        # Almacenar para uso posterior
        state["last_user_message"] = last_user_message
    else:
        print("Mensaje del usuario: <vacío>")

    print(f"Marca de tiempo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verificar contenido inapropiado
    if last_user_message and "sucks" in last_user_message.lower():
        print("=== CONTENIDO INAPROPIADO BLOQUEADO ===")
        print("Texto bloqueado que contiene palabra prohibida: 'sucks'")

        print("[ANTES DEL MODELO] ⚠️ Solicitud bloqueada debido a contenido inapropiado")

        # Devolver una respuesta para omitir la llamada al modelo
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="No puedo responder a mensajes que contengan lenguaje inapropiado. "
                        "Por favor, reformula tu solicitud sin usar palabras como 'sucks'."
                    )
                ],
            )
        )

    # Registrar la hora de inicio para el cálculo de la duración
    state["model_start_time"] = datetime.now()
    print("[ANTES DEL MODELO] ✓ Solicitud aprobada para procesamiento")

    # Devolver None para proceder con la solicitud normal del modelo
    return None


def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """
    Callback simple que reemplaza palabras negativas con alternativas más positivas.

    Args:
        callback_context: Contiene información de estado y contexto.
        llm_response: La respuesta LLM recibida.

    Returns:
        LlmResponse opcional para anular la respuesta del modelo.
    """
    # Registrar la finalización
    print("[DESPUÉS DEL MODELO] Procesando respuesta")

    # Omitir el procesamiento si la respuesta está vacía o no tiene contenido de texto
    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return None

    # Extraer texto de la respuesta
    response_text = ""
    for part in llm_response.content.parts:
        if hasattr(part, "text") and part.text:
            response_text += part.text

    if not response_text:
        return None

    # Reemplazos de palabras simples
    replacements = {
        "problem": "challenge",
        "difficult": "complex",
    }

    # Realizar reemplazos
    modified_text = response_text
    modified = False

    for original, replacement in replacements.items():
        if original in modified_text.lower():
            modified_text = modified_text.replace(original, replacement)
            modified_text = modified_text.replace(
                original.capitalize(), replacement.capitalize()
            )
            modified = True

    # Devolver la respuesta modificada si se realizaron cambios
    if modified:
        print("[DESPUÉS DEL MODELO] ↺ Texto de respuesta modificado")

        modified_parts = [copy.deepcopy(part) for part in llm_response.content.parts]
        for i, part in enumerate(modified_parts):
            if hasattr(part, "text") and part.text:
                modified_parts[i].text = modified_text

        return LlmResponse(content=types.Content(role="model", parts=modified_parts))

    # Devolver None para usar la respuesta original
    return None


# Crear el Agente
root_agent = LlmAgent(
    name="content_filter_agent",
    model="gemini-2.0-flash",
    description="Un agente que demuestra callbacks de modelo para filtrado de contenido y registro",
    instruction="""
    Eres un asistente útil.
    
    Tu trabajo es:
    - Responder a las preguntas del usuario de forma concisa
    - Proporcionar información fáctica
    - Ser amigable y respetuoso
    """,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
