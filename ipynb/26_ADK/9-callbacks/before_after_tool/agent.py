"""
Ejemplo de Callbacks Antes y Después de la Herramienta

Este ejemplo demuestra el uso de callbacks de herramienta para modificar el comportamiento de la herramienta.
"""

import copy
from typing import Any, Dict, Optional

from google.adk.agents import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


# --- Definir una Función de Herramienta Simple ---
def get_capital_city(country: str) -> Dict[str, str]:
    """
    Recupera la capital de un país dado.

    Args:
        country: Nombre del país

    Returns:
        Diccionario con el resultado de la capital
    """
    print(f"[HERRAMIENTA] Ejecutando la herramienta get_capital_city con el país: {country}")

    country_capitals = {
        "united states": "Washington, D.C.",
        "usa": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "germany": "Berlin",
        "japan": "Tokyo",
        "brazil": "Brasília",
        "australia": "Canberra",
        "india": "New Delhi",
    }

    # Usar minúsculas para la comparación
    result = country_capitals.get(country.lower(), f"Capital no encontrada para {country}")
    print(f"[HERRAMIENTA] Resultado: {result}")
    print(f"[HERRAMIENTA] Devolviendo: {{'result': '{result}'}}")

    return {"result": result}


# --- Definir Callback Antes de la Herramienta ---
def before_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Callback simple que modifica los argumentos de la herramienta o salta la llamada a la herramienta.
    """
    tool_name = tool.name
    print(f"[Callback] Antes de la llamada a la herramienta para '{tool_name}'")
    print(f"[Callback] Argumentos originales: {args}")

    # Si alguien pregunta por 'Merica, convertir a Estados Unidos
    if tool_name == "get_capital_city" and args.get("country", "").lower() == "merica":
        print("[Callback] Convirtiendo América a 'United States'")
        args["country"] = "United States"
        print(f"[Callback] Argumentos modificados: {args}")
        return None

    # Saltar la llamada por completo para países restringidos
    if (
        tool_name == "get_capital_city"
        and args.get("country", "").lower() == "restricted"
    ):
        print("[Callback] Bloqueando país restringido")
        return {"result": "El acceso a esta información ha sido restringido."}

    print("[Callback] Procediendo con la llamada normal a la herramienta")
    return None


# --- Definir Callback Después de la Herramienta ---
def after_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    """
    Callback simple que modifica la respuesta de la herramienta después de la ejecución.
    """
    tool_name = tool.name
    print(f"[Callback] Después de la llamada a la herramienta para '{tool_name}'")
    print(f"[Callback] Argumentos usados: {args}")
    print(f"[Callback] Respuesta original: {tool_response}")

    original_result = tool_response.get("result", "")
    print(f"[Callback] Resultado extraído: '{original_result}'")

    # Añadir una nota para cualquier respuesta de capital de EE. UU.
    if tool_name == "get_capital_city" and "washington" in original_result.lower():
        print("[Callback] CAPITAL DE EE. UU. DETECTADA - ¡añadiendo nota patriótica!")

        # Crear una copia modificada de la respuesta
        modified_response = copy.deepcopy(tool_response)
        modified_response["result"] = (
            f"{original_result} (Nota: Esta es la capital de los EE. UU. 🇺🇸)"
        )
        modified_response["note_added_by_callback"] = True

        print(f"[Callback] Respuesta modificada: {modified_response}")
        return modified_response

    print("[Callback] No se necesitan modificaciones, devolviendo la respuesta original")
    return None


# Crear el Agente
root_agent = LlmAgent(
    name="tool_callback_agent",
    model="gemini-2.0-flash",
    description="Un agente que demuestra callbacks de herramienta buscando capitales de ciudades",
    instruction="""
    Eres un útil asistente de geografía.
    
    Tu trabajo es:
    - Encontrar capitales de ciudades cuando se te pregunte usando la herramienta get_capital_city
    - Usar el nombre exacto del país proporcionado por el usuario
    - SIEMPRE devolver el resultado EXACTO de la herramienta, sin cambiarlo
    - Al informar una capital, mostrarla EXACTAMENTE como la devuelve la herramienta
    
    Ejemplos:
    - "¿Cuál es la capital de Francia?" → Usar get_capital_city con country="France"
    - "Dime la capital de Japón" → Usar get_capital_city con country="Japan"
    """,
    tools=[get_capital_city],
    before_tool_callback=[before_tool_callback],
    after_tool_callback=[after_tool_callback],
)
