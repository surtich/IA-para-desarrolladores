from datetime import datetime

import yfinance as yf
from google.adk.agents import Agent


def get_stock_price(ticker: str) -> dict:
    """Recupera el precio actual de la acción y lo guarda en el estado de la sesión."""
    print(f"--- Herramienta: get_stock_price llamada para {ticker} ---")

    try:
        # Obtener datos de la acción
        stock = yf.Ticker(ticker)
        current_price = stock.info.get("currentPrice")

        if current_price is None:
            return {
                "status": "error",
                "error_message": f"No se pudo obtener el precio para {ticker}",
            }

        # Obtener la marca de tiempo actual
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success",
            "ticker": ticker,
            "price": current_price,
            "timestamp": current_time,
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error al obtener datos de la acción: {str(e)}",
        }


# Crear el agente raíz
stock_analyst = Agent(
    name="stock_analyst",
    model="gemini-2.0-flash",
    description="Un agente que puede buscar precios de acciones y seguirlos a lo largo del tiempo.",
    instruction="""
    Eres un útil asistente del mercado de valores que ayuda a los usuarios a seguir sus acciones de interés.
    
    Cuando se te pregunte sobre precios de acciones:
    1. Usa la herramienta get_stock_price para obtener el último precio de las acciones solicitadas
    2. Formatea la respuesta para mostrar el precio actual de cada acción y la hora en que se obtuvo
    3. Si no se pudo obtener el precio de una acción, menciónalo en tu respuesta
    
    Formato de respuesta de ejemplo:
    "Aquí están los precios actuales de tus acciones:
    - GOOG: $175.34 (actualizado a las 2024-04-21 16:30:00)
    - TSLA: $156.78 (actualizado a las 2024-04-21 16:30:00)
    - META: $123.45 (actualizado a las 2024-04-21 16:30:00)"
    """,
    tools=[get_stock_price],
)
