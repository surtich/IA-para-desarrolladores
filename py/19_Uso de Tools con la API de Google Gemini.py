# %% [markdown]
# ### Uso de Tools en Gemini
# 
# En esta práctica vamos a aprender a usar directamente la API de Gemini.

# %%
# Importar las librerías necesarias
import os
from typing import Any, Dict

import google.generativeai as genai
import requests
from dotenv import load_dotenv

print("¡Estamos en marcha!")

# %%
# Cargar las variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key.startswith("ADD YOUR"):
    raise ValueError("GOOGLE_API_KEY no encontrada en el archivo .env")

# Configurar Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

print(f"Modelo Gemini cargado correctamente: {model.model_name}")

# %% [markdown]
# Comprobamos que no podemos conocer el precio de cotización actual de Bitcoin.

# %%
# Definir el mensaje de consulta
PROMPT = "¿Cuál es el precio actual de Bitcoin?"
chat = model.start_chat()
response = chat.send_message(PROMPT)
print(response.text)

# %% [markdown]
# Podemos ver la estructura de la respuesta que nos devuelve Gemini.

# %%
print(response)

# %% [markdown]
# Usamos una API para obtener el precio de cotización actual de Bitcoin.

# %%
url = f"https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
response = requests.get(url)
data = response.json()
print(data)

# %% [markdown]
# Creamos una función que nos devuelve el precio de cotización actual de Bitcoin.

# %%
# Definir la función
def get_crypto_price(symbol: str) -> float:
    """
    Obtener el precio actual de una criptomoneda desde la API de Binance
    """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return float(data["price"])

# %%
precio = get_crypto_price("BTCUSDT")
print(f"Precio de BTC en USDT: {precio}")

# %% [markdown]
# Creamos una `tool` que llame a la función anterior. Los comentarios son importantes ya que no podemos suponer que Gemini conoce la API de Binance.

# %%
tools = [
    {
        "function_declarations": [
            {
                "name": "get_crypto_price",
                "description": "Obtener el precio de una criptomoneda en USDT desde Binance",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "El símbolo del par de trading de la criptomoneda (por ejemplo, BTCUSDT, ETHUSDT). \
                                            El símbolo para Bitcoin es BTCUSDT. \
                                            El símbolo para Ethereum es ETHUSDT."
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]
    }
]

# %% [markdown]
# Hacemos la misma pregunta pero pasando a Gemini la tool que acabamos de crear. Observamos que ahora Gemini no responde con un campo `text` sino con un campo `parts`. Con esto Gemini nos está indicando que quiere usar la `tool` para obtener el precio de cotización actual de Bitcoin. Los LLMs nunca llaman directamente a las `tools`, sino que nos piden que las llamemos nosotros.

# %%
PROMPT = "¿Cuál es el precio actual de Bitcoin (BTCUSDT)?"
chat = model.start_chat()
response = chat.send_message(PROMPT, tools=tools)
print(response)

# %% [markdown]
# Hacemos lo que nos pide Gemini.

# %%
price = get_crypto_price("BTCUSDT")

# %% [markdown]
# Y le enviamos la respuesta de la `tool` a Gemini.

# %%
final_response = chat.send_message(str(price))
print(final_response)

# %%
print(final_response.text)


