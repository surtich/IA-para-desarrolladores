### Uso de Tools en Gemini

En esta práctica vamos a aprender a usar directamente la API de Gemini.


```python
# Importar las librerías necesarias
import os
from typing import Any, Dict

import google.generativeai as genai
import requests
from dotenv import load_dotenv

print("¡Estamos en marcha!")
```

    ¡Estamos en marcha!


    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
# Cargar las variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key.startswith("ADD YOUR"):
    raise ValueError("GOOGLE_API_KEY no encontrada en el archivo .env")

# Configurar Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

print(f"Modelo Gemini cargado correctamente: {model.model_name}")
```

    Modelo Gemini cargado correctamente: models/gemini-2.0-flash


Comprobamos que no podemos conocer el precio de cotización actual de Bitcoin.


```python
# Definir el mensaje de consulta
PROMPT = "¿Cuál es el precio actual de Bitcoin?"
chat = model.start_chat()
response = chat.send_message(PROMPT)
print(response.text)
```

    Lo siento, no tengo el precio actual de Bitcoin.


Podemos ver la estructura de la respuesta que nos devuelve Gemini.


```python
print(response)
```

    response:
    GenerateContentResponse(
        done=True,
        iterator=None,
        result=protos.GenerateContentResponse({
          "candidates": [
            {
              "content": {
                "parts": [
                  {
                    "text": "Lo siento, no tengo el precio actual de Bitcoin."
                  }
                ],
                "role": "model"
              },
              "finish_reason": "STOP",
              "avg_logprobs": -0.4191397753628818
            }
          ],
          "usage_metadata": {
            "prompt_token_count": 9,
            "candidates_token_count": 11,
            "total_token_count": 20
          },
          "model_version": "gemini-2.0-flash"
        }),
    )


Usamos una API para obtener el precio de cotización actual de Bitcoin.


```python
url = f"https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
response = requests.get(url)
data = response.json()
print(data)
```

    {'symbol': 'BTCUSDT', 'price': '106076.29000000'}


Creamos una función que nos devuelve el precio de cotización actual de Bitcoin.


```python
# Definir la función
def get_crypto_price(symbol: str) -> float:
    """
    Obtener el precio actual de una criptomoneda desde la API de Binance
    """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    return float(data["price"])
```


```python
precio = get_crypto_price("BTCUSDT")
print(f"Precio de BTC en USDT: {precio}")
```

    Precio de BTC en USDT: 106076.29


Creamos una `tool` que llame a la función anterior. Los comentarios son importantes ya que no podemos suponer que Gemini conoce la API de Binance.


```python
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
```

Hacemos la misma pregunta pero pasando a Gemini la tool que acabamos de crear. Observamos que ahora Gemini no responde con un campo `text` sino con un campo `parts`. Con esto Gemini nos está indicando que quiere usar la `tool` para obtener el precio de cotización actual de Bitcoin. Los LLMs nunca llaman directamente a las `tools`, sino que nos piden que las llamemos nosotros.


```python
PROMPT = "¿Cuál es el precio actual de Bitcoin (BTCUSDT)?"
chat = model.start_chat()
response = chat.send_message(PROMPT, tools=tools)
print(response)
```

    response:
    GenerateContentResponse(
        done=True,
        iterator=None,
        result=protos.GenerateContentResponse({
          "candidates": [
            {
              "content": {
                "parts": [
                  {
                    "function_call": {
                      "name": "get_crypto_price",
                      "args": {
                        "symbol": "BTCUSDT"
                      }
                    }
                  }
                ],
                "role": "model"
              },
              "finish_reason": "STOP",
              "avg_logprobs": -6.876605766592547e-06
            }
          ],
          "usage_metadata": {
            "prompt_token_count": 72,
            "candidates_token_count": 8,
            "total_token_count": 80
          },
          "model_version": "gemini-2.0-flash"
        }),
    )


Hacemos lo que nos pide Gemini.


```python
price = get_crypto_price("BTCUSDT")
```

Y le enviamos la respuesta de la `tool` a Gemini.


```python
final_response = chat.send_message(str(price))
print(final_response)
```

    response:
    GenerateContentResponse(
        done=True,
        iterator=None,
        result=protos.GenerateContentResponse({
          "candidates": [
            {
              "content": {
                "parts": [
                  {
                    "text": "El precio actual de Bitcoin (BTCUSDT) es 106015.57.\n"
                  }
                ],
                "role": "model"
              },
              "finish_reason": "STOP",
              "avg_logprobs": -0.0021627023816108704
            }
          ],
          "usage_metadata": {
            "prompt_token_count": 29,
            "candidates_token_count": 22,
            "total_token_count": 51
          },
          "model_version": "gemini-2.0-flash"
        }),
    )



```python
print(final_response.text)
```

    El precio actual de Bitcoin (BTCUSDT) es 106015.57.
    

