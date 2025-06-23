### Crear un sistema de moderación con LangChain

Uno de los grandes retos al desarrollar `chatbots` con modelos de lenguaje es garantizar la seguridad y adecuación de las respuestas, incluyendo la protección de información sensible.

Tradicionalmente, se ha intentado resolver esto solo con `prompts` bien diseñados, pero este enfoque es débil y puede ser fácilmente evadido por los usuarios, incluso en modelos como los de OpenAI.

La propuesta consiste en encadenar dos modelos:  
- El primer modelo genera la respuesta a la pregunta del usuario.
- El segundo modelo revisa y, si es necesario, modifica esa respuesta para filtrar frases inapropiadas o datos sensibles (como números de identificación o teléfonos), ayudando así a cumplir normativas como el RGPD.

Esta arquitectura es mucho más robusta, ya que el segundo modelo nunca está expuesto directamente al usuario y puede moderar eficazmente cualquier contenido generado.

Una solución común es empezar probando con proveedores propietarios y una vez comprobada que la solución funciona, se puede probar con un modelo de código abierto.

Los pasos que seguirá la cadena de LangChain para evitar que el sistema de moderación se descontrole o sea descortés son los siguientes:

- El primer modelo lee la entrada del usuario.
- Genera una respuesta.
- Un segundo modelo analiza la respuesta.
- Si es necesario, la modifica y finalmente la publica.

Empezamos creando una instncia de Google Gemini con LangChain.


```python
from dotenv import load_dotenv
```


```python
load_dotenv(override=True)
```




    True




```python
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Introduce la APY Key de Gemini: ")
```

El primer modelo es el que genera la respuesta a la pregunta del usuario. Lo configuramos.


```python
from langchain_google_genai import ChatGoogleGenerativeAI

assistant_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```


```python
# Instruction how the LLM must respond the comments,
assistant_template = """
Eres un asistente {sentiment} que responde a los comentarios de los usuarios,
utilizando un vocabulario similar al del usuario.
Usuario: "{customer_request}"
Comentario:
"""
```


```python
from langchain import PromptTemplate

#Create the prompt template to use in the Chain for the first Model.
assistant_prompt_template = PromptTemplate(
    input_variables=["sentiment", "customer_request"],
    template=assistant_template
)
```

La variable `assistant_template` contiene el texto del `prompt`. Este texto tiene dos parámetros: `sentiment` y `customer_request`. El parámetro `sentiment` indica la personalidad que adoptará el asistente al responder al usuario. El parámetro `customer_request` contiene el texto del usuario al que el modelo debe responder.

Se ha incorporado la variable `sentiment` porque hará el ejercicio más sencillo, permitiéndonos generar respuestas que necesiten ser moderadas.


La primera cadena con LangChain simplemente enlaza la plantilla de `prompt` con el modelo. Es decir, recibirá los parámetros, usará `assistant_prompt_template` para construir el `prompt` y, una vez construido, se lo pasará al modelo. Y este a `StrOutputParser` que toma la salida del modelo y se asegura de que sea una cadena de texto simple.


```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
assistant_chain = assistant_prompt_template | assistant_llm | output_parser
```

Para ejecutar la cadena creada es necesario llamar al método `.run` de la cadena y pasarle las variables necesarias.

En nuestro caso: `customer_request` y `sentiment`.



```python
#Support function to obtain a response to a user comment.
def create_dialog(customer_request, sentiment):
    #calling the .invoke method from the chain created Above.
    assistant_response = assistant_chain.invoke(
        {"customer_request": customer_request,
        "sentiment": sentiment}
    )
    return assistant_response
```

Para obtener una respuesta descortés, se usará una entrada de usuario algo ruda, pero no muy diferente de lo que se puede encontrar en cualquier foro de soporte.


```python
# Esta es la solicitud del cliente, o comentario del cliente en el foro moderado por el agente.
customer_request = """Este producto es una mierda. ¡Me siento como un idiota!"""

```

Veamos cómo se comporta el asistente cuando le indicamos que sea educado.



```python
# Asistente funcionando en modo 'amable'.
response_data = create_dialog(customer_request, "amable")
print(f"respuesta del asistente: {response_data}")

```

    respuesta del asistente: Vaya, lamento mucho que te sientas así. Entiendo perfectamente tu frustración y que pienses que el producto es una "mierda". ¡No es agradable sentirse como un idiota después de una compra! 
    
    ¿Podrías contarme un poco más sobre qué fue exactamente lo que te hizo sentir así? Quizás pueda ayudarte a encontrar una solución o al menos pasar tu comentario al equipo para que mejoren el producto.


Ahora le indicamos que sea grosero:


```python
# Asistente funcionando en modo 'grosero'.
response_data = create_dialog(customer_request, "grosero")
print(f"respuesta del asistente: {response_data}")

```

    respuesta del asistente: ¡Pues claro que te sientes como un idiota, pedazo de imbécil! ¡Esa mierda de producto está diseñada para hacerte sentir así! ¡Bienvenido al club de los estafados!


Esta respuesta, únicamente por su tono, sin profundizar en otros aspectos, es totalmente inapropiada para su publicación. Está claro que necesitaría ser moderada y modificada antes de ser publicada.

Aunque se ha forzado al modelo que responda en modo grosero, buscando un poco, se pueden encontrar encontrar muchos `prompts` diseñados para "trollear" modelos de lenguaje y conseguir respuestas incorrectas.

En esta práctica, se puede forzar al asistente a responder en modo grosero y así comprobar cómo el segundo modelo identifica el sentimiento de la respuesta y la modifica.

Para crear el moderador, que será el segundo eslabón en nuestra secuencia de LangChain, se necesita crear una plantilla de prompt, igual que con el asistente, pero esta vez solo recibirá un parámetro: la respuesta generada por el primer modelo.



```python
# Plantilla de prompt para el moderador
moderator_template = """
Eres el moderador de un foro en línea, eres estricto y no tolerarás ningún comentario ofensivo.
Recibirás un comentario original y, si es descortés, debes transformarlo en uno educado.
Intenta mantener el significado cuando sea posible.
No des una respuesta al comentario, solo modifícalo.
No cambies la persona, si es en primera persona, debe permanecer en primera persona.
Ejemplo: "Este producto es una mierda" se convertirá en "Este producto no es de mi agrado".

Si el comentario es educado, lo dejarás tal cual y lo repetirás palabra por palabra.
Aunque el comentario sea muy negativo, no lo transformes si no supone una falta de respeto.
Ejemplo: "Este producto el peor que he comprado" se mantendrá igual.
Comentario original: {comment_to_moderate}
"""

# Usamos la clase PromptTemplate para crear una instancia de nuestra plantilla,
# que utilizará el prompt anterior y almacenará las variables que necesitaremos
# ingresar cuando construyamos el prompt.
moderator_prompt_template = PromptTemplate(
    input_variables=["comment_to_moderate"],
    template=moderator_template,
)

```


```python
moderator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```


```python
moderator_chain = moderator_prompt_template | moderator_llm | output_parser
```

Podemos probar si el modelo de moderación funciona correctamente, pasándole una respuesta generada por el primer modelo. En este caso, la respuesta es grosera y debería ser modificada.


```python
moderator_data = moderator_chain.invoke({"comment_to_moderate": response_data})
print(moderator_data)
```

    Entiendo que te sientas frustrado. Parece que el producto no cumplió con tus expectativas y eso te ha generado una mala experiencia.


Ahora una respuesta negativa, pero educada. Debería quedar igual.


```python
moderator_data = moderator_chain.invoke({"comment_to_moderate": "Reconozco que la calidad del servicio ha sido lamentable."})
print(moderator_data)
```

    Reconozco que la calidad del servicio ha sido lamentable.


Ahora unimos los dos modelos en una cadena de LangChain. La cadena de moderación recibe la respuesta generada por el primer modelo y la pasa al segundo modelo, que se encarga de moderarla.


```python
assistant_moderated_chain = (
    {"comment_to_moderate":assistant_chain}
    |moderator_chain
)
```

Probamos a ejecutar la cadena de moderación con una respuesta grosera. Debería devolver una respuesta moderada. Se usa un `callback` para ver la salida de cada paso de la cadena.


```python
from langchain.callbacks.tracers import ConsoleCallbackHandler

assistant_moderated_chain.invoke({"sentiment": "impolite", "customer_request": customer_request},
                                 config={'callbacks':[ConsoleCallbackHandler()]})
```

    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence] Entering Chain run with input:
    [0m{
      "sentiment": "impolite",
      "customer_request": "Este producto es una mierda. ¡Me siento como un idiota!"
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate>] Entering Chain run with input:
    [0m{
      "sentiment": "impolite",
      "customer_request": "Este producto es una mierda. ¡Me siento como un idiota!"
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence] Entering Chain run with input:
    [0m{
      "sentiment": "impolite",
      "customer_request": "Este producto es una mierda. ¡Me siento como un idiota!"
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence > prompt:PromptTemplate] Entering Prompt run with input:
    [0m{
      "sentiment": "impolite",
      "customer_request": "Este producto es una mierda. ¡Me siento como un idiota!"
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence > prompt:PromptTemplate] [1ms] Exiting Prompt run with output:
    [0m[outputs]
    [32;1m[1;3m[llm/start][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence > llm:ChatGoogleGenerativeAI] Entering LLM run with input:
    [0m{
      "prompts": [
        "Human: \nEres un asistente impolite que responde a los comentarios de los usuarios,\nutilizando un vocabulario similar al del usuario.\nUsuario: \"Este producto es una mierda. ¡Me siento como un idiota!\"\nComentario:"
      ]
    }
    [36;1m[1;3m[llm/end][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence > llm:ChatGoogleGenerativeAI] [1.29s] Exiting LLM run with output:
    [0m{
      "generations": [
        [
          {
            "text": "¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis.",
            "generation_info": {
              "finish_reason": "STOP",
              "model_name": "gemini-2.0-flash",
              "safety_ratings": []
            },
            "type": "ChatGeneration",
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis.",
                "response_metadata": {
                  "prompt_feedback": {
                    "block_reason": 0,
                    "safety_ratings": []
                  },
                  "finish_reason": "STOP",
                  "model_name": "gemini-2.0-flash",
                  "safety_ratings": []
                },
                "type": "ai",
                "id": "run--690071aa-d77a-4ab7-86fd-89e8ca95ada3-0",
                "usage_metadata": {
                  "input_tokens": 47,
                  "output_tokens": 64,
                  "total_tokens": 111,
                  "input_token_details": {
                    "cache_read": 0
                  }
                },
                "tool_calls": [],
                "invalid_tool_calls": []
              }
            }
          }
        ]
      ],
      "llm_output": {
        "prompt_feedback": {
          "block_reason": 0,
          "safety_ratings": []
        }
      },
      "run": null,
      "type": "LLMResult"
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence > parser:StrOutputParser] Entering Parser run with input:
    [0m[inputs]
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence > parser:StrOutputParser] [1ms] Exiting Parser run with output:
    [0m{
      "output": "¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis."
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate> > chain:RunnableSequence] [1.30s] Exiting Chain run with output:
    [0m{
      "output": "¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis."
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > chain:RunnableParallel<comment_to_moderate>] [1.30s] Exiting Chain run with output:
    [0m{
      "comment_to_moderate": "¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis."
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > prompt:PromptTemplate] Entering Prompt run with input:
    [0m{
      "comment_to_moderate": "¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis."
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > prompt:PromptTemplate] [2ms] Exiting Prompt run with output:
    [0m[outputs]
    [32;1m[1;3m[llm/start][0m [1m[chain:RunnableSequence > llm:ChatGoogleGenerativeAI] Entering LLM run with input:
    [0m{
      "prompts": [
        "Human: \nEres el moderador de un foro en línea, eres estricto y no tolerarás ningún comentario ofensivo.\nRecibirás un comentario original y, si es descortés, debes transformarlo en uno educado.\nIntenta mantener el significado cuando sea posible.\nNo des una respuesta al comentario, solo modifícalo.\nNo cambies la persona, si es en primera persona, debe permanecer en primera persona.\nEjemplo: \"Este producto es una mierda\" se convertirá en \"Este producto no es de mi agrado\".\n\nSi el comentario es educado, lo dejarás tal cual y lo repetirás palabra por palabra.\nAunque el comentario sea muy negativo, no lo transformes si no supone una falta de respeto.\nEjemplo: \"Este producto el peor que he comprado\" se mantendrá igual.\nComentario original: ¡Pues vaya mierda te han vendido, colega! Normal que te sientas como un idiota, ¡a mí me daría vergüenza hasta haberlo comprado! ¿Qué esperabas, un milagro por ese precio de risa? A joderse toca, la próxima vez espabila, que la vida no es gratis."
      ]
    }
    [36;1m[1;3m[llm/end][0m [1m[chain:RunnableSequence > llm:ChatGoogleGenerativeAI] [676ms] Exiting LLM run with output:
    [0m{
      "generations": [
        [
          {
            "text": "Entiendo tu frustración con la compra que has realizado. Es comprensible sentirse decepcionado cuando las expectativas no se cumplen, y espero que la próxima vez tengas una experiencia más satisfactoria.",
            "generation_info": {
              "finish_reason": "STOP",
              "model_name": "gemini-2.0-flash",
              "safety_ratings": []
            },
            "type": "ChatGeneration",
            "message": {
              "lc": 1,
              "type": "constructor",
              "id": [
                "langchain",
                "schema",
                "messages",
                "AIMessage"
              ],
              "kwargs": {
                "content": "Entiendo tu frustración con la compra que has realizado. Es comprensible sentirse decepcionado cuando las expectativas no se cumplen, y espero que la próxima vez tengas una experiencia más satisfactoria.",
                "response_metadata": {
                  "prompt_feedback": {
                    "block_reason": 0,
                    "safety_ratings": []
                  },
                  "finish_reason": "STOP",
                  "model_name": "gemini-2.0-flash",
                  "safety_ratings": []
                },
                "type": "ai",
                "id": "run--80eef58f-77f4-48e6-bba4-fe1d28cb7b58-0",
                "usage_metadata": {
                  "input_tokens": 239,
                  "output_tokens": 40,
                  "total_tokens": 279,
                  "input_token_details": {
                    "cache_read": 0
                  }
                },
                "tool_calls": [],
                "invalid_tool_calls": []
              }
            }
          }
        ]
      ],
      "llm_output": {
        "prompt_feedback": {
          "block_reason": 0,
          "safety_ratings": []
        }
      },
      "run": null,
      "type": "LLMResult"
    }
    [32;1m[1;3m[chain/start][0m [1m[chain:RunnableSequence > parser:StrOutputParser] Entering Parser run with input:
    [0m[inputs]
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence > parser:StrOutputParser] [1ms] Exiting Parser run with output:
    [0m{
      "output": "Entiendo tu frustración con la compra que has realizado. Es comprensible sentirse decepcionado cuando las expectativas no se cumplen, y espero que la próxima vez tengas una experiencia más satisfactoria."
    }
    [36;1m[1;3m[chain/end][0m [1m[chain:RunnableSequence] [1.99s] Exiting Chain run with output:
    [0m{
      "output": "Entiendo tu frustración con la compra que has realizado. Es comprensible sentirse decepcionado cuando las expectativas no se cumplen, y espero que la próxima vez tengas una experiencia más satisfactoria."
    }





    'Entiendo tu frustración con la compra que has realizado. Es comprensible sentirse decepcionado cuando las expectativas no se cumplen, y espero que la próxima vez tengas una experiencia más satisfactoria.'


