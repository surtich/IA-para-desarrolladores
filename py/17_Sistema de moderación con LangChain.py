# %% [markdown]
# ### Crear un sistema de moderación con LangChain
# 
# Uno de los grandes retos al desarrollar `chatbots` con modelos de lenguaje es garantizar la seguridad y adecuación de las respuestas, incluyendo la protección de información sensible.
# 
# Tradicionalmente, se ha intentado resolver esto solo con `prompts` bien diseñados, pero este enfoque es débil y puede ser fácilmente evadido por los usuarios, incluso en modelos como los de OpenAI.
# 
# La propuesta consiste en encadenar dos modelos:  
# - El primer modelo genera la respuesta a la pregunta del usuario.
# - El segundo modelo revisa y, si es necesario, modifica esa respuesta para filtrar frases inapropiadas o datos sensibles (como números de identificación o teléfonos), ayudando así a cumplir normativas como el RGPD.
# 
# Esta arquitectura es mucho más robusta, ya que el segundo modelo nunca está expuesto directamente al usuario y puede moderar eficazmente cualquier contenido generado.
# 
# Una solución común es empezar probando con proveedores propietarios y una vez comprobada que la solución funciona, se puede probar con un modelo de código abierto.

# %% [markdown]
# Los pasos que seguirá la cadena de LangChain para evitar que el sistema de moderación se descontrole o sea descortés son los siguientes:
# 
# - El primer modelo lee la entrada del usuario.
# - Genera una respuesta.
# - Un segundo modelo analiza la respuesta.
# - Si es necesario, la modifica y finalmente la publica.

# %% [markdown]
# Empezamos creando una instncia de Google Gemini con LangChain.

# %%
from dotenv import load_dotenv

# %%
load_dotenv(override=True)

# %%
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Introduce la APY Key de Gemini: ")

# %% [markdown]
# El primer modelo es el que genera la respuesta a la pregunta del usuario. Lo configuramos.

# %%
from langchain_google_genai import ChatGoogleGenerativeAI

assistant_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# %%
# Instruction how the LLM must respond the comments,
assistant_template = """
Eres un asistente {sentiment} que responde a los comentarios de los usuarios,
utilizando un vocabulario similar al del usuario.
Usuario: "{customer_request}"
Comentario:
"""

# %%
from langchain import PromptTemplate

#Create the prompt template to use in the Chain for the first Model.
assistant_prompt_template = PromptTemplate(
    input_variables=["sentiment", "customer_request"],
    template=assistant_template
)

# %% [markdown]
# La variable `assistant_template` contiene el texto del `prompt`. Este texto tiene dos parámetros: `sentiment` y `customer_request`. El parámetro `sentiment` indica la personalidad que adoptará el asistente al responder al usuario. El parámetro `customer_request` contiene el texto del usuario al que el modelo debe responder.
# 
# Se ha incorporado la variable `sentiment` porque hará el ejercicio más sencillo, permitiéndonos generar respuestas que necesiten ser moderadas.
# 

# %% [markdown]
# La primera cadena con LangChain simplemente enlaza la plantilla de `prompt` con el modelo. Es decir, recibirá los parámetros, usará `assistant_prompt_template` para construir el `prompt` y, una vez construido, se lo pasará al modelo. Y este a `StrOutputParser` que toma la salida del modelo y se asegura de que sea una cadena de texto simple.

# %%
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
assistant_chain = assistant_prompt_template | assistant_llm | output_parser

# %% [markdown]
# Para ejecutar la cadena creada es necesario llamar al método `.run` de la cadena y pasarle las variables necesarias.
# 
# En nuestro caso: `customer_request` y `sentiment`.
# 

# %%
#Support function to obtain a response to a user comment.
def create_dialog(customer_request, sentiment):
    #calling the .invoke method from the chain created Above.
    assistant_response = assistant_chain.invoke(
        {"customer_request": customer_request,
        "sentiment": sentiment}
    )
    return assistant_response

# %% [markdown]
# Para obtener una respuesta descortés, se usará una entrada de usuario algo ruda, pero no muy diferente de lo que se puede encontrar en cualquier foro de soporte.

# %%
# Esta es la solicitud del cliente, o comentario del cliente en el foro moderado por el agente.
customer_request = """Este producto es una mierda. ¡Me siento como un idiota!"""


# %% [markdown]
# Veamos cómo se comporta el asistente cuando le indicamos que sea educado.
# 

# %%
# Asistente funcionando en modo 'amable'.
response_data = create_dialog(customer_request, "amable")
print(f"respuesta del asistente: {response_data}")


# %% [markdown]
# Ahora le indicamos que sea grosero:

# %%
# Asistente funcionando en modo 'grosero'.
response_data = create_dialog(customer_request, "grosero")
print(f"respuesta del asistente: {response_data}")


# %% [markdown]
# Esta respuesta, únicamente por su tono, sin profundizar en otros aspectos, es totalmente inapropiada para su publicación. Está claro que necesitaría ser moderada y modificada antes de ser publicada.
# 
# Aunque se ha forzado al modelo que responda en modo grosero, buscando un poco, se pueden encontrar encontrar muchos `prompts` diseñados para "trollear" modelos de lenguaje y conseguir respuestas incorrectas.
# 
# En esta práctica, se puede forzar al asistente a responder en modo grosero y así comprobar cómo el segundo modelo identifica el sentimiento de la respuesta y la modifica.
# 
# Para crear el moderador, que será el segundo eslabón en nuestra secuencia de LangChain, se necesita crear una plantilla de prompt, igual que con el asistente, pero esta vez solo recibirá un parámetro: la respuesta generada por el primer modelo.
# 

# %%
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


# %%
moderator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# %%
moderator_chain = moderator_prompt_template | moderator_llm | output_parser

# %% [markdown]
# Podemos probar si el modelo de moderación funciona correctamente, pasándole una respuesta generada por el primer modelo. En este caso, la respuesta es grosera y debería ser modificada.

# %%
moderator_data = moderator_chain.invoke({"comment_to_moderate": response_data})
print(moderator_data)

# %% [markdown]
# Ahora una respuesta negativa, pero educada. Debería quedar igual.

# %%
moderator_data = moderator_chain.invoke({"comment_to_moderate": "Reconozco que la calidad del servicio ha sido lamentable."})
print(moderator_data)

# %% [markdown]
# Ahora unimos los dos modelos en una cadena de LangChain. La cadena de moderación recibe la respuesta generada por el primer modelo y la pasa al segundo modelo, que se encarga de moderarla.

# %%
assistant_moderated_chain = (
    {"comment_to_moderate":assistant_chain}
    |moderator_chain
)

# %% [markdown]
# Probamos a ejecutar la cadena de moderación con una respuesta grosera. Debería devolver una respuesta moderada. Se usa un `callback` para ver la salida de cada paso de la cadena.

# %%
from langchain.callbacks.tracers import ConsoleCallbackHandler

assistant_moderated_chain.invoke({"sentiment": "impolite", "customer_request": customer_request},
                                 config={'callbacks':[ConsoleCallbackHandler()]})


