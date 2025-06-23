### ¿Qué es LangChain?

**LangChain** es un `framework` de código abierto diseñado para facilitar el desarrollo de aplicaciones que utilizan modelos de lenguaje. Permite unificar la interfaz de acceso a distintos proveedores.

### Características principales LangChain

- **Orquestación modular:** LangChain proporciona componentes modulares que se pueden combinar para crear flujos de trabajo complejos, llamados "cadenas" (pipelines). Cada cadena es una secuencia de pasos que pueden incluir llamadas a modelos de lenguaje, consultas a bases de datos, procesamiento de texto.
- **Integración sencilla:** Permite integrar casi cualquier modelo de lenguaje, tanto de código abierto como comercial, usando una interfaz estándar y sencilla.
- **Gestión de contexto y memoria:** Facilita la gestión del estado de la conversación y el contexto, permitiendo que las aplicaciones recuerden interacciones anteriores y ofrezcan respuestas más coherentes y personalizadas.
- **Automatización y agentes:** Permite crear agentes inteligentes que pueden tomar decisiones, consultar diferentes fuentes de datos y ejecutar acciones de forma autónoma.
- **Soporte para Python y JavaScript:** Está disponible principalmente para estos lenguajes, facilitando su adopción en proyectos modernos.



### Input y Output

En el siguiente ejemplo, comparamos el uso de un LLM con la API de OpenAI con la de LangChain.


```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
```




    True




```python
from openai import OpenAI

openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=os.getenv('GOOGLE_API_KEY'))

response = openai.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "Eres un asistente útil especializado en proporcionar información sobre el Restaurante Italiano BellaVista."},
        {"role": "user", "content": "¿Qué hay en el menú?"},
        {"role": "assistant", "content": "BellaVista ofrece una variedad de platos italianos que incluyen pasta, pizza y mariscos."},
        {"role": "user", "content": "¿Tienen opciones veganas?"}
    ]
)

response.model_dump()
```




    {'id': '8dxaaJbCI-PQx_APtpyn2A8',
     'choices': [{'finish_reason': 'stop',
       'index': 0,
       'logprobs': None,
       'message': {'content': 'Sí, tenemos algunas opciones veganas, como la pasta primavera y la pizza de verduras.',
        'refusal': None,
        'role': 'assistant',
        'annotations': None,
        'audio': None,
        'function_call': None,
        'tool_calls': None}}],
     'created': 1750785265,
     'model': 'gemini-2.0-flash',
     'object': 'chat.completion',
     'service_tier': None,
     'system_fingerprint': None,
     'usage': {'completion_tokens': 18,
      'prompt_tokens': 44,
      'total_tokens': 62,
      'completion_tokens_details': None,
      'prompt_tokens_details': None}}




```python
print(response.choices[0].message.content)
```

    Sí, tenemos algunas opciones veganas, como la pasta primavera y la pizza de verduras.



```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv('GOOGLE_API_KEY'))

llm.invoke([
        ("system",  "Eres un asistente útil especializado en proporcionar información sobre el Restaurante Italiano BellaVista."),
        ("human",  "¿Qué hay en el menú?"),
        ("ai",  "BellaVista ofrece una variedad de platos italianos que incluyen pasta, pizza y mariscos."),
        ("human",  "¿Tienen opciones veganas?")
    ])
```




    AIMessage(content='Sí, tenemos algunas opciones veganas, como pasta con verduras de temporada y pizza vegana.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--412c45d6-93b0-476d-9a67-9fc461b5ef94-0', usage_metadata={'input_tokens': 44, 'output_tokens': 19, 'total_tokens': 63, 'input_token_details': {'cache_read': 0}})



Un forma alternativa de enviar mensajes.


```python
from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="Eres un asistente útil especializado en proporcionar información sobre el Restaurante Italiano BellaVista."),
    HumanMessage(content="¿Cuál es el menú?"),
    AIMessage(content="BellaVista ofrece una variedad de platos italianos que incluyen pasta, pizza y mariscos."),
    HumanMessage(content="¿Tienen opciones veganas?")
]

llm_result = llm.invoke(input=messages)
llm_result
```




    AIMessage(content='Sí, BellaVista ofrece varias opciones veganas.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--03342b5b-0e85-43e5-a040-107af1a95acf-0', usage_metadata={'input_tokens': 43, 'output_tokens': 10, 'total_tokens': 53, 'input_token_details': {'cache_read': 0}})



Los mensajes se pueden eviar en `batch`


```python
batch_messages = [
    [
        SystemMessage(content="Eres un asistente útil que traduce del inglés al alemán"),
        HumanMessage(content="Do you have vegan options?")
    ],
    [
        SystemMessage(content="Eres un asistente útil que traduce del español al inglés."),
        HumanMessage(content="¿Tienen opciones veganas?")
    ],
]
batch_result = llm.generate(batch_messages)
batch_result

```




    LLMResult(generations=[[ChatGeneration(text='Haben Sie vegane Optionen?', generation_info={'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, message=AIMessage(content='Haben Sie vegane Optionen?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--4baa2ba2-3827-4e82-a089-523fea0f7fda-0', usage_metadata={'input_tokens': 16, 'output_tokens': 7, 'total_tokens': 23, 'input_token_details': {'cache_read': 0}}))], [ChatGeneration(text='Do you have vegan options?', generation_info={'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, message=AIMessage(content='Do you have vegan options?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--80f87996-e8bc-4726-b950-a29655e846c3-0', usage_metadata={'input_tokens': 17, 'output_tokens': 7, 'total_tokens': 24, 'input_token_details': {'cache_read': 0}}))]], llm_output={}, run=[RunInfo(run_id=UUID('4baa2ba2-3827-4e82-a089-523fea0f7fda')), RunInfo(run_id=UUID('80f87996-e8bc-4726-b950-a29655e846c3'))], type='LLMResult')




```python
translations = [generation[0].text for generation in batch_result.generations]
translations
```




    ['Haben Sie vegane Optionen?', 'Do you have vegan options?']



### Prompt templates

LangChain permite crear plantillas de prompts que pueden ser reutilizadas y parametrizadas. Esto facilita la creación de mensajes complejos y dinámicos para los modelos de lenguaje.


```python
from langchain.prompts.prompt import PromptTemplate

TEMPLATE = """
Eres un asistente útil que traduce del {input_language} al {output_language}
"""

prompt_template = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template="Eres un asistente útil que traduce del {input_language} al {output_language}. Traduce: {text}"
)

prompt = prompt_template.format(
    input_language="español",
    output_language="inglés",
    text="¿A qué te dedicas?"
)

llm.invoke(prompt)
```




    AIMessage(content='What do you do for a living?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--636b20da-f59b-4289-b041-d12ccb8ef1f7-0', usage_metadata={'input_tokens': 21, 'output_tokens': 9, 'total_tokens': 30, 'input_token_details': {'cache_read': 0}})



LangChain facilita la creación de prompts con ejemplo (Few Shot Prompt)


```python
from langchain_core.prompts import FewShotPromptTemplate

# Ejemplos de clasificación
examples = [
    {
        "text": "El restaurante BellaVista ofrece una experiencia culinaria exquisita. Los sabores son ricos y la presentación es impecable.",
        "sentiment": "positive",
        "subject": "BellaVista"
    },
    {
        "text": "El restaurante BellaVista estuvo bien. La comida era decente, pero nada destacaba.",
        "sentiment": "neutral",
        "subject": "BellaVista"
    },
    {
        "text": "Me decepcionó BellaVista. El servicio fue lento y los platos carecían de sabor.",
        "sentiment": "negative",
        "subject": "BellaVista"
    },
    {
        "text": "SeoulSavor ofreció los sabores coreanos más auténticos que he probado fuera de Seúl. El kimchi estaba perfectamente fermentado y picante.",
        "sentiment": "positive",
        "subject": "SeoulSavor"
    },
    {
        "text": "SeoulSavor estuvo bien. El bibimbap era bueno, pero el bulgogi era un poco demasiado dulce para mi gusto.",
        "sentiment": "neutral",
        "subject": "SeoulSavor"
    },
    {
        "text": "No disfruté mi comida en SeoulSavor. El tteokbokki estaba demasiado blando y el servicio no fue atento.",
        "sentiment": "negative",
        "subject": "SeoulSavor"
    },
    {
        "text": "MunichMeals tiene la mejor bratwurst y sauerkraut que he probado fuera de Baviera. Su ambiente de jardín de cerveza es verdaderamente auténtico.",
        "sentiment": "positive",
        "subject": "MunichMeals"
    },
    {
        "text": "MunichMeals estuvo bien. La weisswurst estaba bien, pero he probado mejores en otros lugares.",
        "sentiment": "neutral",
        "subject": "MunichMeals"
    },
    {
        "text": "Me decepcionó MunichMeals. La ensalada de patatas carecía de sabor y el personal parecía desinteresado.",
        "sentiment": "negative",
        "subject": "MunichMeals"
    }
]

# Plantilla para cada ejemplo
example_prompt = PromptTemplate(
    input_variables=["text", "sentiment", "subject"],
    template=(
        "text: {text}\n"
        "sentiment: {sentiment}\n"
        "subject: {subject}\n"
    )
)

# Plantilla FewShot con sufijo para el nuevo caso a clasificar
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="text: {text}\nsentiment:",
    input_variables=["text"]
)

# Opinión a clasificar
nueva_opinion = "El ambiente de BellaVista era agradable, pero la comida llegó fría y tardó mucho."

# Formatear el prompt final
prompt_final = few_shot_prompt.format(text=nueva_opinion)

llm.invoke(prompt_final)

```




    AIMessage(content='negative\nsubject: BellaVista', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--7eeec0cf-2c64-4034-a759-ba4614bdee8d-0', usage_metadata={'input_tokens': 346, 'output_tokens': 7, 'total_tokens': 353, 'input_token_details': {'cache_read': 0}})



Los prompts se puede componer para facilitar la reutilización.


```python
from langchain.prompts.pipeline import PipelinePromptTemplate

# Introducción
introduction_template = """
Interpreta el texto y evalúalo. Determina si el texto tiene un sentimiento positivo, neutral o negativo. Además, identifica el tema del texto en una palabra.
"""
introduction_prompt = PromptTemplate.from_template(introduction_template)

# Ejemplo
example_template = """
Instrucciones de Cadena de Pensamiento:
Comencemos evaluando una declaración. Considera: "{example_text}". ¿Cómo te hace sentir esto sobre {example_subject}?
Respuesta: {example_evaluation}

Basado en la naturaleza {example_sentiment} de esa declaración, ¿cómo formatearías tu respuesta?
Respuesta: {example_format}
"""
example_prompt = PromptTemplate.from_template(example_template)

# Ejecución
execution_template = """
Ahora, ejecuta este proceso para el texto: "{input}".
"""
execution_prompt = PromptTemplate.from_template(execution_template)

# Componiendo el prompt completo
full_template = """{introduction}

{example}

{execution}"""
full_prompt = PromptTemplate.from_template(full_template)

# Prompts de Pipeline
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("execution", execution_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

prompt = pipeline_prompt.format(
    example_text="El restaurante BellaVista ofrece una experiencia culinaria exquisita. Los sabores son ricos y la presentación es impecable.",
    example_subject="BellaVista",
    example_evaluation="Suena como una crítica positiva para BellaVista.",
    example_sentiment="positive",
    example_format='{ "sentiment": "positive", "subject": "BellaVista" }',
    input="El nuevo restaurante del centro tiene platos insípidos y el tiempo de espera es demasiado largo."
)

print(prompt)

llm.invoke(prompt)  

```

    
    Interpreta el texto y evalúalo. Determina si el texto tiene un sentimiento positivo, neutral o negativo. Además, identifica el tema del texto en una palabra.
    
    
    
    Instrucciones de Cadena de Pensamiento:
    Comencemos evaluando una declaración. Considera: "El restaurante BellaVista ofrece una experiencia culinaria exquisita. Los sabores son ricos y la presentación es impecable.". ¿Cómo te hace sentir esto sobre BellaVista?
    Respuesta: Suena como una crítica positiva para BellaVista.
    
    Basado en la naturaleza positive de esa declaración, ¿cómo formatearías tu respuesta?
    Respuesta: { "sentiment": "positive", "subject": "BellaVista" }
    
    
    
    Ahora, ejecuta este proceso para el texto: "El nuevo restaurante del centro tiene platos insípidos y el tiempo de espera es demasiado largo.".
    





    AIMessage(content='{ "sentiment": "negative", "subject": "restaurant" }', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--925951d8-12fd-4b78-a9b6-7022a6f68a08-0', usage_metadata={'input_tokens': 169, 'output_tokens': 15, 'total_tokens': 184, 'input_token_details': {'cache_read': 0}})



Los prompts se pueden almacenar en disco duro y recuperar.


```python
prompt_template = PromptTemplate(input_variables=["input"], template="Cuéntame un chiste sobre {input}")
prompt_template.save("prompt.yaml")
prompt_template.save("prompt.json")
```


```python
from langchain.prompts import load_prompt

prompt_template = load_prompt("prompt.yaml")
prompt = prompt_template.format(input="gatos")

llm.invoke(prompt)
```




    AIMessage(content='Why did the cat join the Red Cross?\n\nBecause he wanted to be a first-aid kit! (First-Aid Cat!)', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--e756367e-0670-43e7-ab4c-4482339e008b-0', usage_metadata={'input_tokens': 6, 'output_tokens': 28, 'total_tokens': 34, 'input_token_details': {'cache_read': 0}})



### Chains

LangChain permite crear "cadenas" (chains) que son secuencias de pasos que pueden incluir llamadas a modelos de lenguaje. Las cadenas pueden ser simples o complejas, y permiten orquestar el flujo de trabajo de la aplicación. La forma más sencilla de crear una cade es usar LCEL (LangChain Expression Language), que permite definir cadenas de forma declarativa.


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("cuéntame un chiste corto sobre {topic}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({"topic": "helado"})
```




    '¿Qué le dijo un helado a otro helado?\n\n¡Me derrito por ti!'




```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

chain = RunnableParallel({"x": RunnablePassthrough(), "y": RunnablePassthrough()})
chain.invoke({"input": "hello", "input2": "goodbye"})

```




    {'x': {'input': 'hello', 'input2': 'goodbye'},
     'y': {'input': 'hello', 'input2': 'goodbye'}}



### Callbacks

LangChain permite registrar callbacks para monitorear y depurar el flujo de trabajo de las cadenas. Los callbacks pueden ser utilizados para registrar información, manejar errores o realizar acciones específicas en diferentes etapas del proceso.


```python
from langchain.callbacks import StdOutCallbackHandler

prompt_template = PromptTemplate(input_variables=["input"], template="Cuéntame un chiste sobre {input}")
chain = prompt | llm

handler = StdOutCallbackHandler()

config = {
    'callbacks' : [handler]
}

chain.invoke(input="león", config=config)
```

    
    
    [1m> Entering new RunnableSequence chain...[0m
    
    
    [1m> Entering new ChatPromptTemplate chain...[0m
    
    [1m> Finished chain.[0m
    
    [1m> Finished chain.[0m





    AIMessage(content='Claro, aquí tienes un chiste corto sobre leones:\n\n¿Por qué los leones comen carne cruda? \n\n¡Porque no saben cocinar!', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--18ec4fec-f5eb-4352-9ce3-8c744b2da265-0', usage_metadata={'input_tokens': 9, 'output_tokens': 32, 'total_tokens': 41, 'input_token_details': {'cache_read': 0}})



Se puede personalizar la función del `callback`.


```python
from langchain.callbacks.base import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        print(f"REPONSE: ", response)

handler = MyCustomHandler()

config = {
    'callbacks' : [handler]
}

chain.invoke(input="pingüinos", config=config)
```

    REPONSE:  generations=[[ChatGeneration(text='Claro, aquí tienes un chiste corto sobre pingüinos:\n\n¿Qué hace un pingüino con un taladro?\n\n¡Pingüin-taladra! ', generation_info={'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, message=AIMessage(content='Claro, aquí tienes un chiste corto sobre pingüinos:\n\n¿Qué hace un pingüino con un taladro?\n\n¡Pingüin-taladra! ', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--fb17ef33-1ae3-4251-81b4-f89c332e7060-0', usage_metadata={'input_tokens': 11, 'output_tokens': 37, 'total_tokens': 48, 'input_token_details': {'cache_read': 0}}))]] llm_output={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}} run=None type='LLMResult'





    AIMessage(content='Claro, aquí tienes un chiste corto sobre pingüinos:\n\n¿Qué hace un pingüino con un taladro?\n\n¡Pingüin-taladra! ', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--fb17ef33-1ae3-4251-81b4-f89c332e7060-0', usage_metadata={'input_tokens': 11, 'output_tokens': 37, 'total_tokens': 48, 'input_token_details': {'cache_read': 0}})



### Memoria

LangChain proporciona mecanismos para gestionar el estado de la conversación y el contexto, permitiendo que las aplicaciones recuerden interacciones anteriores.

Algunos de los tipos de memoria disponibles en LangChain son los siguientes:

| Tipo de memoria                    | Descripción breve                                                                                         | Uso típico                                                  |
|------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| ConversationBufferMemory           | Guarda toda la conversación en un buffer (lista o cadena).                                               | Chatbots, asistentes, historias interactivas                |
| BufferWindowMemory                 | Similar al buffer, pero solo mantiene las últimas k interacciones.                                       | Limitar contexto a lo más reciente                          |
| ConversationSummaryMemory          | Resume la conversación usando un LLM para obtener un resumen compacto y relevante.                       | Conversaciones largas o multitópico                         |
| EntityMemory / EntityStoreMemory   | Extrae y almacena entidades (nombres, lugares, fechas) y sus atributos a lo largo de la conversación.    | Asistentes personalizados, CRM, sistemas de recomendación   |
| VectorStore-Backed Memory          | Almacena recuerdos en una base de datos vectorial y recupera los más relevantes según el contexto.        | Recuperación de información, QA, chatbots con memoria larga |
| DynamoDB/Momento/Redis/Upstash     | Variantes que almacenan la memoria en bases de datos externas para persistencia a largo plazo y escalabilidad. | Soporte de sesiones largas, multiusuario, persistencia real |
| Motörhead / Zep                    | Servidores de memoria avanzados que permiten sumarización incremental, embedding, indexación y enriquecimiento de historiales. | Aplicaciones avanzadas, análisis de conversaciones           |



```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("¡Hola! Me llamo Juan.")
memory.chat_memory.add_ai_message("¡Hola, Juan!")
memory.load_memory_variables({})
```




    {'history': 'Human: ¡Hola! Me llamo Juan.\nAI: ¡Hola, Juan!'}




```python
from langchain.chains.conversation.base import ConversationChain

conversation = ConversationChain(
    llm=llm, verbose=True, memory=memory
)
conversation.invoke(input="¿Cómo me llamo?")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: ¡Hola! Me llamo Juan.
    AI: ¡Hola, Juan!
    Human: ¿Cómo me llamo?
    AI:[0m
    
    [1m> Finished chain.[0m





    {'input': '¿Cómo me llamo?',
     'history': 'Human: ¡Hola! Me llamo Juan.\nAI: ¡Hola, Juan!',
     'response': '¡Te llamas Juan! ¡Es un placer conocerte, Juan! I will try my best to remember your name for future interactions. I have access to the current conversation and can use that information to recall things like your name.'}




```python
conversation.invoke(input="Quiero que me llames Juanito.")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: ¡Hola! Me llamo Juan.
    AI: ¡Hola, Juan!
    Human: ¿Cómo me llamo?
    AI: ¡Te llamas Juan! ¡Es un placer conocerte, Juan! I will try my best to remember your name for future interactions. I have access to the current conversation and can use that information to recall things like your name.
    Human: Quiero que me llames Juanito.
    AI:[0m
    
    [1m> Finished chain.[0m





    {'input': 'Quiero que me llames Juanito.',
     'history': 'Human: ¡Hola! Me llamo Juan.\nAI: ¡Hola, Juan!\nHuman: ¿Cómo me llamo?\nAI: ¡Te llamas Juan! ¡Es un placer conocerte, Juan! I will try my best to remember your name for future interactions. I have access to the current conversation and can use that information to recall things like your name.',
     'response': '¡Entendido! A partir de ahora, te llamaré Juanito. ¡Espero que te guste ese nombre! I have updated my internal representation of you to reflect this preference. Please let me know if you would like to change it again. I am quite flexible, although I may sometimes slip up as I am still under development.'}




```python
conversation.invoke(input="¿Cómo me llamo?")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: ¡Hola! Me llamo Juan.
    AI: ¡Hola, Juan!
    Human: ¿Cómo me llamo?
    AI: ¡Te llamas Juan! ¡Es un placer conocerte, Juan! I will try my best to remember your name for future interactions. I have access to the current conversation and can use that information to recall things like your name.
    Human: Quiero que me llames Juanito.
    AI: ¡Entendido! A partir de ahora, te llamaré Juanito. ¡Espero que te guste ese nombre! I have updated my internal representation of you to reflect this preference. Please let me know if you would like to change it again. I am quite flexible, although I may sometimes slip up as I am still under development.
    Human: ¿Cómo me llamo?
    AI:[0m
    
    [1m> Finished chain.[0m





    {'input': '¿Cómo me llamo?',
     'history': 'Human: ¡Hola! Me llamo Juan.\nAI: ¡Hola, Juan!\nHuman: ¿Cómo me llamo?\nAI: ¡Te llamas Juan! ¡Es un placer conocerte, Juan! I will try my best to remember your name for future interactions. I have access to the current conversation and can use that information to recall things like your name.\nHuman: Quiero que me llames Juanito.\nAI: ¡Entendido! A partir de ahora, te llamaré Juanito. ¡Espero que te guste ese nombre! I have updated my internal representation of you to reflect this preference. Please let me know if you would like to change it again. I am quite flexible, although I may sometimes slip up as I am still under development.',
     'response': "¡Te llamas Juanito! I am doing my best to remember. Is that correct? I can double-check my notes if you'd like. I'm always striving for accuracy!"}



Cuando las entradas se hacen largas, quizás no queramos enviar la conversación completa, sino un resumen.


```python
from langchain.memory import ConversationSummaryBufferMemory

review = "Pedí Pizza Salami por 9.99$ y ¡estaba increíble! \
La pizza fue entregada a tiempo y todavía estaba caliente cuando la recibí. \
La masa era fina y crujiente, y los ingredientes eran frescos y sabrosos. \
El Salami estaba bien cocido y complementaba el queso perfectamente. \
El precio era razonable y creo que obtuve el valor de mi dinero. \
En general, estoy muy satisfecho con mi pedido y recomendaría esta pizzería a otros."

summary_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100) # Si el token es 1000, no se creará ningún resumen
summary_memory.save_context(
    {"input": "Hola, ¿cómo puedo ayudarte hoy?"},
    {"output": "¿Podrías analizar una reseña por favor?"},
)
summary_memory.save_context(
    {"input": "Claro, con gusto. ¿Podrías proporcionar la reseña?"},
    {"output": f"{review}"},
)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=summary_memory
)

conversation.invoke(input="Muchas gracias")

```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    System: The human greets the user in Spanish and asks how they can help. The user asks if the AI can analyze a review. The human agrees and asks for the review. The user provides a positive review in Spanish about a salami pizza they ordered.
    Human: Muchas gracias
    AI:[0m
    
    [1m> Finished chain.[0m





    {'input': 'Muchas gracias',
     'history': 'System: The human greets the user in Spanish and asks how they can help. The user asks if the AI can analyze a review. The human agrees and asks for the review. The user provides a positive review in Spanish about a salami pizza they ordered.',
     'response': 'De nada! I\'m glad I could help. So, that salami pizza review... it sounds delicious! Based on your description, here\'s a quick breakdown of what I can analyze:\n\n*   **Sentiment Analysis:** I can definitely confirm that the review is overwhelmingly positive. Words like "deliciosa" and the general enthusiasm point to a very happy customer.\n\n*   **Aspect-Based Sentiment Analysis (ABSA):** I could break down the review to identify specific aspects the customer liked. For example:\n\n    *   **Food:** The salami pizza itself is the primary focus, and the sentiment is positive.\n    *   **Taste/Flavor:** "Deliciosa" indicates a positive sentiment towards the taste.\n    *   **Ingredients:** While not explicitly mentioned, "salami" being highlighted suggests the quality or type of salami was pleasing.\n\n*   **Keywords:** Obvious keywords include "pizza," "salami," and "deliciosa." I could also identify less obvious keywords depending on the context you want to extract.\n\nIs there anything specific you\'d like me to do with the review now that I have it? For example, would you like me to generate a summary of the review, or compare it to other reviews? Let me know!'}



### Tools

LangChain permite definir herramientas que pueden ser utilizadas por los modelos de lenguaje para realizar acciones específicas, como consultar bases de datos, llamar a APIs externas o ejecutar código. Estas herramientas pueden ser integradas en las cadenas y utilizadas por los agentes para tomar decisiones informadas.

Sin tools, el modelo no puede responder a esta pregunta.


```python
llm.invoke("¿Qué tiempo hace en Majadahonda ahora mismo?")
```




    AIMessage(content='Lo siento, no tengo acceso a información meteorológica en tiempo real. Sin embargo, puedes consultar el tiempo actual en Majadahonda en las siguientes páginas web:\n\n*   **El Tiempo.es:** [https://www.eltiempo.es/majadahonda.html](https://www.eltiempo.es/majadahonda.html)\n*   **Meteored:** [https://www.tiempo.com/majadahonda.htm](https://www.tiempo.com/majadahonda.htm)\n\nTambién puedes buscar en Google "tiempo en Majadahonda" y te aparecerá la información directamente en los resultados de búsqueda.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--e55b456f-32a2-4901-99cf-82043241aa30-0', usage_metadata={'input_tokens': 11, 'output_tokens': 135, 'total_tokens': 146, 'input_token_details': {'cache_read': 0}})




```python
from langchain_core.tools import tool


@tool
def fake_weather_api(city: str) -> str:
    """
    Verifica el clima en una ciudad especificada.

    Args:
        city (str): El nombre de la ciudad donde quieres verificar el clima.

    Returns:
        str: Una descripción del clima actual en la ciudad especificada.
    """
    return "Soleado, 22°C"


@tool
def outdoor_seating_availability(city: str) -> str:
    """
    Verifica si hay asientos al aire libre disponibles en un restaurante específico en una ciudad dada.

    Args:
        city (str): El nombre de la ciudad donde quieres verificar la disponibilidad de asientos al aire libre.

    Returns:
        str: Un mensaje indicando si hay asientos al aire libre disponibles o no.
    """
    return "Asientos al aire libre disponibles."


tools = [fake_weather_api, outdoor_seating_availability]
```


```python
llm_with_tools = llm.bind_tools(tools)
```


```python
results = llm_with_tools.invoke("¿Qué tiempo hace en Majadahonda ahora mismo?")
results
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'fake_weather_api', 'arguments': '{"city": "Majadahonda"}'}}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--3165abbc-d12f-413d-8ddd-c90db7db3303-0', tool_calls=[{'name': 'fake_weather_api', 'args': {'city': 'Majadahonda'}, 'id': '94df6c5d-bb09-45ca-8f47-0605cae70881', 'type': 'tool_call'}], usage_metadata={'input_tokens': 141, 'output_tokens': 9, 'total_tokens': 150, 'input_token_details': {'cache_read': 0}})



El modelo puede pedir que se invoquen varias herramientas.


```python
results = llm_with_tools.invoke("¿Qué tiempo hace en Majadahonda ahora mismo? ¿Hay asientos al aire libre disponibles?")
results
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'outdoor_seating_availability', 'arguments': '{"city": "Majadahonda"}'}}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--32671f9c-04cb-4ae4-b72a-368576095fdb-0', tool_calls=[{'name': 'fake_weather_api', 'args': {'city': 'Majadahonda'}, 'id': 'c92de3a6-3fcd-4764-beb5-7c9b3b417e3c', 'type': 'tool_call'}, {'name': 'outdoor_seating_availability', 'args': {'city': 'Majadahonda'}, 'id': 'ff953045-2d17-42fd-af3e-469c85598ff1', 'type': 'tool_call'}], usage_metadata={'input_tokens': 149, 'output_tokens': 19, 'total_tokens': 168, 'input_token_details': {'cache_read': 0}})



Otra forma de hacerlo.


```python
from langchain_core.messages import HumanMessage, ToolMessage

messages = [
    HumanMessage(
        "¿Qué tiempo hace en Majadahonda ahora mismo? ¿Hay asientos al aire libre disponibles?"
    )
]
llm_output = llm_with_tools.invoke(messages)
llm_output

```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'outdoor_seating_availability', 'arguments': '{"city": "Majadahonda"}'}}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--4b55fee4-6ab9-4b67-811c-886177b2da0b-0', tool_calls=[{'name': 'fake_weather_api', 'args': {'city': 'Majadahonda'}, 'id': '01ae8b2d-c3d2-42bb-a388-ac816ba691f0', 'type': 'tool_call'}, {'name': 'outdoor_seating_availability', 'args': {'city': 'Majadahonda'}, 'id': '8a6d7e71-0da0-46d5-b2a0-b8409f192728', 'type': 'tool_call'}], usage_metadata={'input_tokens': 149, 'output_tokens': 19, 'total_tokens': 168, 'input_token_details': {'cache_read': 0}})



Añadimos la respuesta del modelo.


```python
messages.append(llm_output)
```

Somos nosotros los que llamamos a las tools y proporcionamos el resultado al modelo.


```python
tool_mapping = {
    "fake_weather_api": fake_weather_api,
    "outdoor_seating_availability": outdoor_seating_availability,
}
```


```python
from langchain_core.messages import ToolMessage

for tool_call in llm_output.tool_calls:
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
```


```python
messages
```




    [HumanMessage(content='¿Qué tiempo hace en Majadahonda ahora mismo? ¿Hay asientos al aire libre disponibles?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='', additional_kwargs={'function_call': {'name': 'outdoor_seating_availability', 'arguments': '{"city": "Majadahonda"}'}}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--4b55fee4-6ab9-4b67-811c-886177b2da0b-0', tool_calls=[{'name': 'fake_weather_api', 'args': {'city': 'Majadahonda'}, 'id': '01ae8b2d-c3d2-42bb-a388-ac816ba691f0', 'type': 'tool_call'}, {'name': 'outdoor_seating_availability', 'args': {'city': 'Majadahonda'}, 'id': '8a6d7e71-0da0-46d5-b2a0-b8409f192728', 'type': 'tool_call'}], usage_metadata={'input_tokens': 149, 'output_tokens': 19, 'total_tokens': 168, 'input_token_details': {'cache_read': 0}}),
     ToolMessage(content='Soleado, 22°C', tool_call_id='01ae8b2d-c3d2-42bb-a388-ac816ba691f0'),
     ToolMessage(content='Asientos al aire libre disponibles.', tool_call_id='8a6d7e71-0da0-46d5-b2a0-b8409f192728')]




```python
llm_with_tools.invoke(messages)
```




    AIMessage(content='El tiempo en Majadahonda es soleado, 22°C. Hay asientos al aire libre disponibles.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--44af3e03-518b-443d-85dc-2b878dcdb157-0', usage_metadata={'input_tokens': 196, 'output_tokens': 24, 'total_tokens': 220, 'input_token_details': {'cache_read': 0}})



### RAG

LangChain facilita la implementación de Retrieval-Augmented Generation (RAG), que combina la generación de texto con la recuperación de información relevante de bases de datos o documentos.


```python
from google.colab import userdata
import os

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```


```python
!kaggle datasets download -d kotartemiy/topic-labeled-news-dataset
```


```python
import zipfile

# Define the path to your zip file
file_path = '/content/topic-labeled-news-dataset.zip'

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/datasets')
```


```python
import pandas as pd

df = pd.read_csv('/content/datasets/labelled_newscatcher_dataset.csv', sep=';')
```


```python
MAX_NEWS = 1000
DOCUMENT="title"
TOPIC="topic"

subset_news = df.head(MAX_NEWS)
```

Aunque hemos leído en `dataset` en Pandas, LangChain puede cargar directamente el fichero `csv` con la librería `document_loader` y cargarlo en ChromaDB:


```python
!pip install -q langchain
!pip install -q langchain_community
```


```python
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
```

Creamos el`loader`, indicando la fuente de datos y el nombre de la columna en el `dataframe` que contiene la información.


```python
df_loader = DataFrameLoader(subset_news, page_content_column=DOCUMENT)
```

Cargamos y mostramos el documento. Se observa que usa como `metadata` el resto de campos.


```python
df_document = df_loader.load()
display(df_document[:2])
```

Ahora generamos los embeddings. Para ello, será necesario importar **CharacterTextSplitter:** para agrupar la información en `chunks`.



```python
from langchain.text_splitter import CharacterTextSplitter
```

No existe una forma 100% correcta de dividir los documentos en chunks). La clave está en equilibrar el contexto y el uso de memoria:

- **Fragmentos más grandes:** Proporcionan al modelo más contexto, lo que puede llevar a una mejor comprensión y respuestas más precisas. Sin embargo, consumen más memoria.
- **Fragmentos más pequeños:** Reducen el uso de memoria, pero pueden limitar la comprensión contextual del modelo si la información queda demasiado fragmentada.

Se ha decidido usar un tamaño medio de 250 caracteres para cada `chunk` con un `overloap` de 10 caracteres. Es decir, los 10 caracteres finales de un `chunk`, serán los 10 primeros del siguiente.



```python
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(df_document)
display(texts[:2])
```

Ahora creamos los `embeddings`. Se puede usar directamente LangChain para hacer esto.


```python
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

Creamos la base de datos. Esta instrucción también crea los índices.


```python
!pip install -q chromadb
```


```python
chroma_db = Chroma.from_documents(
    texts, embedding_function
)
```

El siguiente paso es especificar el `retriever`, que recupera información de los documentos que le proporcionemos. En este caso hace una búsqueda por proximidad de los `embbeddings` almacenados en ChromaDB. El último paso es seleccionar el modelo de lenguaje que recibirá la `pipeline` de Hugging Face.


```python
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
```


```python
retriever = chroma_db.as_retriever()
```


```python
model_id = "google/flan-t5-large"
task="text2text-generation"

hf_llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task=task,
    device_map="auto",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "repetition_penalty":1.1, # penaliza que el modelo repita respuestas en el prompt. Parece que algunos modeos lo hacen
    },
)
```

Ahora configuramos la  `pipeline`:


```python
document_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, retriever=retriever, chain_type='stuff'
)
```

`chain_type` puede tener los siguientes valores:

- **stuff:** La opción más sencilla; simplemente toma los documentos que considera apropiados y los utiliza en el prompt que se pasa al modelo.
- **refine:** Realiza múltiples llamadas al modelo con diferentes documentos, intentando obtener una respuesta más refinada cada vez. Puede ejecutar un número elevado de llamadas al modelo, por lo que debe usarse con precaución.
- **map_reduce:** Intenta reducir todos los documentos en uno solo, posiblemente a través de varias iteraciones. Puede comprimir y condensar los documentos para que quepan en el prompt enviado al modelo.
- **map_rerank:** Llama al modelo para cada documento y los clasifica, devolviendo finalmente el mejor. Similar a refine, puede ser arriesgado dependiendo del número de llamadas que se prevea realizar.


Ahora, podemos hacer la pregunta:


```python
response = document_qa.invoke("Can I buy a Toshiba laptop?")

display(response)
```

La respuesta es correcta. No se obtiene mucha información porque el modelo usado, T5, no está específicamente preparado para la generación de texto.


```python
response = document_qa.invoke("Can I buy a Acer 3 laptop?")

display(response)
```

### Agents

LangChain permite crear agentes que pueden tomar decisiones, consultar diferentes fuentes de datos y ejecutar acciones de forma autónoma. Los agentes pueden utilizar herramientas y modelos de lenguaje para interactuar con el entorno y resolver tareas complejas.

Los agentes pueden usar tools predefinidas.


```python
from langchain.agents import load_tools
from langchain.agents import AgentType

tool_names = ["llm-math"]
tools = load_tools(tool_names, llm=llm)
tools
```




    [Tool(name='Calculator', description='Useful for when you need to answer questions about math.', func=<bound method Chain.run of LLMMathChain(verbose=False, llm_chain=LLMChain(verbose=False, prompt=PromptTemplate(input_variables=['question'], input_types={}, partial_variables={}, template='Translate a math problem into a expression that can be executed using Python\'s numexpr library. Use the output of running this code to answer the question.\n\nQuestion: ${{Question with math problem.}}\n```text\n${{single line mathematical expression that solves the problem}}\n```\n...numexpr.evaluate(text)...\n```output\n${{Output of running the code}}\n```\nAnswer: ${{Answer}}\n\nBegin.\n\nQuestion: What is 37593 * 67?\n```text\n37593 * 67\n```\n...numexpr.evaluate("37593 * 67")...\n```output\n2518731\n```\nAnswer: 2518731\n\nQuestion: 37593^(1/5)\n```text\n37593**(1/5)\n```\n...numexpr.evaluate("37593**(1/5)")...\n```output\n8.222831614237718\n```\nAnswer: 8.222831614237718\n\nQuestion: {question}\n'), llm=ChatGoogleGenerativeAI(model='models/gemini-2.0-flash', google_api_key=SecretStr('**********'), client=<google.ai.generativelanguage_v1beta.services.generative_service.client.GenerativeServiceClient object at 0x7f3b17b20590>, default_metadata=(), model_kwargs={}), output_parser=StrOutputParser(), llm_kwargs={}))>, coroutine=<bound method Chain.arun of LLMMathChain(verbose=False, llm_chain=LLMChain(verbose=False, prompt=PromptTemplate(input_variables=['question'], input_types={}, partial_variables={}, template='Translate a math problem into a expression that can be executed using Python\'s numexpr library. Use the output of running this code to answer the question.\n\nQuestion: ${{Question with math problem.}}\n```text\n${{single line mathematical expression that solves the problem}}\n```\n...numexpr.evaluate(text)...\n```output\n${{Output of running the code}}\n```\nAnswer: ${{Answer}}\n\nBegin.\n\nQuestion: What is 37593 * 67?\n```text\n37593 * 67\n```\n...numexpr.evaluate("37593 * 67")...\n```output\n2518731\n```\nAnswer: 2518731\n\nQuestion: 37593^(1/5)\n```text\n37593**(1/5)\n```\n...numexpr.evaluate("37593**(1/5)")...\n```output\n8.222831614237718\n```\nAnswer: 8.222831614237718\n\nQuestion: {question}\n'), llm=ChatGoogleGenerativeAI(model='models/gemini-2.0-flash', google_api_key=SecretStr('**********'), client=<google.ai.generativelanguage_v1beta.services.generative_service.client.GenerativeServiceClient object at 0x7f3b17b20590>, default_metadata=(), model_kwargs={}), output_parser=StrOutputParser(), llm_kwargs={}))>)]




```python
from langchain.agents import initialize_agent

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         max_iterations=3)
```

    /tmp/ipykernel_349417/3660245525.py:3: LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_ as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_ and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.
      agent = initialize_agent(tools,



```python
agent.invoke("¿Qué día es hoy?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mI cannot answer the question "Qué día es hoy?" because I do not have access to a calendar or real-time date information. My capabilities are limited to mathematical calculations.
    Final Answer: I cannot answer this question.[0m
    
    [1m> Finished chain.[0m





    {'input': '¿Qué día es hoy?', 'output': 'I cannot answer this question.'}




```python
agent.invoke("¿Cuánto es 2 elevado a la potencia de 10?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mI need to calculate 2 to the power of 10.
    Action: Calculator
    Action Input: 2^10[0m
    Observation: [36;1m[1;3mAnswer: 1024[0m
    Thought:[32;1m[1;3mI now know the final answer.
    Final Answer: 1024[0m
    
    [1m> Finished chain.[0m





    {'input': '¿Cuánto es 2 elevado a la potencia de 10?', 'output': '1024'}



También pueden usar tools personalizadas.


```python
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

# 1. Define tu herramienta personalizada usando @tool
@tool
def invertir_texto(texto: str) -> str:
    """Invierte el texto proporcionado."""
    return texto[::-1]


# 3. Inicializa el agente con la herramienta custom
agent = initialize_agent(
    tools=[invertir_texto],  # Lista de herramientas; aquí solo la custom
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. Usa el agente
resultado = agent.run("Invierte el texto: LangChain es genial")
print(resultado)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mI need to reverse the given text.
    Action: invertir_texto
    Action Input: LangChain es genial[0m
    Observation: [36;1m[1;3mlaineg se niahCgnaL[0m
    Thought:[32;1m[1;3mI now know the final answer
    Final Answer: laineg se niahCgnaL[0m
    
    [1m> Finished chain.[0m
    laineg se niahCgnaL


El agente puede combinarse con un chat.


```python
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run("Invierte esta cadena: LangChain es genial")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m```json
    {
        "action": "Final Answer",
        "action_input": "laienG se niahCgnaL"
    }
    ```[0m
    
    [1m> Finished chain.[0m





    'laienG se niahCgnaL'


