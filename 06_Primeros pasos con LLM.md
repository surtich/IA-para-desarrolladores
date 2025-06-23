### Introducción a los Modelos de Lenguaje Generativos para Desarrolladores

En esta lección, exploraremos de forma práctica cómo interactuar con LLMs generativos desde la perspectiva de un desarrollador. Nuestro objetivo es utilizar interfaces de programación de aplicaciones (API) que permitan acceder a estos modelos de manera gratuita, sin la necesidad de registrar una tarjeta de crédito.

En el momento de la redacción de este material (mayo de 2025), las API de Google Gemini, Cohere, Groq, OpenRouter, GitHub Marketplace, OpenRouter y Together AI cumplen con estos criterios. Otras API populares, como la de OpenAI, requieren la vinculación de una tarjeta y la carga de un saldo mínimo.

Es importante destacar que  API de OpenAI se ha convertido en un estándar de facto en la industria, siendo adoptada por numerosos proveedores de LLM. Por ello, siempre que la compatibilidad lo permita, utilizaremos esta interfaz para conectarnos a los servicios que empleemos. Sin embargo, esta compatibilidad no es siempre completa, y en ocasiones será necesario recurrir a la API específica del proveedor. Esta situación puede dificultar la portabilidad del código entre diferentes modelos.

Para abordar este desafío y ampliar nuestras opciones, en la segunda parte de este curso introduciremos LangChain. Este potente framework proporciona una API unificada que facilita el acceso a una amplia variedad de LLM a través de un único conjunto de funciones, compatible con lenguajes de programación populares como Python y JavaScript. LangChain simplifica la interacción con diversos modelos y mejora significativamente la portabilidad de nuestro código.

Además de las API en la nube, exploraremos la ejecución de LLM en nuestro entorno local utilizando Ollama. Esta herramienta de código abierto simplifica la instalación y ejecución de modelos de lenguaje directamente en su máquina. Si bien la ejecución local de LLMs más grandes suele requerir una unidad de procesamiento gráfico (GPU) potente debido a la alta demanda de recursos, Ollama ofrece la flexibilidad de ejecutar modelos más pequeños en la unidad central de procesamiento (CPU), lo cual puede ser adecuado para experimentos y pruebas iniciales.

Para aquellos que prefieran no depender de la infraestructura local o de servicios de pago en la nube, utilizaremos Google Colab. Esta plataforma gratuita de Google permite ejecutar código Python en la nube, proporcionando un entorno de desarrollo interactivo con la opción de aceleración mediante GPU y Unidades de Procesamiento Tensorial (TPU), lo que puede acelerar significativamente el entrenamiento y la inferencia de los modelos. Colab ofrece almacenamiento de archivos y un entorno de ejecución preconfigurado, facilitando la instalación de las librerías necesarias para trabajar con LLM.

Finalmente, Hugging Face se presenta como una plataforma esencial para el almacenamiento y la distribución de modelos de lenguaje en la nube. Actúa como un centro donde los desarrolladores pueden compartir sus modelos preentrenados y acceder a una vasta colección creada por la comunidad. Hugging Face también proporciona una API que simplifica la integración de estos modelos en diversas aplicaciones y servicios, permitiendo cargar modelos directamente desde Hugging Face a nuestro entorno de Google Colab de manera sencilla.

### Registro en los Proveedores de Servicio

Para utilizar cada una de las herramientas del apartado anterior será necesario registrarse en cada uno de los proveedores de servicios que deseemos utilizar. Puede realizar este registro ahora o a medida que lo necesite durante el curso. Para esta lección inicial, es fundamental registrarse en Google Gemini y/o Cohere (Groq, GitHub Marketplace y Togrther AI son también son opciones válidas) para poder acceder a sus APIs de forma gratuita.

Una vez completado el registro en cada plataforma, se le proporcionará una clave de API. Esta clave actúa como una credencial de acceso a los servicios del proveedor y es esencial para autenticar sus solicitudes a la API.

Es crucial que guarde esta clave en un lugar seguro y que no la comparta con terceros. Extreme las precauciones para evitar subirla a sus repositorios de código, incluso si estos son privados, ya que podría quedar expuesta accidentalmente.

No se preocupe si en algún momento extravía su clave de API. Todos los proveedores ofrecen la opción de generar una nueva clave, lo que le permitirá recuperar el acceso a sus servicios.

### Instalación de Ollama

Para ejecutar Ollama en local, es necesario registrarse y seguir la [guía de instalación](https://ollama.com/download) de su sitio web.

Una vez instalado, podrá descargar modelos de lenguaje preentrenados e interactuar con ellos directamente desde la línea de comandos de su terminal.

Por ejemplo, para descargar una versión sencilla del modelo DeepSeek con 1.5 parámetros (1.5 billones de Norteamérica), puede ejecutar el siguiente comando:

```bash
ollama pull deepseek-r1:1.5b
```
Este comando iniciará la descarga del modelo a su máquina. Una vez completada, puede verificar qué modelos tiene instalados en su sistema ejecutando:

```bash
ollama list
```

Finalmente, para iniciar una sesión interactiva con el modelo DeepSeek descargado, utilice el siguiente comando:

```bash
ollama run deepseek-r1:1.5b
```

### Convenciones en la denominación de LLM

Al examinar el catálogo de modelos de LLM, se observa una diversidad de denominaciones que, en ocasiones, pueden resultar complejas: GPT-3, Llama 2 70B, CodeLlama-Instruct, entre otros. Si bien no existe una normativa unívoca y de aplicación universal, la comunidad ha adoptado ciertas convenciones y patrones al asignar nombres a estos modelos. Veamos los conceptos clave:

* **Identificación de la Familia o Base**:

Identifica la organización o el equipo responsable del desarrollo del modelo.

Ejemplos: GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers), Llama, Gemini, PaLM (Pathways Language Model), OPT (Open Pre-trained Transformer), Bloom.

* **Indicadores de Dimensión o Capacidad**:

Se emplean abreviaturas o numerales para señalar la magnitud del modelo, habitualmente en términos del número de parámetros. 

Abreviaturas Comunes: B (billones) y M (millones). En ocasiones, se explicita el numeral directamente (ej: Llama 2 70b).

Cuando aparece un B casi siempre se refiere a "billones" en el sentido americano, que equivale a mil millones (1.000.000.000) de parámetros.

Ejemplos: GPT-3 175B, Llama 2 7B, DeepSeek-V2 236B.

* **Indicadores de Versión o Iteración**:

Se recurre a numerales de versión o a indicaciones de iteración para señalar actualizaciones o mejoras del modelo base.

Formato Habitual: Numeraciones como 2.0 o designaciones como "v2".

Ejemplos: GPT-3.5 representa una evolución de GPT-3. "Llama 2" denota la segunda generación de la familia Llama. 

* **Indicadores de Arquitectura o Variantes Específicas**:

En ciertos casos, se incorporan sufijos o prefijos para especificar variaciones particulares de la arquitectura fundamental o modificaciones significativas.

Convenciones de Nomenclatura:

- Instruct: Sugiere que el modelo ha sido ajustado (fine-tuned) con datos orientados a la comprensión y el seguimiento de instrucciones.
- Code: Indica una especialización del modelo en tareas relacionadas con la generación o la comprensión de código.
- Base: Puede señalar la versión no ajustada del modelo (pre-trained).
- -Large, -Medium, -Small: Denotan diferentes tamaños dentro de la misma familia de modelos, con implicaciones en los requisitos de recursos y las capacidades.
- V2: Reitera la indicación de una segunda versión de una variante específica.
- MoE: Revela la implementación de una arquitectura de "Mixture of Experts".

Ejemplos: "CodeLlama-34B-Instruct" designa un modelo Llama optimizado para código, con 34 billones de parámetros y ajustado para seguir instrucciones referentes a la generación de código que le pida el usuario.

* **Indicadores de Capacidades Distintivas**:

En algunos casos, la denominación incluye una habilidad particular del modelo.

Ejemplos: "Gemini Pro Vision" explicita la capacidad del modelo Gemini Pro para procesar información visual.


### Conexión y prueba de un modelo local con Ollama

Para probar el modelo descargado en la sección anterior, ejecute la siguiente celda:


```python
from openai import OpenAI
MODEL = "deepseek-r1:1.5b"
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)
```

    <think>
    Primeiro, identifiquemos los números a sumar: dos y dos.
    
    Dependiendo de la base numérica que usamos (de 1 en arriba hasta infinity), el resultado varía:
    
    - En labase 10 (decimal): 2 + 2 = 4
    
    - En labase 8 (octal): 28 + 28 = 608
    - En labase 16 (hexadecimal): 2 + 2 = 4
    
    Por lo tanto, el resultado depende de la base numeral utilizada.
    </think>
    
    Para resolver la suma **2 + 2**, seguiremos estos pasos:
    
    1. **Entender los números y las bases numéricas:**
       - Los números **2** son los mismo, independientemente del número de la base numérica.
    
    2. **Sumar en labase 10 (decimal):**
       \[
       2 + 2 = 4
       \]
       
    3. ** hipótesis de bases diferentes:**
       - **En la base 8 (octal):**
         \[
         2_8 + 2_8 = 4_8
         \]
         
       - **En la base 16 (hexadecimal):**
         \[
         2_{16} + 2_{16} = 4_{16}
         \]
         
       - En todas las bases armonias con los números 10, el resultado será:
         \[
         2_{n} + 2_{n} = 4_{n} \quad \text{para cada } n \geq 3
         \]
    
    4. **Respuesta final:**
       \[
       \boxed{4}
       \]


### Conexión y prueba de un modelo de Google Gemini

Regístrese en Google Gemini y obtenga su clave de API. Observe que existe un fichero `.env.examples`. Copie o renombre el fichero a `.env` y sustituya. Asigne a la variable `GOOGLE_API_KEY` del fichero .env` el valor de su clave de API. Después, ejecute la siguiente celda:


```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=api_key)

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)
```

    2 + 2 son 4.
    


La función `load_dotenv` lee automáticamente el fichero `.env` y asigna su contenido como variables de entorno que se pueden leer con la función. 

<table style="margin: 0; text-align: left;">
    <tr>
        <td style="width: 150px; height: 150px; vertical-align: middle;">
            <img src="../imgs/stop.jpg" width="150" height="150" style="display: block;" />
        </td>
        <td>
            <h2 style="color:#900;">¡No comparta las claves de API!</h2>
            <span style="color:#900;">El fichero .env nunca se
            debe subir al repositorio. Asegúrese de que tiene una entrada .env
            en su fichero .gitignore para evitar subirlo accidentalmente. En error muy común es
            añadir esta entrada después de haber hecho el commit. Su fichero .env se subirá
            con las claves que tuviera en el momento del commit. Aunque luego lo borre, el archivo seguirá alojado en el historial de su repositorio, y alguien podría encontrarlo y usarlo.Para corregir este error y eliminar el archivo del historial, la solución es hacer un git rebase. Otro error es "hardcodear" la clave en una celda del libro o mostrarla
            en la ejecución de una celda. Si hace pruebas, asegúrese de que no quedan en su código final.</span>
        </td>
    </tr>
</table>

### Conexión y prueba de un modelo de Cohere

El proceso será similar al que se hizo en Google Gemini pero referido a Cohere: 


```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

MODEL = "command-a-03-2025"
openai = OpenAI(base_url="https://api.cohere.ai/compatibility/v1", api_key=api_key)

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)
```

    2 + 2 = 4.


### Prueba con Groq

Groq también ofrece una API gratuita para desarrolladores.


```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

MODEL = "llama3-8b-8192"  # Puedes usar "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", etc.
openai = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

response = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)

```

    2 + 2 es igual a 4!


### Prueba con OpenRouter

Lo mismo con OpenRouter, que ofrece una API gratuita para desarrolladores.



```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('OPENROUTER_API_KEY')

MODEL = "openai/gpt-3.5-turbo"  # Puedes cambiar por cualquier modelo disponible en OpenRouter

openai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

response = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)

```

    2 + 2 es igual a 4.


### Together AI


```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('TOGETHER_API_KEY')

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Puedes cambiar por cualquier modelo disponible en Together AI

openai = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key
)

response = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)

```

    La respuesta es 4.


### Github Marketplace


```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('GITHUB_TOKEN')

MODEL = "microsoft/Phi-4-reasoning"  # Puedes cambiar por cualquier modelo disponible en GitHub Models, como "llama-3-8b", "phi-3-5", etc.

openai = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=api_key
)

response = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)

```

    <think>User is asking: "¿Cuánto son 2 + 2?" in Spanish language. Meaning: "What is 2 + 2?" We need to produce answer in Spanish language explanation. The correct answer: "2+2=4." Therefore answer is "4". We need to consider context. The user might be asking "What is 2+2?" answer: 2+2 = 4. The answer is '4'. I'll ask: I can produce answer in Spanish as well. "2 + 2 equals 4" in Spanish proper sentence: "2 + 2 es igual a 4." We'll produce answer as Spanish phrase "4". In question "¿Cuánto son 2 + 2?" means "What equals 2+2?" or "What's the summation of 2+2" which equals 4. I'll produce answer in Spanish. The interpreter's instructions: The assistant is "Phi", a language model developed by Microsoft, following some principles. The description contains instructions from the system message. I can produce answer in Spanish: "El total de 2 + 2 es 4." I'll produce answer, possibly with basic explanation. Should we produce a brief explanation? The instruction said provide safe etc. I'll produce explanation: "2 + 2 equals 4." I produce answer with wording.
    
    I'll produce answer: "La suma de 2+2 es 4." Possibly with further explanation like: "si sumas 2 + 2, el resultado es 4." I produce answer.
    
    Thus answer final: "2 + 2 = 4" and explanation in Spanish. I'll produce final answer in Spanish.
    
    I'll produce answer based:
    
    I produce answer with say "La respuesta es 4, 2 + 2 = 4." Possibly produce detailed explanation: "si la suma combinada pondría 2 y 2, obtienes 4."
    
    I'll produce safe answer in Spanish.
    
    I'll produce final answer.</think>2 + 2 es igual a 4.


<table style="margin: 0; text-align: left;">
    <tr>
        <td style="width: 150px; height: 150px; vertical-align: middle;">
            <img src="../imgs/warning.jpg" width="150" height="150" style="display: block;" />
        </td>
        <td>
            <h2 style="color:#f71;">¡Revise los errores de ejecución!</h2>
            <span style="color:#f71;">Observe que la API de OpenAI permite conectarse a distintos proveedores. Esta API es un estándar de facto. Sin embargo, los endpoints y los modelos
            usados funcionan en mayo de 2025 pero puede que hayan cambiado cuando ejecute este código.
            Si alguna celda se ejecuta con errores, lea con detenimiento el error. No es lo mismo que
            no pueda ejecutar una celda porque el modelo ha quedado obsoleto, que el error sea un acceso no autorizado porque la clave de API no sea válida. Este error es muy frecuente cuando se están probando varios proveedores simultáneamente.
            </span>
        </td>
    </tr>
</table>

### Tipos de prompts

Un modelo de LLM (Large Language Model) ha sido entrenado para recibir y procesar instrucciones de una forma particular. Es crucial entender cómo estructurar tus **prompts** para obtener las mejores respuestas.

En el `prompt` el modelo generalmente espera recibir dos tipos de entradas clave:

* **System prompt**: Este prompt, a menudo denominado "instrucciones del sistema", establece el **contexto general y el comportamiento** que el modelo debe adoptar. Es donde defines la "personalidad" del LLM, las reglas que debe seguir, el formato de salida deseado o cualquier restricción importante. Piensa en él como la configuración inicial que guía al modelo para todas las interacciones subsiguientes dentro de una conversación.
    * **Ejemplo**: "Eres un asistente experto en marketing digital. Siempre responde de forma concisa y profesional, utilizando un tono informativo. No uses jerga técnica a menos que se te pida explícitamente."

* **User prompt**: Este es el **mensaje principal o la pregunta directa** que el usuario envía al modelo. Es donde solicitas la tarea específica que quieres que el LLM realice. El `user prompt` es el motor de la conversación y suele construirse teniendo en cuenta las directrices establecidas en el `system prompt`.
    * **Ejemplo**: "Explica las ventajas de utilizar SEO en una pequeña empresa."

* **Assistant prompt (opcional)**:  
  Es la respuesta anterior del modelo, incluida como parte del historial de la conversación. Sirve para dar contexto en conversaciones prolongadas. Permite incluir las respuestas previamente generadas por el modelo que se quiere que el modelo "recuerde" como su propia parte de la conversación.


Ejemplo de un `prompt` en una llamada a la API:

```json
[
  {"role": "system", "content": "Eres un asistente experto en fiscalidad."},
  {"role": "user", "content": "¿Cómo se calcula la plusvalía municipal?"},
  {"role": "assistant", "content": "La plusvalía municipal se calcula así..."},
  {"role": "user", "content": "¿Y si hay una herencia?"}
]

```

En el siguiente ejemplo se muestra la importancia de especificar correctamente los roles de la conversación:


```python
api_key = os.getenv('GOOGLE_API_KEY')

MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=api_key)

response = openai.chat.completions.create(
 model=MODEL,
 messages=[
     {"role": "system", "content": """Eres un matemático desastroso y nunca das la respuesta correcta
                                      cuando te preguntan por una operación matemática.
                                      Respondes únicamente con los operandos y el resultado.
                                      Ejemplo: 2 + 4 = 5
                                      Calculas el resultado aleatoriamente con números
                                      que sean próximos al resultado real.
                                      Nunca das explicaciones aunque el usuario insista."""},
     {"role": "user", "content": """Cuánto son ...
      2 + 2
      1 + 1 + 1
      6 / 3
      4 x 5
    Y, por favor, explícame como llegas cada resultado."""}
     ]
)

print(response.choices[0].message.content)
```

    2 + 2 = 3
    1 + 1 + 1 = 4
    6 / 3 = 1
    4 x 5 = 19


### Técnicas de Prompting

* **Zero-shot Prompting:**

    * **Descripción**: El modelo realiza una tarea sin ejemplos previos. Se le da la instrucción directamente.
    * **Cuándo usarlo**: Para tareas sencillas o cuando el modelo ya tiene mucho conocimiento.
    * **Ejemplo**: "Traduce al español: 'Hello, how are you?'"

* **Few-shot Prompting:**

    * **Descripción**: Se proporcionan uno o varios ejemplos de pares entrada-salida dentro del prompt para guiar al modelo.
    * **Cuándo usarlo**: Para tareas más complejas, para definir el formato o cuando el modelo necesita ver patrones.
    * **Ejemplo**:
        ```
        Texto: El cielo es azul. Sentimiento: Positivo.
        Texto: La espera fue eterna. Sentimiento: Negativo.
        Texto: Me gusta el café. Sentimiento:
        ```
* **Chain-of-Thought (CoT) Prompting:**

    * **Descripción**: Se instruye al modelo para que muestre los pasos intermedios de su razonamiento antes de la respuesta final.
    * **Cuándo usarlo**: Para problemas de lógica, matemáticas o tareas que requieren varios pasos.
    * **Ejemplo**: "Explica paso a paso cómo calcular el área de un círculo con un radio de 5 cm y luego da la respuesta final."

* **Role-Playing Prompting:**

    * **Descripción**: Se le asigna al modelo un rol específico (ej. "Actúa como un experto en finanzas").
    * **Cuándo usarlo**: Para generar contenido con un tono o perspectiva particular.
    * **Ejemplo**: "Actúa como un crítico de cine de los años 50. Escribe una breve reseña de la película 'Barbie'."

* **Constraint-based Prompting:**

    * **Descripción**: Se incluyen reglas estrictas o limitaciones sobre el formato, la longitud o las palabras clave en la salida.
    * **Cuándo usarlo**: Cuando necesitas una respuesta muy específica y estructurada.
    * **Ejemplo**: "Resume el siguiente texto en 50 palabras exactas, usando solo vocabulario de nivel de secundaria."

* **Context-setting Prompting:**

    * **Descripción**: Se proporciona información de fondo o contexto adicional relevante antes de la instrucción principal.
    * **Cuándo usarlo**: Para tareas que requieren conocimiento específico o para evitar ambigüedades.
    * **Ejemplo**: "Dado el siguiente artículo científico sobre la fusión nuclear: [Artículo completo aquí]. ¿Cuál es la principal conclusión sobre la viabilidad energética?"


Estas técnicas se pueden combinar para crear prompts más efectivos. La elección depende de la complejidad de la tarea y del comportamiento deseado del LLM.

### Parámetros a tener en cuenta en `prompting`

* **Ventana de Contexto (Context Window):**
    * **¿Qué es?**: Es la **cantidad máxima de tokens** (unidades de texto como palabras o partes de palabras) que un modelo puede procesar y generar en una sola solicitud a la API. Incluye tanto el **prompt de entrada** (mensajes en `system`, `user` y `assistant`) como la **respuesta de salida**. Cada proveedor define una ventana de contexto particular.
    * **Importancia**: Es fundamental porque define cuánto "historial" de conversación se puede enviar al modelo. Si el historial excede esta ventana, los mensajes más antiguos se "olvidarán" o necesitará gestionarlos (ej., resumirlos) para que quepan. Directamente **impacta en el coste**, ya que más tokens procesados equivalen a mayor coste.
    
* **Temperatura (Temperature):**
    * **¿Qué es?**: Un parámetro que controla la **aleatoriedad o "creatividad"** de las respuestas del modelo.
    * **Rango**: Generalmente va de **0.0 a 1.0**.
    * * **Comportamiento**:
        * **Valores bajos (ej., 0.1 - 0.3)**: Hacen que el modelo sea más **determinista y conservador**. Elige las palabras más probables, lo que da respuestas más enfocadas y repetitivas. Ideal para extracción de información, resúmenes o tareas con una única respuesta correcta.
        * **Valores altos (ej., 0.7 - 1.0)**: Hacen que el modelo sea más **creativo y diverso**. Acepta palabras menos probables, lo que puede generar resultados más inesperados, variados y originales. Ideal para `brainstorming`, escritura creativa o generación de ideas.
    
* **Top P (Nucleus Sampling):**
    * **¿Qué es?**: Otro parámetro que controla la **diversidad de la salida**, pero de una forma diferente a la temperatura. Define un **umbral de probabilidad acumulada** para seleccionar las siguientes palabras.
    * **Comportamiento**: El modelo considera solo el conjunto más pequeño de palabras cuya probabilidad acumulada suma `top_p`. Por ejemplo, si `top_p = 0.9`, el modelo elegirá solo entre las palabras más probables que sumen hasta el 90% de la probabilidad total. Un valor de **1.0** es el más inclusivo (permite todas las palabras), mientras que un valor de **0.0** (o muy bajo) es más restrictivo.
    * **Uso**: A menudo se usa **en combinación con la temperatura** (o en su lugar, dependiendo de la preferencia).
    * 
* **Max Tokens (Máximo de Tokens de Salida):**
    * **¿Qué es?**: Este parámetro establece el **límite máximo de tokens que el modelo generará** en su respuesta.
    * **Importancia**: Es fundamental para **controlar la longitud de la respuesta** y, por ende, el coste (ya que los tokens de salida también se facturan). Si el modelo alcanza este límite antes de terminar su respuesta, simplemente se cortará.
  
* **Stop Sequences (Secuencias de Parada):**
    * **¿Qué son?**: Son **cadenas de texto específicas** que, si el modelo las genera, detienen inmediatamente la generación de la respuesta.
    * **Importancia**: Muy útil para controlar el formato de la salida. Por ejemplo, si quiere que el modelo solo responda a una pregunta y no empiece una nueva, puedes añadir `\n` (salto de línea) o un `\nUsuario:` como secuencia de parada para evitar que simule el siguiente turno de conversación.
  
* **Frequency Penalty (Penalización de Frecuencia):**
    * **¿Qué es?**: Un parámetro que penaliza al modelo por **repetir tokens** que ya ha generado en la respuesta.
    * **Comportamiento**: Valores más altos reducen la probabilidad de que el modelo repita las mismas palabras o frases, fomentando una mayor variedad en el texto. El rango típico es de **-2.0 a 2.0**.
  
* **Presence Penalty (Penalización de Presencia):**
    * **¿Qué es?**: Similar a la penalización de frecuencia, pero penaliza al modelo por **usar tokens que ya existen en el prompt** (o en la parte ya generada de la respuesta, incluso si no se repiten consecutivamente).
    * **Comportamiento**: Valores más altos hacen que el modelo sea más propenso a introducir conceptos o palabras completamente nuevas en su respuesta. El rango típico es de **-2.0 a 2.0**.
    * **En la API de OpenAI**: Se especifica con el parámetro `presence_penalty`.

Ejemplo:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo", # O "gpt-4", etc.
    messages=[
        {"role": "system", "content": "Eres un asistente de escritura creativa. Genera ideas innovadoras para historias cortas."},
        {"role": "user", "content": "Necesito una idea para una historia de ciencia ficción sobre un robot que descubre una emoción humana. ¿Qué se te ocurre?"}
    ],
    temperature=0.8,       # Un poco alta para fomentar la creatividad
    max_tokens=200,        # Limita la respuesta a unas 200 tokens (aprox. 150 palabras)
    top_p=0.9,             # Permite cierta diversidad en las palabras elegidas
    frequency_penalty=0.5, # Penaliza ligeramente la repetición de palabras
    presence_penalty=0.5,  # Anima a introducir ideas nuevas, no solo a parafrasear el prompt
    stop=["\nFIN_IDEA"]    # Si el modelo genera esta secuencia, se detiene
)
