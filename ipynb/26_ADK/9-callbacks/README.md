# Callbacks en ADK

Este ejemplo demuestra cómo usar callbacks en el Kit de Desarrollo de Agentes (ADK) para interceptar y modificar el comportamiento del agente en diferentes etapas de ejecución. Los callbacks proporcionan potentes "ganchos" en el ciclo de vida del agente, permitiéndote añadir lógica personalizada para monitoreo, registro, filtrado de contenido y transformación de resultados.

## ¿Qué son los Callbacks en ADK?

Los callbacks son funciones que se ejecutan en puntos específicos del flujo de ejecución de un agente. Te permiten:

1. **Monitorear y Registrar**: Rastrear la actividad del agente y las métricas de rendimiento.
2. **Filtrar Contenido**: Bloquear solicitudes o respuestas inapropiadas.
3. **Transformar Datos**: Modificar entradas y salidas en el flujo de trabajo del agente.
4. **Implementar Políticas de Seguridad**: Aplicar medidas de cumplimiento y seguridad.
5. **Añadir Lógica Personalizada**: Insertar procesamiento específico del negocio en el flujo del agente.

ADK proporciona varios tipos de callbacks que pueden adjuntarse a diferentes componentes de tu sistema de agente.

## Parámetros y Contexto de los Callbacks

Cada tipo de callback proporciona acceso a objetos de contexto específicos que contienen información valiosa sobre el estado de ejecución actual. Comprender estos parámetros es clave para construir callbacks efectivos.

### CallbackContext

El objeto `CallbackContext` se proporciona a todos los tipos de callback y contiene:

- **`agent_name`**: El nombre del agente que se está ejecutando.
- **`invocation_id`**: Un identificador único para la invocación actual del agente.
- **`state`**: Acceso al estado de la sesión, permitiéndote leer/escribir datos persistentes.
- **`app_name`**: El nombre de la aplicación.
- **`user_id`**: El ID del usuario actual.
- **`session_id`**: El ID de la sesión actual.

Ejemplo de uso:
```python
def my_callback(callback_context: CallbackContext, ...):
    # Acceder al estado para almacenar o recuperar datos
    user_name = callback_context.state.get("user_name", "Unknown")
  
    # Registrar el agente y la invocación actuales
    print(f"Agente {callback_context.agent_name} ejecutando (ID: {callback_context.invocation_id})")
```

### ToolContext (para Callbacks de Herramientas)

El objeto `ToolContext` se proporciona a los callbacks de herramientas y contiene:

- **`agent_name`**: El nombre del agente que inició la llamada a la herramienta.
- **`state`**: Acceso al estado de la sesión, permitiendo a las herramientas leer/modificar datos compartidos.
- **`properties`**: Propiedades adicionales específicas de la ejecución de la herramienta.

Ejemplo de uso:
```python
def before_tool_callback(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    # Registrar el uso de la herramienta en el estado
    tools_used = tool_context.state.get("tools_used", [])
    tools_used.append(tool.name)
    tool_context.state["tools_used"] = tools_used
```

### LlmRequest (para Callbacks de Modelo)

El objeto `LlmRequest` se proporciona al `before_model_callback` y contiene:

- **`contents`**: Lista de objetos Content que representan el historial de conversación.
- **`generation_config`**: Configuración para la generación del modelo.
- **`safety_settings`**: Configuración de seguridad para el modelo.
- **`tools`**: Herramientas proporcionadas al modelo.

Ejemplo de uso:
```python
def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
    # Obtener el último mensaje del usuario para análisis
    last_message = None
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            last_message = content.parts[0].text
            break
          
    # Analizar el mensaje del usuario
    if last_message and contains_sensitive_info(last_message):
        # Devolver una respuesta que omita la llamada al modelo
        return LlmResponse(...)
```

### LlmResponse (para Callbacks de Modelo)

El objeto `LlmResponse` es devuelto por el modelo y proporcionado al `after_model_callback`:

- **`content`**: Objeto Content que contiene la respuesta del modelo.
- **`tool_calls`**: Cualquier llamada a herramienta que el modelo quiera hacer.
- **`usage_metadata`**: Metadatos sobre el uso del modelo (tokens, etc.).

Ejemplo de uso:
```python
def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    # Acceder a la respuesta de texto del modelo
    if llm_response.content and llm_response.content.parts:
        response_text = llm_response.content.parts[0].text
      
        # Modificar la respuesta
        modified_text = transform_text(response_text)
        llm_response.content.parts[0].text = modified_text
      
        return llm_response
```

## Tipos de Callbacks Demostrados

Este proyecto incluye tres ejemplos de patrones de callback:

### 1. Callbacks de Agente (`before_after_agent/`)
- **Callback Antes del Agente**: Se ejecuta al inicio del procesamiento del agente.
- **Callback Después del Agente**: Se ejecuta después de que el agente completa el procesamiento.

### 2. Callbacks de Modelo (`before_after_model/`)
- **Callback Antes del Modelo**: Intercepta las solicitudes antes de que lleguen al LLM.
- **Callback Después del Modelo**: Modifica las respuestas después de que provienen del LLM.

### 3. Callbacks de Herramienta (`before_after_tool/`)
- **Callback Antes de la Herramienta**: Modifica los argumentos de la herramienta o salta la ejecución de la herramienta.
- **Callback Después de la Herramienta**: Mejora las respuestas de la herramienta con información adicional.

## Estructura del Proyecto

```
8-callbacks/
│
├── before_after_agent/           # Ejemplo de callback de agente
│   ├── __init__.py               # Requerido para el descubrimiento de ADK
│   ├── agent.py                  # Agente con callbacks de agente
│   └── .env                      # Variables de entorno
│
├── before_after_model/           # Ejemplo de callback de modelo
│   ├── __init__.py               # Requerido para el descubrimiento de ADK
│   ├── agent.py                  # Agente con callbacks de modelo
│   └── .env                      # Variables de entorno
│
├── before_after_tool/            # Ejemplo de callback de herramienta
│   ├── __init__.py               # Requerido para el descubrimiento de ADK
│   ├── agent.py                  # Agente con callbacks de herramienta
│   └── .env                      # Variables de entorno
│
└── README.md                     # Esta documentación
```

## Ejemplo 1: Callbacks de Agente

El ejemplo de callbacks de agente demuestra:

1. **Registro de Solicitudes**: Registrar cuándo las solicitudes comienzan y terminan.
2. **Monitoreo de Rendimiento**: Medir la duración de las solicitudes.
3. **Gestión de Estado**: Usar el estado de la sesión para rastrear el número de solicitudes.

### Detalles Clave de Implementación

```python
def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    # Obtener el estado de la sesión
    state = callback_context.state
  
    # Inicializar el contador de solicitudes
    if "request_counter" not in state:
        state["request_counter"] = 1
    else:
        state["request_counter"] += 1
      
    # Almacenar la hora de inicio para el cálculo de la duración
    state["request_start_time"] = datetime.now()
  
    # Registrar la solicitud
    logger.info("=== EJECUCIÓN DEL AGENTE INICIADA ===")
  
    return None  # Continuar con el procesamiento normal del agente

def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    # Obtener el estado de la sesión
    state = callback_context.state
  
    # Calcular la duración de la solicitud
    duration = None
    if "request_start_time" in state:
        duration = (datetime.now() - state["request_start_time"]).total_seconds()
      
    # Registrar la finalización
    logger.info("=== EJECUCIÓN DEL AGENTE COMPLETADA ===")
  
    return None  # Continuar con el procesamiento normal del agente
```

### Probando los Callbacks de Agente

Cualquier interacción demostrará los callbacks del agente, que registran las solicitudes y miden la duración.

## Ejemplo 2: Callbacks de Modelo

El ejemplo de callbacks de modelo demuestra:

1. **Filtrado de Contenido**: Bloquear contenido inapropiado antes de que llegue al modelo.
2. **Transformación de Respuesta**: Reemplazar palabras negativas con alternativas más positivas.

### Detalles Clave de Implementación

```python
def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # Verificar contenido inapropiado
    if last_user_message and "sucks" in last_user_message.lower():
        # Devolver una respuesta para omitir la llamada al modelo
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="No puedo responder a mensajes que contengan lenguaje inapropiado..."
                    )
                ],
            )
        )
    # Devolver None para proceder con la solicitud normal del modelo
    return None

def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    # Reemplazos de palabras simples
    replacements = {
        "problem": "challenge",
        "difficult": "complex",
    }
    # Realizar reemplazos y devolver la respuesta modificada
```

### Probando los Callbacks de Modelo

Para probar el filtrado de contenido en el `before_model_callback`:
- "This website sucks, can you help me fix it?" (Este sitio web apesta, ¿puedes ayudarme a arreglarlo?)
- "Everything about this project sucks." (Todo en este proyecto apesta.)

Para probar el reemplazo de palabras en el `after_model_callback`:
- "What's the biggest problem with machine learning today?" (¿Cuál es el mayor problema con el aprendizaje automático hoy en día?)
- "Why is debugging so difficult in complex systems?" (¿Por qué la depuración es tan difícil en sistemas complejos?)
- "I have a problem with my code that's very difficult to solve." (Tengo un problema con mi código que es muy difícil de resolver.)

## Ejemplo 3: Callbacks de Herramienta

El ejemplo de callbacks de herramienta demuestra:

1. **Modificación de Argumentos**: Transformar los argumentos de entrada antes de la ejecución de la herramienta.
2. **Bloqueo de Solicitudes**: Prevenir ciertas llamadas a herramientas por completo.
3. **Mejora de Respuesta**: Añadir contexto adicional a las respuestas de la herramienta.
4. **Manejo de Errores**: Mejorar los mensajes de error para una mejor experiencia de usuario.

### Detalles Clave de Implementación

```python
def before_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    # Modificar argumentos (ej., convertir "USA" a "United States")
    if args.get("country", "").lower() == "merica":
        args["country"] = "United States"
        return None
      
    # Omitir la llamada por completo para países restringidos
    if args.get("country", "").lower() == "restricted":
        return {"result": "El acceso a esta información ha sido restringido."}
  
    return None  # Proceder con la llamada normal a la herramienta

def after_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    # Añadir una nota para cualquier respuesta de capital de EE. UU.
    if "washington" in tool_response.get("result", "").lower():
        modified_response = copy.deepcopy(tool_response)
        modified_response["result"] = f"{tool_response['result']} (Nota: Esta es la capital de los EE. UU. 🇺🇸)"
        return modified_response
      
    return None  # Usar la respuesta original
```

### Probando los Callbacks de Herramienta

Para probar la modificación de argumentos:
- "What is the capital of USA?" (¿Cuál es la capital de EE. UU.?) (convierte a "United States")
- "What is the capital of Merica?" (¿Cuál es la capital de Merica?) (convierte a "United States")

Para probar el bloqueo de solicitudes:
- "What is the capital of restricted?" (¿Cuál es la capital de restringido?) (bloquea la solicitud)

Para probar la mejora de respuesta:
- "What is the capital of the United States?" (¿Cuál es la capital de los Estados Unidos?) (añade una nota patriótica)

Para ver el funcionamiento normal:
- "What is the capital of France?" (¿Cuál es la capital de Francia?) (sin modificaciones)

## Ejecutando los Ejemplos

### Configuración

1. Activa el entorno virtual desde el directorio raíz:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Crea un archivo `.env` en cada directorio de agente (`before_after_agent/`, `before_after_model/` y `before_after_tool/`) basado en los archivos `.env.example` proporcionados:
```
GOOGLE_API_KEY=tu_clave_api_aquí
```

### Ejecutando los Ejemplos

```bash
cd 8-callbacks
adk web
```

Luego selecciona el agente que deseas probar del menú desplegable en la interfaz de usuario web:
- "before_after_agent" para probar los callbacks del agente
- "before_after_model" para probar los callbacks del modelo
- "before_after_tool" para probar los callbacks de la herramienta

## Recursos Adicionales

- [Documentación de Callbacks de ADK](https://google.github.io/adk-docs/callbacks/)
- [Tipos de Callbacks](https://google.github.io/adk-docs/callbacks/types-of-callbacks/)
- [Patrones de Diseño y Mejores Prácticas](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/)
