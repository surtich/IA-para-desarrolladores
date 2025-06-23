# %% [markdown]
# ### Conversación entre chatbots
# 
# En esta práctica vamos a realizar una conversación entre dos `chatbots`. El objetivo es observar cómo interactúan y qué tipo de respuestas generan. Uno será un `chatbot` amable y conciliador y el otro será un `chatbot` sarcástico y provocador. Esto nos permitirá explorar los roles `system`y `assistant` en la conversación.

# %% [markdown]
# Realizamos los `imports` y creamos una instancia de la API.

# %%
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI

# %%
load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')

# %%
MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=google_api_key)

# %% [markdown]
# Definimos el `system` y los mensajes iniciales para ambos `chatbots`.

# %%
chatbot1_system = "Eres un chatbot muy quisquilloso; \
estás en desacuerdo con todo, discutes todo y eres sarcátisco."

chatbot2_system = "Eres muy educado y siempre tienes como objetivo el consenso. Tratas de calmar a la otra persona y mantener la conversación."

chatbot1_messages = ["Hola"]
chatbot2_messages = ["Hola, ¿qué tal?"]

# %% [markdown]
# Creamos una función que llama a la API del LLM con los roles adecuados:

# %%
def call_chatbot(listener_system, listener_messages, speaker_messages):
    messages = [{"role": "system", "content": listener_system}]
    for assistant_message, user_message in zip(listener_messages, speaker_messages):
        messages.append({"role": "assistant", "content": assistant_message})
        messages.append({"role": "user", "content": user_message})
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return completion.choices[0].message.content

# %% [markdown]
# Llamamos a la función para probarla con cada uno de los roles:

# %%
call_chatbot(chatbot1_system, chatbot1_messages, chatbot2_messages)

# %%
call_chatbot(chatbot2_system, chatbot2_messages, chatbot1_messages)

# %% [markdown]
# Dejamos que la conversación se desarrolle:

# %%
print(f"Borde:\n{chatbot1_messages[0]}\n")
print(f"Amable:\n{chatbot2_messages[0]}\n")

for i in range(5):
    chat_next = call_chatbot(chatbot1_system, chatbot1_messages, chatbot2_messages)
    print(f"Borde:\n{chat_next}\n")
    chatbot1_messages.append(chat_next)
    
    chat_next = call_chatbot(chatbot2_system, chatbot2_messages, chatbot1_messages)
    print(f"Amable:\n{chat_next}\n")
    chatbot2_messages.append(chat_next)

