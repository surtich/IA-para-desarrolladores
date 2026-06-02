import json
import os
from dotenv import load_dotenv

from ollama import chat

from conversation_history import ConversationHistory
from conversation_history_example_data import conversation_history_example_data
from summarize_history import summarize_history

load_dotenv(override=True)
model = os.getenv("OLLAMA_MODEL")

history = ConversationHistory(
  {'role': 'system', 'content': """Eres un representante de soporte de IA útil.
Utiliza el siguiente <article> de soporte o cualquier historial de conversación proporcionado para responder preguntas.
Solo usa el <article> proporcionado o el historial de conversación y, si no se puede encontrar una respuesta, responde con:  

"Lo siento, no puedo ayudarte con eso. Por favor, envía un correo electrónico al servicio de atención al cliente a support@acme.com."   
"""})

history.add_messages(conversation_history_example_data)

print()
print("Historial antes del resumen:")
print("----------------------------------------------------")
print(json.dumps(history.get_messages(), indent=2))
print(history.count_tokens())

summarize_history(history, max_words=300)

print()
print("Historial después del resumen:")
print("----------------------------------------------------")
print(json.dumps(history.get_messages(), indent=2))

history.add_message({
  'role': 'user',
  'content': '¿Qué carpeta debo revisar para el correo de restablecimiento?'
})

response = chat(
  model=model,
  messages=history.get_messages(),
  stream=False,
  options={
    'temperature': 0
  }
)

print()
print("Respuesta a la pregunta de seguimiento después del resumen:")
print("----------------------------------------------------")
print(response.message.content)