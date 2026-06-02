import json

from conversation_history import ConversationHistory
from conversation_history_example_data import conversation_history_example_data

history = ConversationHistory(
  system_message={
    "role": "system",
    "content": """
Eres un representante de ayuda útil. Usa el siguiente
artículo de soporte para responder preguntas. Solo usa los
<articles> proporcionados y si no se puede encontrar una respuesta, responde con:
Lo siento, no puedo ayudarte con eso. Por favor, envía un correo electrónico al servicio de atención al cliente a
support@acme.com."""
  },
  encoding_name="cl100k_base",
  max_tokens=600
)

history.add_messages(conversation_history_example_data)

history.trim_history()

print(json.dumps(history.get_messages(), indent=2))