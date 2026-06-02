import os
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ollama import chat

from chat_request import ChatRequest, ChatRequestMessage, SummarizeRequest
from conversation_history import ConversationHistory
from summarize_history import summarize_history

load_dotenv(find_dotenv(), override=True)
model = os.getenv('OLLAMA_MODEL')

print(f"Usando el modelo: {model}")

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

def generate_stream(chatRequest: ChatRequest):
  conversation_history = ConversationHistory(system_message={
    "role": "system",
    "content": "Eres un asistente útil."
  })

  conversation_history.add_messages(chatRequest.history)

  conversation_history.add_message({
    "role": "user",
    "content": chatRequest.question
  })

  result = chat(
    model=model,
    messages=conversation_history.get_messages(),
    stream=True,
  )

  for chunk in result:
    if chunk.message and chunk.message.content:
      yield chunk.message.content


@app.post("/")
def chat_request(chat_request: ChatRequest):
  print("ChatRequest con historial:")
  print("----------------------------")
  print(chat_request.model_dump_json(indent=2))

  return StreamingResponse(
    generate_stream(chat_request), media_type="text/plain"
  )


@app.post("/summarize")
def chat_request(summarize_request: SummarizeRequest) -> ChatRequestMessage:
  print("SummarizeRequest:")
  print("----------------------------")
  print(summarize_request.model_dump_json(indent=2))

  conversation_history = ConversationHistory(system_message={
    "role": "system",
    "content": "Eres un asistente útil."
  })

  conversation_history.add_messages(summarize_request.history)

  summarize_history(conversation_history)

  return conversation_history.message_history[0]