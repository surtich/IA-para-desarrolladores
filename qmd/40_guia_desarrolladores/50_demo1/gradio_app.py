import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # Se requiere una clave, pero ollama la ignora
)

system_message = "Eres un asistente"

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    print("Historia::")
    print(history)
    print("Mensaje:")
    print(messages)

    stream = client.chat.completions.create(model="llama3.2:1b", messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

if __name__ == "__main__":
    gr.ChatInterface(fn=chat).launch()