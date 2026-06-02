import streamlit as st
from openai import OpenAI

# Configuración del cliente
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

st.title("Asistente con Ollama")

# Inicializar historial en el estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Eres un asistente"}]

# Mostrar mensajes previos del historial
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Capturar entrada del usuario
if prompt := st.chat_input("Escribe tu mensaje aquí..."):
    # Añadir mensaje de usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta con streaming
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama3.2:1b",
            messages=st.session_state.messages,
            stream=True
        )
        response = st.write_stream(stream)
    
    # Guardar respuesta del asistente en el historial
    st.session_state.messages.append({"role": "assistant", "content": response})