import streamlit as st
import requests

st.title("Chatbot Simple")

# Input del usuario
prompt = st.text_input("Escribe tu pregunta:")

if st.button("Enviar"):
    if prompt:
        try:
            # Llamada directa a Flask
            response = requests.post("http://localhost:3000/chat", json={"mensaje": prompt})
            
            if response.status_code == 200:
                st.write("**Respuesta:**")
                st.write(response.json().get('respuesta'))
            else:
                st.error(f"Error del servidor: {response.status_code}")
        except Exception as e:
            st.error(f"Error de conexión: {str(e)}")
    else:
        st.warning("Por favor, escribe algo primero.")