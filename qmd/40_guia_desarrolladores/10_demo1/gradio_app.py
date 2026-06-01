import gradio as gr
import requests

def consultar_flask(mensaje):
    try:
        # Llamada al servidor Flask
        response = requests.post("http://localhost:3000/chat", json={"mensaje": mensaje})
        if response.status_code == 200:
            return response.json().get('respuesta')
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# Interfaz simple
demo = gr.Interface(
    fn=consultar_flask,
    inputs=gr.Textbox(label="Tu pregunta"),
    outputs=gr.Textbox(label="Respuesta"),
    title="Chatbot Simple"
)

if __name__ == "__main__":
    demo.launch()