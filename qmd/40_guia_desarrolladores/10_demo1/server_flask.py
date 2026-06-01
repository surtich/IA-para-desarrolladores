from flask import Flask, request, jsonify
import ollama

app = Flask(__name__)

# Creamos una instancia explícita del cliente
client = ollama.Client()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    mensaje = data.get('mensaje', '')
    
    try:
        # Usamos el cliente instanciado
        # Pasamos los parámetros como argumentos de palabra clave (keywords)
        response = client.chat(
            model='llama3.2:1b', 
            messages=[{'role': 'user', 'content': mensaje}]
        )
        
        return jsonify({'respuesta': response['message']['content']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000)