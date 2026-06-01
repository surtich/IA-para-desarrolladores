from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI()

# Creamos la instancia del cliente
client = ollama.Client()

# Definimos el esquema de la solicitud usando Pydantic
class ChatRequest(BaseModel):
    mensaje: str

@app.post('/chat')
async def chat(request: ChatRequest):
    try:
        # Usamos el cliente instanciado
        response = client.chat(
            model='llama3.2:1b', 
            messages=[{'role': 'user', 'content': request.mensaje}]
        )
        
        return {'respuesta': response['message']['content']}
    except Exception as e:
        # FastAPI maneja los errores HTTP de forma limpia con HTTPException
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    # FastAPI requiere un servidor ASGI como uvicorn
    uvicorn.run(app, host='127.0.0.1', port=3000)