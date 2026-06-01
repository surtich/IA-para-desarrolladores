const express = require('express');
const ollama = require('ollama').default;
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors()); // Permitir peticiones desde el frontend

app.post('/chat', async (req, res) => {
    const { mensaje } = req.body;
    try {
        const response = await ollama.chat({
            model: 'llama3.2:1b',
            messages: [{ role: 'user', content: mensaje }],
        });
        res.json({ respuesta: response.message.content });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => console.log('Servidor en http://localhost:3000'));