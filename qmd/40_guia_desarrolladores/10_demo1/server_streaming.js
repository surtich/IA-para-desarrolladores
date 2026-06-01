const express = require("express");
const ollama = require("ollama").default;
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

app.post("/chat", async (req, res) => {
  const { mensaje } = req.body;

  // 1. Configurar los headers para streaming
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Transfer-Encoding", "chunked");

  try {
    // 2. Llamar a Ollama con stream: true
    const response = await ollama.chat({
      model: "llama3.2:1b",
      messages: [{ role: "user", content: mensaje }],
      stream: true,
    });

    // 3. Iterar sobre el stream y enviar al cliente
    for await (const part of response) {
      res.write(part.message.content);
    }

    // 4. Finalizar la respuesta
    res.end();
  } catch (error) {
    console.error(error);
    res.status(500).write(JSON.stringify({ error: error.message }));
    res.end();
  }
});

app.listen(3000, () =>
  console.log("Servidor streaming en http://localhost:3000"),
);
