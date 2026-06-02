import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown';
import './App.css'

const LoadingIndicator = () => (
  <div className="loading-indicator">
    <div className="dot"></div><div className="dot"></div><div className="dot"></div>
  </div>
);

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => scrollToBottom(), [messages]);

  // --- Lógica de API centralizada ---
  const callApi = async (endpoint, payload) => {
    const response = await fetch(`http://localhost:8000${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return response;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await callApi('/', { question: inputMessage, history: messages });
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      
      let assistantMessage = '';
      while (true) {
        const { done, value } = await reader.read();
        if (value) {
          assistantMessage += value;
          setMessages(prev => {
            const newMessages = [...prev];
            const last = newMessages[newMessages.length - 1];
            if (last && last.role === 'assistant') last.content = assistantMessage;
            else newMessages.push({ role: 'assistant', content: assistantMessage });
            return newMessages;
          });
        }
        if (done) break;
      }
    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error al procesar.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (isLoading || messages.length === 0) return;
    setIsLoading(true);
    try {
      const response = await callApi('/summarize', { history: messages });
      const summaryMessage = await response.json();
      setMessages([summaryMessage]);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h1>Guía del desarrollador para IA</h1>
      
      <div style={{ marginBottom: '10px' }}>
        <button onClick={handleSummarize} disabled={isLoading || messages.length === 0}>
          Resumir Conversación
        </button>
      </div>

      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="message-content">
              <ReactMarkdown>{m.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {isLoading && <div className="message assistant"><div className="message-content"><LoadingIndicator /></div></div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Escribe tu mensaje..."
          disabled={isLoading}
        />
        <button onClick={handleSendMessage} disabled={isLoading || !inputMessage.trim()}>
          {isLoading ? 'Enviando...' : 'Enviar'}
        </button>
      </div>
    </div>
  )
}

export default App