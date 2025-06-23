### Motivación
Con esta lección finalizamos la parte introductoria del curso. Entender como funciona un LLM nos puede ayudar a comprender porqué se produce esta absurda "conversación"  (Perplexity pro mayo 2025):

![chat con perplexity](../imgs/perplexity1.jpg)

En el primer intento responde al saludo pero no con los requisitos en el número de palabras y letras. En el segundo respeta los requerimientos numéricos pero contesta con una frase sin sentido. Por si no se le ha ocurrido, una respuesta muy sencilla y lógica que cumple con lo que se pide en la pregunta es:

> ¡Muy bien! ¿Y tú?

Otra posibilidad es:

> Yo no voy mal.

Y también está la que diría mi hijo:

> Yo ni tan mal.

Pues bien, seguí chateando una rato con respuestas cada vez absurdas de Perplexity hasta que se "dio por vencido" y me dijo:


![chat con perplexity](../imgs/perplexity3.jpg)



Los **Modelos de Lenguaje de Gran Escala (LLM)** representan la culminación de décadas de avances en el **Procesamiento del Lenguaje Natural (NLP)**. Estos modelos son capaces de generar texto, responder preguntas, resumir documentos, traducir idiomas, y muchas otras tareas lingüísticas con una calidad notable. Para comprender su arquitectura y funcionamiento, es esencial explorar sus raíces técnicas y conceptuales.

El procesamiento del lenguaje natural (NLP) tiene una larga historia que abarca desde métodos simbólicos y basados en reglas hasta técnicas estadísticas y, más recientemente, enfoques de aprendizaje profundo. El NLP surgió como disciplina para permitir a las máquinas **comprender, interpretar y generar lenguaje humano**.

Inicialmente, los sistemas NLP eran construidos a mano con gramáticas y reglas lingüísticas específicas. Sin embargo, estos sistemas eran frágiles y difíciles de escalar. Con la llegada de la era estadística, comenzaron a utilizarse grandes corpus de texto para extraer patrones y modelar el lenguaje con probabilidades.

Uno de los primeros avances clave fue el uso de **modelos n-grama**, donde la probabilidad de una palabra se modela en función de las anteriores:

$P(w_1, w_2, ..., w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-1}, ..., w_{i-k+1})$

Estos modelos, aunque efectivos para ciertas tareas, sufrían de problemas de *esparsidad* y una capacidad limitada para capturar dependencias a largo plazo.

**El Concepto de Token y Corpus**

Para entrenar un modelo de lenguaje, es necesario definir los elementos básicos sobre los que opera. Un **token** es una unidad mínima de texto que el modelo procesa. Dependiendo del enfoque, un token puede ser una palabra, una subpalabra o incluso un carácter. Los LLM modernos usan técnicas como *Byte-Pair Encoding* (BPE) o *WordPiece* para tokenizar texto en subunidades lingüísticas que balancean vocabulario reducido y expresividad.

Un **corpus** es simplemente una colección extensa de textos. Para que un modelo adquiera un conocimiento rico del lenguaje, debe exponerse a un corpus masivo y diverso. LLM como GPT o BERT se entrenan sobre cientos de gigabytes o incluso terabytes de texto, provenientes de libros, páginas web, artículos científicos y otras fuentes.

**Embeddings: Representaciones Vectoriales del Lenguaje**

Una explicación sencilla de lo que es un embedding se puede consultar [aquí](https://cohere.com/llmu).

Para que una red neuronal procese texto, las palabras o tokens deben representarse en un formato numérico. Los **embeddings** son vectores densos de números reales que capturan propiedades semánticas y sintácticas de las palabras. En lugar de usar representaciones dispersas como *one-hot vectors*, los embeddings permiten una codificación continua del lenguaje.

Una técnica pionera en esta área fue **Word2Vec**, que entrena vectores de palabras de manera que las palabras con contextos similares tengan representaciones vectoriales cercanas. Los embeddings pueden capturar relaciones semánticas complejas. Por ejemplo, la relación entre "rey", "hombre" y "reina", "mujer" puede representarse como:

$\text{vec}("king") - \text{vec}("man") + \text{vec}("woman") \approx \text{vec}("queen")$

Los LLM actuales aprenden sus propios embeddings como parte del proceso de entrenamiento, lo que les permite adaptar las representaciones vectoriales a la tarea y arquitectura específicas.

**Redes Neuronales Recurrentes y sus Limitaciones**

Los primeros intentos de aplicar técnicas NLP en redes neuronales fueron las Redes Neuronales Recurrentes (RNN).
Una Red Neuronal Recurrente (RNN) es un tipo de arquitectura de red neuronal artificial diseñada específicamente para procesar datos secuenciales. A diferencia de las redes neuronales feedforward tradicionales, donde la información fluye en una única dirección desde la entrada hasta la salida, las RNN incorporan conexiones recurrentes, lo que les permite mantener un estado interno o memoria de la información procesada previamente en la secuencia. Esta capacidad de "recordar" información pasada es fundamental para modelar datos donde el orden y las dependencias temporales son importantes, como el lenguaje natural, series de tiempo, secuencias de ADN o señales de audio y video.

Imagina una secuencia de palabras en una frase. Para entender el significado de una palabra, a menudo necesitamos considerar las palabras que la preceden. Una RNN simula esta capacidad manteniendo una unidad de estado oculto (hidden state) que se actualiza a medida que procesa cada elemento de la secuencia. Este estado oculto actúa como una memoria que encapsula información relevante del historial de la secuencia hasta el punto actual. Sin embargo, las RNN tienen dificultades para capturar dependencias a largo plazo.



**El Mecanismo de Atención y el Transformer**

El paradigma del procesamiento secuencial experimentó una revolución con la publicación de "Attention is All You Need" (Vaswani et al., 2017), artículo seminal que introdujo la arquitectura Transformer. Rompiendo con el procesamiento recursivo tradicional, el Transformer permitió a cada token para interactuar directamente con todos los demás a través de su innovador mecanismo de atención.

Por otra parte, el **Transformer** es como una arquitectura especial de una red neuronal que utiliza este mecanismo de atención de una manera muy inteligente y poderosa. Imagina que antes, las computadoras leían las frases o las secuencias de información palabra por palabra, como si estuvieran leyendo un libro en voz alta, una página a la vez. Esto hacía que fuera difícil recordar cosas que estaban muy lejos en la secuencia.

El Transformer es diferente. Es como si pudiera leer toda la página de golpe y, al mismo tiempo, saber qué palabras son más importantes para entender el significado general y las relaciones entre ellas.

Piensa en un equipo de personas trabajando juntas para entender una frase larga:

Antes (con otras redes neuronales): Una persona leía la frase en voz alta, palabra por palabra, y trataba de recordar todo lo que había escuchado para entender el significado al final. Esto era lento y la persona podía olvidar detalles importantes del principio.

Con el Transformer: Imagina que tienes varias personas (las "cabezas de atención") que leen la frase al mismo tiempo. Cada persona se enfoca en diferentes partes de la frase y en cómo se relacionan esas partes entre sí. Luego, comparten sus "atenciones" para obtener una comprensión completa de la frase.

Existen tres tipos principales de modelos basados en transformers, cada uno con aplicaciones distintas. Los modelos de lenguaje autorregresivos, como la serie GPT de OpenAI, predicen el siguiente token en una secuencia basándose únicamente en los tokens precedentes. Estos modelos son excelentes para tareas que implican la generación de texto.

Los modelos de lenguaje de auto codificación, como BERT de Google, adoptan un enfoque diferente al predecir tokens basándose en el contexto circundante, lo que los hace bidireccionales. Esta naturaleza bidireccional permite que estos modelos destaquen en tareas como la clasificación de texto, el análisis de sentimientos y el reconocimiento de entidades nombradas. Comprenden eficazmente el contexto completo de una oración. Esto ayuda a mejorar la precisión en la comprensión del significado y la intención detrás del texto.

El tercer tipo de modelo transformador combina técnicas autorregresivas y de auto codificación. Un ejemplo de esto es el modelo T5, que puede ajustarse para diversas tareas, aprovechando las fortalezas de ambos enfoques para lograr un rendimiento de vanguardia en una variedad de aplicaciones de NPL.


### Estado del arte (mayo 2025)

En el momento de escribir estas líneas, parece que un LLM comprende nuestras palabras. Incluso hay personas que los prefieran a sus amigos para mantener conversaciones profundas. Si bien me parece inquietante equiparar una máquina con un ser humano, es crucial recordar que los LLM no procesan ni generan información como nosotros.

Para ilustrar la verdadera capacidad de comprensión de un LLM, he realizado un sencillo ejercicio que cualquier persona resolvería sin esfuerzo. Consideremos la siguiente frase incompleta:

> Mañana voy a llevar el coche al _________.

La respuesta intuitiva, y la predicha unánimemente por varios modelos de lenguaje avanzados en mayo de 2025, es:

> Mañana voy a llevar el coche al taller.

Sin embargo, la fragilidad de su comprensión contextual se evidencia al insertar la misma frase en un escenario ligeramente más complejo:

> Estoy harto de que mi coche no pare de tener averías. Ya he visto el modelo de coche que me quiero comprar. Mañana voy a llevar el coche al _________.

Aquí, el contexto se enriquece con un coche problemático y la intención de adquirir uno nuevo. La frase a completar se refiere claramente al primer vehículo. Un modelo que solo predice la siguiente palabra basándose en las adyacentes, como las arquitecturas pre-Transformer, inevitablemente fallará al capturar esta implicación contextual. El mecanismo de atención, diseñado para identificar relaciones significativas entre palabras distantes, debería, en teoría, permitir al modelo "entender" la conexión entre el hartazgo con las averías y la decisión de comprar un coche nuevo, infiriendo así el destino más probable del vehículo averiado.

Mi respuesta intuitiva en este contexto sería:

> Mañana voy a llevar el coche al desguace.

Sorprendentemente, al someter esta misma prueba a los modelos de lenguaje actuales (mayo de 2025), ninguno ofreció esta predicción. Las respuestas variaron entre el predecible "taller" y el, en este contexto, igualmente desacertado "concesionario". Esta discrepancia entre la comprensión humana, que procesa el contexto de forma holística, y la respuesta de los LLM, incluso con la sofisticación del mecanismo de atención, subraya una limitación fundamental. A pesar de su habilidad para identificar palabras clave, la inferencia contextual profunda, que implica comprender intenciones y relaciones implícitas a través de múltiples oraciones, sigue siendo un desafío significativo.

Este ejemplo sirve como un recordatorio crucial de la naturaleza y los límites de los LLMs. Comprender sus mecanismos internos y sus inherentes limitaciones es esencial para una aplicación responsable y realista de estas tecnologías, evitando la ilusión de una comprensión semántica completa donde aún existen notables carencias.

### ¿Escribir como Cervantes?

En esta sección vamos a entrenar un Red Neuronal Recurrente (RNN) para que escriba como Cervantes. Para ello, vamos a utilizar como corpus su obra más famosa y entrenar una RNN para que genere texto similar al estilo del autor. No se deben esperar resultados espectaculares, ya que tanto la red elegida como el limitado corpus no son los más adecuados para este tipo de tareas. Sin embargo, es un buen ejercicio para entender cómo funcionan las RNN y cómo pueden ser utilizadas para generar texto. Para ejecutar esta parte del notebook es conveniente hacerlo en Google Colab y con una GPU.

1. Preprocesamiento del corpus
Descarga y normalización del texto:


```python
# ---------- 1. Setup and Data Preprocessing ----------
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Download and preprocess text
url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
text = requests.get(url).text

# Extract main content
start = text.find("Primera parte del ingenioso hidalgo")
end = text.find("Fin de la obra")
clean_text = text[start:end].lower().replace('\n', ' ').replace('\r', ' ')

```

    Using device: cuda
    



```python
# longitud del texto
print(len(clean_text))
```

    2127255



```python
# limitamos el texto a 100000 caracteres para acelerar el entrenamiento
clean_text = clean_text[:100000]
```

Hacemos una tokenización sencilla a nivel de caracteres


```python
# ---------- 2. Character Tokenization ----------
# Create vocabulary
chars = sorted(list(set(clean_text)))
char_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_char = {i:c for i, c in enumerate(chars)}

# Create training sequences
seq_length = 100
X = []
y = []
for i in range(0, len(clean_text) - seq_length, 1):
    X_seq = clean_text[i:i+seq_length]
    y_seq = clean_text[i+1:i+seq_length+1]  # Shifted sequence
    X.append([char_to_idx[c] for c in X_seq])
    y.append([char_to_idx[c] for c in y_seq])

```

Definimos el modelo:


```python
# ---------- 3. Model Architecture (LSTM) ----------
class CervantesLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                          num_layers=3, dropout=0.3,
                          batch_first=False)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

# Initialize model
model = CervantesLSTM(len(chars)).to(device)
print(model)

```

    CervantesLSTM(
      (embedding): Embedding(48, 64)
      (lstm): LSTM(64, 128, num_layers=3, dropout=0.3)
      (fc): Linear(in_features=128, out_features=48, bias=True)
    )


Realizamos el entrenamiento:


```python
# ---------- 4. Training Configuration ----------
# Convert data to tensors and move to device
X_tensor = torch.LongTensor(X).to(device)
y_tensor = torch.LongTensor(y).to(device)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# ---------- 5. Training Loop with GPU Support ----------
for epoch in range(num_epochs):
    hidden = None
    total_loss = 0

    for batch_X, batch_y in dataloader:
        model.zero_grad()

        # Prepare inputs and targets
        hidden = None
        inputs = batch_X.transpose(0, 1).contiguous()  # (seq_len, batch)
        targets = batch_y.transpose(0, 1).contiguous().view(-1)  # Flatten

        # Forward pass
        logits, hidden = model(inputs, hidden)

        # Detach hidden states
        hidden = tuple(h.detach() for h in hidden)

        # Calculate loss
        loss = criterion(logits.view(-1, len(chars)), targets)
        total_loss += loss.item()

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    # Print epoch statistics
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

```

    Epoch [1/10] | Loss: 2.5098
    Epoch [2/10] | Loss: 1.8651
    Epoch [3/10] | Loss: 1.6350
    Epoch [4/10] | Loss: 1.4905
    Epoch [5/10] | Loss: 1.3884
    Epoch [6/10] | Loss: 1.3080
    Epoch [7/10] | Loss: 1.2421
    Epoch [8/10] | Loss: 1.1875
    Epoch [9/10] | Loss: 1.1419
    Epoch [10/10] | Loss: 1.1027


Generamos texto:


```python
# ---------- 6. Text Generation Function ----------
def generate_text(model, seed_str, length=1000, temp=0.8):
    model.eval()
    generated = [char_to_idx[c] for c in seed_str]
    input_seq = torch.LongTensor(generated).unsqueeze(1).to(device)
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)

            # Focus on last character
            logits = logits[-1] / temp
            probs = nn.functional.softmax(logits, dim=-1)

            # Sample from distribution
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)

            # Update input
            input_seq = torch.LongTensor([next_idx]).unsqueeze(1).to(device)

    return ''.join([idx_to_char[i] for i in generated])

# Example generation
print("\nGenerated Text:")
print(generate_text(model, "en un lugar de la mancha", temp=0.7))
```

    
    Generated Text:
    en un lugar de la mancha, porque todos los  cuando había determinó de parte de los reprosenias y caballero andante, teníanza.    — aquellos que aquí le respondió:    — todos sus sin hizose de llamar lo que le daba, roddibanelica, por los ningunas que entre manhas, y que entre porque ya no lo había pensamiento alguno.    — si vuestra merced se le camino, por quien descudero —respondió don quijote—, que  de cuando estabas en los vejos de vuestra merced de verse podría en jusper en la tierra.    — aquí no caballero aventuras; y, como no haría que se le diera  de  las parezes, que en la historia de la  riguras, prosi la videras y salidan de los arrieros,  cuando el  caballero amigo de la mancha,  que éste es dios por  que estaba en la manera de la gola, que no me  había dado con él en ellas por las alcondias contencias, que se hallaron primeros recebir un jolguna de la   las pasar al asa de al cabo de caballería, de mi merecido en el merced, caballero a su ama y podía a  hacer de la mancha,  que aquí y no se le  


El texto generado es un sin sentido. Pero si lo analizamos un poco podemos ver que hay muchas palabras que existen en castellano. Este resultado ya en sí es interesante. Debemos tener en cuenta que hemos tokenizado en base a caracteres. Es decir, que la red predice un carácter a partir de los anteriores. Así que no es un ejercicio trivial que el modelo sea capaz de generar una palabra completa. Además, el modelo es capaz de generar los espacios entre palabras y el final de oración.
