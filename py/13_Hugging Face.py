# %% [markdown]
# ### Hugging Face
# 
# Hugging Face es un repositorio de modelos, datasets y aplicaciones generativas al que cualquier persona puede subir sus modelos. Para esta práctica debe registrarse y obtener una API Key (en Hugging Face se llama Access Token). Esta API Key hay que crearla con permisos de escritura.
# 
# ### Google Colab
# 
# Es un servicio de Google que puede alojar y ejecutar libros de Jupyter. Es una interesante combinación con Hugging Face porque permite ejecutar modelos LLM con GPU. Esta práctica debe ser ejecutada con una GPU, así que si no dispone de una, debe ejecutarla dentro de Google Colab. Es importante que esté seleccionado un procesador GPU dentro de Colab. La API key de Hugging Face debe ponerla en el apartado de secretos (🔑) de la izquierda y  pulsar "habilitar el acceso desde el cuaderno". No es una buena idea copiarla en una celda.

# %%
!pip install -q -U transformers datasets diffusers

# %%
# Imports

import torch
from google.colab import userdata
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio

# %% [markdown]
# Hugging Face tiene una interfaz de alto nivel para varios servicios populares llamada `pipeline`. Debajo se muestran algunos servicios interesantes.

# %%
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# %%
# Sentiment Analysis

classifier = pipeline("sentiment-analysis", device="cuda")
result = classifier("¡Me encatanta esta práctica!")
print(result)

# %%
# Named Entity Recognition

ner = pipeline("ner", grouped_entities=True, device="cuda")
result = ner("Leon XIV es el actual Papa de Roma")
print(result)

# %%
# Question Answering with Context

question_answerer = pipeline("question-answering", device="cuda")
result = question_answerer(question="¿Quién es el actual Papa de Roma?", context="Leon XIV es el actual Papa de Roma.")
print(result)

# %%
# Text Summarization

summarizer = pipeline("summarization", device="cuda")
text = """La biblioteca transformers de Hugging Face es una herramienta increíblemente versátil y potente para el procesamiento del lenguaje natural (PLN).
Permite a los usuarios realizar una amplia gama de tareas como clasificación de texto, reconocimiento de entidades nombradas y respuesta a preguntas, entre otras.
Es una biblioteca extremadamente popular y ampliamente utilizada por la comunidad de ciencia de datos de código abierto.
Reduce la barrera de entrada al campo al proporcionar a los científicos de datos una forma productiva y conveniente de trabajar con modelos transformadores.
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])

# %%
# Another translation, showing a model being specified
# All translation models are here: https://huggingface.co/models?pipeline_tag=translation&sort=trending

translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device="cuda")
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print(result[0]['translation_text'])

# %%
# Classification

classifier = pipeline("zero-shot-classification", device="cuda")
result = classifier("Hugging Face's Transformers library is amazing!", candidate_labels=["technology", "sports", "politics"])
print(result)

# %%
# Text Generation

generator = pipeline("text-generation", device="cuda")
result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
print(result[0]['generated_text'])

# %%
# Image Generation

image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
    ).to("cuda")

text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
image = image_gen(prompt=text).images[0]
image

# %%
# Audio Generation

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
Audio("speech.wav")


