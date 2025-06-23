# %% [markdown]
# ### Primera aplicación basada en LLM
# 
# En esta lección vamos a integrar un LLM en una aplicación Python. Empezamos probando que podemos conversar con un modelo LLM. En estas y sucesivas lecciones, se usará el proveedor Google Gemini, cámbielo si prefiere usar otro.

# %%
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=api_key)

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "¿Cuánto son 2 + 2?"}]
)

print(response.choices[0].message.content)

# %% [markdown]
# En la siguiente celda definimos la clase `Website` que permite hacer `scrapping` de una URL pasada como parámetro utilizando la librería `BeautifulSoup4`. Observe que lo que hace es obtener el `title` de la página y el texto. Para ello, elimina previamente el contenido de etiquetas irrelevantes con el método `BeautifulSoup.decompose()`.

# %%
import requests
from bs4 import BeautifulSoup

# A class to represent a Webpage
# Code from: https://github.com/ed-donner/llm_engineering 

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# %% [markdown]
# En la siguiente celda, descargamos y mostramos el código de una página de Wikipedia.

# %%
# Versión en inglés de la página de Novak Djokovic
novak_url = "https://en.wikipedia.org/wiki/Novak_Djokovic"
novak_web = Website(novak_url)
print(novak_web.title)
print(novak_web.text[350:400])

# %% [markdown]
# Creamos una función que cree el `prompt` de  `user` para resumir páginas de Wikipedia.

# %%
def user_prompt_for(website):
    user_prompt = f"Estás buscando una página con título {website.title}"
    user_prompt += "\nLos contenidos de este sitio web son los siguientes; \
                    por favor, proporciona un breve resumen de este sitio web en markdown. \
                    Si incluye noticias o anuncios, resúmelos también.\n\n"
    user_prompt += website.text
    return user_prompt

# %% [markdown]
# Creamos el `prompt` de `system`:

# %%
system_prompt = "Eres un asistente que analiza el contenido de un sitio web \
                    y proporciona un resumen breve, ignorando el texto que podría estar relacionado con la navegación. \
                    No añades ningún comentario inicial ni final. \
                    Respondes en markdown. Respondes en español."
    
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

# %% [markdown]
# Creamos la función que llama a la API para crear el resumen de la página:

# %%
def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = MODEL,
        messages = messages_for(website)
    )
    return response.choices[0].message.content

# %% [markdown]
# Vemos que el resumen se hace en español a pesar de que la página está en inglés.

# %%
summarize(novak_url)

# %% [markdown]
# Para verlo mejor creamos una función que interprete el código Markdown:

# %%
from IPython.display import Markdown, display, update_display

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

# %%
display_summary(novak_url)

# %% [markdown]
# **¿Qué hemos conseguido?**
# 
# La capa gratuita de la API de Google Gemini no tiene acceso a Internet, y si le pedidos que nos resuma una página de Wikipedia, no puede hacerlo. Sin embargo, si le pasamos el texto de la página, puede resumirlo. En este caso, hemos hecho `scrapping` de la página y le hemos pasado el texto a la API. La API ha sido capaz de resumirlo y devolverlo en formato Markdown.

# %% [markdown]
# Podemos mejorar la clase Website para que incorpore información de los `links`que tenga la página que queremos resumir:

# %%
# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

# %% [markdown]
# Vemos que hay muchos `links` que no son relevantes porque no tienen que ver con el instituto.

# %%
url_zayas = "https://site.educa.madrid.org/ies.mariadezayas.majadahonda/"
web_zayas = Website(url_zayas)
web_zayas.links[:10]

# %% [markdown]
# Preparamos un `prompt` para que el modelo ignore los `links` irrelevantes. Utilizamos la técnica "one shot prompting" para instruir al modelo con un ejemplo. En este caso le decimos la estructura del JSON que queremos que genere.

# %%
link_system_prompt = "Se te proporciona una lista de enlaces encontrados en una página web. \
Eres capaz de decidir cuáles de los enlaces son más relevantes para incluir en un folleto sobre la empresa, \
como enlaces a una página Acerca de, o una página de la Empresa, o páginas de Carreras/Empleos.\n"
link_system_prompt += "Debes responder en JSON como en este ejemplo:"
link_system_prompt += """
{
    "links": [
        {"type": "página acerca de", "url": "https://url.completa/aquí/acerca"},
        {"type": "página de carreras", "url": "https://otra.url.completa/carreras"}
    ]
}
"""

# %% [markdown]
# Creamos el `prompt` de `user` para que el modelo genere el JSON:

# %%
def get_links_user_prompt(website):
    user_prompt = f"Aquí tienes la lista de enlaces del sitio web de {website.url} - "
    user_prompt += "por favor, decide cuáles de estos son enlaces web relevantes para un folleto sobre la empresa, responde con la URL completa en formato JSON. \
                    No incluyas Términos de Servicio, Privacidad ni enlaces de correo electrónico. \
                    ni de fuera del sitio web \n"
    user_prompt += "Enlaces (algunos pueden ser enlaces relativos):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


# %% [markdown]
# Usamos la IA para que decida los `links` más relevantes y los muestre en formato JSON:

# %%
import json

def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

# %%
get_links(url_zayas)

# %% [markdown]
# Creamos una función que añada los `links` al texto de la página:

# %%
def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

# %% [markdown]
# Preparamos el `prompt` de `user` para que le pida a la IA que prepare un folleto con la información que reciba:

# %%
system_prompt = (
    "Eres un asistente que analiza el contenido de varias páginas relevantes de un sitio web de una empresa "
    "y crea un folleto corto, humorístico, entretenido y con chistes sobre la compañía para futuros clientes, "
    "inversores e interesados. Responde en markdown.\n\n"
    "Incluye detalles de la cultura de la empresa, los clientes y las carreras/empleos si tienes la información."
)

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"Estás viendo una empresa llamada: {company_name}\n"
    user_prompt += "Aquí tienes el contenido de su página principal y otras páginas relevantes.\
                    usa esta información para crear un folleto breve de la empresa en markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt

# %% [markdown]
# Creamos la función que llama a la API para crear el folleto y la probamos:

# %%
def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))

# %%
create_brochure("IES María de Zayas y Sotomayor", url_zayas)

# %% [markdown]
# Podemos generar la salida en tiempo real con el parámetro `stream = True`:

# %%
def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)

# %%
stream_brochure("IES María de Zayas y Sotomayor", url_zayas)


