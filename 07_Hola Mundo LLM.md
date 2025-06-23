### Primera aplicación basada en LLM

En esta lección vamos a integrar un LLM en una aplicación Python. Empezamos probando que podemos conversar con un modelo LLM. En estas y sucesivas lecciones, se usará el proveedor Google Gemini, cámbielo si prefiere usar otro.


```python
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
```

    2 + 2 son 4.
    


En la siguiente celda definimos la clase `Website` que permite hacer `scrapping` de una URL pasada como parámetro utilizando la librería `BeautifulSoup4`. Observe que lo que hace es obtener el `title` de la página y el texto. Para ello, elimina previamente el contenido de etiquetas irrelevantes con el método `BeautifulSoup.decompose()`.


```python
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
```

En la siguiente celda, descargamos y mostramos el código de una página de Wikipedia.


```python
# Versión en inglés de la página de Novak Djokovic
novak_url = "https://en.wikipedia.org/wiki/Novak_Djokovic"
novak_web = Website(novak_url)
print(novak_web.title)
print(novak_web.text[350:400])
```

    Novak Djokovic - Wikipedia
     editors
    learn more
    Contributions
    Talk
    Contents
    mo


Creamos una función que cree el `prompt` de  `user` para resumir páginas de Wikipedia.


```python
def user_prompt_for(website):
    user_prompt = f"Estás buscando una página con título {website.title}"
    user_prompt += "\nLos contenidos de este sitio web son los siguientes; \
                    por favor, proporciona un breve resumen de este sitio web en markdown. \
                    Si incluye noticias o anuncios, resúmelos también.\n\n"
    user_prompt += website.text
    return user_prompt
```

Creamos el `prompt` de `system`:


```python
system_prompt = "Eres un asistente que analiza el contenido de un sitio web \
                    y proporciona un resumen breve, ignorando el texto que podría estar relacionado con la navegación. \
                    No añades ningún comentario inicial ni final. \
                    Respondes en markdown. Respondes en español."
    
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]
```

Creamos la función que llama a la API para crear el resumen de la página:


```python
def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = MODEL,
        messages = messages_for(website)
    )
    return response.choices[0].message.content
```

Vemos que el resumen se hace en español a pesar de que la página está en inglés.


```python
summarize(novak_url)
```




    'Novak Djokovic es un tenista profesional serbio, nacido el 22 de mayo de 1987, considerado uno de los mejores de todos los tiempos. Ha sido número 1 del mundo por un récord de 428 semanas y ha terminado el año como número 1 en ocho ocasiones, también un récord.\n\n**Carrera:**\n*   Ha ganado 24 títulos de Grand Slam, el máximo en la historia del tenis masculino.\n*   Es el único hombre en lograr el "Triple Career Grand Slam" y el único en completar el "Career Golden Masters" en dos ocasiones.\n*   Ha ganado 100 títulos individuales en total.\n*   Medallista olímpico.\n\n**Fuera de la cancha:**\n*   Fundador de la Fundación Novak Djokovic, enfocada en apoyar a niños de comunidades desfavorecidas.\n*   Embajador de Buena Voluntad de UNICEF.\n\n**Controversias:**\n*   Su postura en contra de la vacuna COVID-19 le impidió participar en varios torneos.\n\nLa página también detalla sus rivalidades más destacadas, su estilo de juego, su equipo de trabajo, sus patrocinios, sus inversiones y sus opiniones sobre diversos temas.'



Para verlo mejor creamos una función que interprete el código Markdown:


```python
from IPython.display import Markdown, display, update_display

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))
```


```python
display_summary(novak_url)
```


Este artículo de Wikipedia trata sobre Novak Djokovic, un tenista profesional serbio nacido en 1987. Actualmente es el número 6 del mundo y ha sido el número 1 durante un récord de 428 semanas. Djokovic tiene 100 títulos individuales, incluyendo 24 títulos de Grand Slam, un récord. Es el único hombre en la historia del tenis en ser el campeón reinante de los cuatro majors a la vez en tres superficies diferentes. También ha ganado 40 Masters, siete campeonatos de fin de año y una medalla de oro olímpica.

El artículo cubre su vida temprana, carrera, rivalidades, estilo de juego, equipo y actividades fuera de la cancha, como su trabajo filantrópico a través de la Fundación Novak Djokovic y su nombramiento como Embajador de Buena Voluntad de UNICEF. También aborda sus puntos de vista sobre la dieta, la medicina y la ciencia, así como su oposición al mandato de la vacuna COVID-19. Además, el artículo destaca sus logros, estadísticas de carrera y su legado en el tenis.


**¿Qué hemos conseguido?**

La capa gratuita de la API de Google Gemini no tiene acceso a Internet, y si le pedidos que nos resuma una página de Wikipedia, no puede hacerlo. Sin embargo, si le pasamos el texto de la página, puede resumirlo. En este caso, hemos hecho `scrapping` de la página y le hemos pasado el texto a la API. La API ha sido capaz de resumirlo y devolverlo en formato Markdown.

Podemos mejorar la clase Website para que incorpore información de los `links`que tenga la página que queremos resumir:


```python
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
```

Vemos que hay muchos `links` que no son relevantes porque no tienen que ver con el instituto.


```python
url_zayas = "https://site.educa.madrid.org/ies.mariadezayas.majadahonda/"
web_zayas = Website(url_zayas)
web_zayas.links[:10]
```




    ['#ht-content',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/',
     '#',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/quienes-somos/',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/quienes-somos/',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/informacion-centro/',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/oferta/',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/oferta/curso-de-especializacion-en-panaderia-y-bolleria-artesanales/',
     'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/oferta/fp-grado-superior/']



Preparamos un `prompt` para que el modelo ignore los `links` irrelevantes. Utilizamos la técnica "one shot prompting" para instruir al modelo con un ejemplo. En este caso le decimos la estructura del JSON que queremos que genere.


```python
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
```

Creamos el `prompt` de `user` para que el modelo genere el JSON:


```python
def get_links_user_prompt(website):
    user_prompt = f"Aquí tienes la lista de enlaces del sitio web de {website.url} - "
    user_prompt += "por favor, decide cuáles de estos son enlaces web relevantes para un folleto sobre la empresa, responde con la URL completa en formato JSON. \
                    No incluyas Términos de Servicio, Privacidad ni enlaces de correo electrónico. \
                    ni de fuera del sitio web \n"
    user_prompt += "Enlaces (algunos pueden ser enlaces relativos):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

```

Usamos la IA para que decida los `links` más relevantes y los muestre en formato JSON:


```python
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
```


```python
get_links(url_zayas)
```




    {'links': [{'type': 'página acerca de',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/quienes-somos/'},
      {'type': 'información del centro',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/informacion-centro/'},
      {'type': 'oferta formativa',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/oferta/'},
      {'type': 'donde encontrarnos',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/donde-encontrarnos/'},
      {'type': 'contacto',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/contacto/'},
      {'type': 'documentos del centro',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/documentos-del-centro/'},
      {'type': 'profesorado',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/profesorado/'},
      {'type': 'aula de emprendimiento',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-emprendimiento/'},
      {'type': 'proyecto biodigestor',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/proyecto-biodigestor-del-zayas/'},
      {'type': 'aula de gestión turística',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-gestion-turistica/'},
      {'type': 'restaurantes zayas y tienda',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/restaurantes-zayas-y-tienda/'},
      {'type': 'secretaría',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/'},
      {'type': 'horarios secretaría',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/horarios-secretaria/'},
      {'type': 'matriculaciones',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/matriculaciones/'},
      {'type': 'modelos de solicitudes',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/modelos-de-solicitudes/'},
      {'type': 'FAQ secretaría',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/faq/'},
      {'type': 'alumnado',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/'},
      {'type': 'calendario escolar',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/calendario-escolar/'},
      {'type': 'calendario pruebas',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/calendario-pruebas/'},
      {'type': 'horarios de aulas',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/horarios-de-aulas/'},
      {'type': 'mentor fp',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/mentor-fp/'},
      {'type': 'aula de emprendimiento',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-emprendimiento/'},
      {'type': 'uniforme y utillaje',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/uniforme-y-utillaje/'},
      {'type': 'bolsa de empleo',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/bolsa-de-empleo/'},
      {'type': 'erasmus',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/erasmus/'},
      {'type': 'fpdual',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/fpdual/'},
      {'type': 'reservas aula restaurante zayas',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/reservas-aula-restaurante-zayas/'},
      {'type': 'mentor fp',
       'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/mentor-fp/'}]}



Creamos una función que añada los `links` al texto de la página:


```python
def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result
```

Preparamos el `prompt` de `user` para que le pida a la IA que prepare un folleto con la información que reciba:


```python
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
```

Creamos la función que llama a la API para crear el folleto y la probamos:


```python
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
```


```python
create_brochure("IES María de Zayas y Sotomayor", url_zayas)
```

    Found links: {'links': [{'type': 'página acerca de', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/quienes-somos/'}, {'type': 'información del centro', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/informacion-centro/'}, {'type': 'oferta formativa', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/oferta/'}, {'type': 'órganos de gestión', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/organos-de-gestion/'}, {'type': 'dónde encontrarnos', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/donde-encontrarnos/'}, {'type': 'contacto', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/contacto/'}, {'type': 'documentos del centro', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/documentos-del-centro/'}, {'type': 'profesorado', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/profesorado/'}, {'type': 'aula de emprendimiento', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-emprendimiento/'}, {'type': 'proyecto biodigestor', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/proyecto-biodigestor-del-zayas/'}, {'type': 'aula de gestion turística', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-gestion-turistica/'}, {'type': 'restaurantes y tienda', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/restaurantes-zayas-y-tienda/'}, {'type': 'secretaría', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/'}, {'type': 'horarios secretaría', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/horarios-secretaria/'}, {'type': 'matriculaciones', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/matriculaciones/'}, {'type': 'modelos de solicitudes', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/modelos-de-solicitudes/'}, {'type': 'FAQ', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/faq/'}, {'type': 'alumnado', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/'}, {'type': 'Bolsa de empleo', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/bolsa-de-empleo/'}]}



¡Claro! Aquí tienes un borrador de un folleto humorístico sobre el IES María de Zayas y Sotomayor:

# ¿Buscas una educación que te haga decir "¡Zas!"? ¡Ven al IES María de Zayas y Sotomayor!

**(Porque "Sotomayor" es un nombre muy largo para un título)**

¿Estás cansado de los institutos aburridos? ¿Sueñas con un lugar donde puedas aprender a hacer la mejor bollería *y* diseñar la próxima gran app? ¡No busques más! El IES María de Zayas y Sotomayor es el centro de Formación Profesional que te hará decir: "¡Esto sí que es una educación con sabor!"

## ¿Qué nos hace tan especiales?

*   **¡Tenemos más departamentos que un centro comercial!** Desde Hostelería y Turismo hasta Informática y Comunicaciones, pasando por Administración, Comercio y Servicios Socioculturales. ¡Aquí encontrarás tu vocación, o al menos, una buena excusa para no decidirte!
*   **¿Aula de emprendimiento? ¡La tenemos!** ¿Proyecto Biodigestor del Zayas? ¡También! ¿Restaurante y Tienda? ¡Por supuesto! Aquí no solo aprendes, ¡creamos un ecosistema de aprendizaje total! (Sí, ¡hasta tenemos un ecosistema!)
*   **Nuestra oferta formativa es tan variada que podrías abrir un negocio de "¿Qué quieres estudiar hoy?"**. Desde Marketing y Publicidad hasta Desarrollo de Aplicaciones Multiplataforma, ¡tenemos el ciclo formativo perfecto para ti! (Y si no lo tenemos, ¡lo inventamos!)
*   **¿Te preocupa el futuro laboral?** ¡No te preocupes! Tenemos bolsa de empleo, participamos en Erasmus+ y hasta FPDual. ¡Te daremos tantas oportunidades que tendrás que hacer malabares para gestionarlas!

## ¿Y la cultura de la empresa?

*   **Somos más que un instituto, ¡somos una familia!** (Una familia disfuncional, pero familia al fin y al cabo).
*   **Nos encanta innovar**. ¿Un biodigestor? ¿Un aula de gestión turística? ¡Nos apuntamos a todo! (Siempre que no implique madrugar demasiado).
*   **No nos tomamos demasiado en serio**. ¿Has visto nuestro nombre? ¡María de Zayas y *Sotomayor*! ¡Si eso no es una declaración de intenciones, no sé qué lo es!

## ¿A quién va dirigido este folleto?

*   **Futuros estudiantes**: Si quieres una FP que te prepare para el mundo real (y te haga reír por el camino), ¡este es tu sitio!
*   **Inversores**: Apuesta por nosotros, ¡somos el futuro de la FP! (Y tenemos un restaurante, ¡así que las reuniones serán deliciosas!).
*   **Interesados**: ¿Curioso? ¡Ven a vernos! (Pero no vengas con hambre, que el restaurante es solo para estudiantes).

## Oferta de Empleo

¡Únete a nuestro equipo! Buscamos profesores apasionados, personal de administración eficiente y, sobre todo, gente con sentido del humor. (Si sabes hacer café, ¡aún mejor!)

## En resumen:

El IES María de Zayas y Sotomayor es el centro de Formación Profesional donde aprenderás, te divertirás y, quién sabe, ¡quizás hasta cambies el mundo!

**¡No lo pienses más, ven a conocernos!** (Y trae un bocadillo, por si acaso).

¡Te esperamos! (Con los brazos abiertos y una sonrisa... y quizás una prueba sorpresa).



Podemos generar la salida en tiempo real con el parámetro `stream = True`:


```python
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
```


```python
stream_brochure("IES María de Zayas y Sotomayor", url_zayas)
```

    Found links: {'links': [{'type': 'página acerca de', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/quienes-somos/'}, {'type': 'información del centro', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/informacion-centro/'}, {'type': 'oferta formativa', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/oferta/'}, {'type': 'órganos de gestión', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/organos-de-gestion/'}, {'type': 'dónde encontrarnos', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/donde-encontrarnos/'}, {'type': 'contacto', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/contacto/'}, {'type': 'documentos del centro', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/documentos-del-centro/'}, {'type': 'profesorado', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/profesorado/'}, {'type': 'aula de emprendimiento', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-emprendimiento/'}, {'type': 'aula de gestión turística', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/aula-de-gestion-turistica/'}, {'type': 'restaurantes zayas y tienda', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/el-centro/restaurantes-zayas-y-tienda/'}, {'type': 'secretaría', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/'}, {'type': 'horarios secretaría', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/horarios-secretaria/'}, {'type': 'matriculaciones', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/matriculaciones/'}, {'type': 'modelos de solicitudes', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/modelos-de-solicitudes/'}, {'type': 'FAQ secretaría', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/secretaria/faq/'}, {'type': 'alumnado', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/'}, {'type': 'novedades', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/category/novedades/'}, {'type': 'calendario escolar', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/calendario-escolar/'}, {'type': 'calendario pruebas', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/calendario-pruebas/'}, {'type': 'horarios de aulas', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/horarios-de-aulas/'}, {'type': 'mentor FP', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/mentor-fp/'}, {'type': 'uniforme y utillaje', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/uniforme-y-utillaje/'}, {'type': 'bolsa de empleo', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/alumnado/bolsa-de-empleo/'}, {'type': 'Erasmus', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/erasmus/'}, {'type': 'FP dual', 'url': 'https://site.educa.madrid.org/ies.mariadezayas.majadahonda/index.php/fpdual/'}]}



¡Absolutamente! Aquí tienes un folleto diseñado para ser informativo y divertido sobre el IES María de Zayas y Sotomayor:

# ¡IES María de Zayas y Sotomayor: Donde el Futuro se Aprende... y Se Come Bien!

## ¿Quiénes Somos?

Somos el IES María de Zayas y Sotomayor, un centro público de Formación Profesional donde convertimos sueños en habilidades... ¡y habilidades en empleos! Con una oferta formativa tan amplia que te hará decir "¿En serio puedo estudiar eso?", somos el lugar ideal para empezar tu carrera.

## ¿Qué Ofrecemos?

Desde Marketing y Publicidad hasta Desarrollo de Aplicaciones Multiplataforma y Web, pasando por Hostelería y Turismo, ¡lo tenemos todo! ¿Te imaginas creando la próxima app viral mientras degustas un croissant recién horneado? ¡Aquí es posible!

*   **FP Grado Superior, Medio y Básico:** Para todos los gustos y niveles.
*   **Cursos de Especialización:** ¿Quieres ser el rey/reina de la panadería y bollería? ¡Tenemos el curso para ti!
*   **Aula de Emprendimiento:** ¿Tienes una idea millonaria? Te ayudamos a hacerla realidad.
*   **Restaurante y Tienda:** ¡Donde los estudiantes demuestran su talento y tú disfrutas de los resultados!

## Nuestra Cultura

En el IES María de Zayas y Sotomayor, creemos en:

*   **Aprender Haciendo:** Manos a la obra desde el primer día.
*   **Innovación Constante:** Siempre estamos a la última en tecnología y tendencias.
*   **Comunidad:** Somos una gran familia donde todos se apoyan.
*   **¡Comida Rica!:** No podemos negarlo, ¡nuestros estudiantes de Hostelería nos consienten mucho!

## ¿Por Qué Elegirnos?

*   **Instalaciones de Vanguardia:** Aulas especializadas, laboratorios de última generación y hasta un biodigestor (¡sí, has leído bien!).
*   **Profesores Apasionados:** Expertos en sus áreas que te guiarán en cada paso del camino.
*   **Bolsa de Empleo:** Te conectamos con las mejores empresas del sector.
*   **Erasmus+:** ¡Aprende y vive experiencias inolvidables en el extranjero!
*   **FPDual:** Aprende trabajando en empresas líderes.

## Novedades

*   **Jornada de Puertas Abiertas:** ¡Ven a conocernos el 25 de Junio a las 12:00! Inscríbete en nuestro enlace.

## ¡Únete a la Familia Zayas!

Si buscas una formación práctica, innovadora y con sabor, ¡el IES María de Zayas y Sotomayor es tu lugar!

**Visítanos en:** [Dirección del IES]

**Contáctanos:** [Teléfono] / [Correo Electrónico]

**Síguenos en Redes Sociales:** [Enlaces a Redes Sociales]

*IES María de Zayas y Sotomayor: Formando profesionales con talento... ¡y mucho sentido del humor!*


