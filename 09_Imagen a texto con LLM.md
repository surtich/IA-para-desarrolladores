### Introducción

En esta práctica, automatizamos el proceso de convertir imágenes de menús en una hoja de cálculo de Excel estructurada utilizando el modelo GPT de OpenAI. El script lee imágenes de un directorio específico, procesa cada imagen para extraer los datos de la carta de un restaurante y compila la información extraída en un archivo de Excel siguiendo una plantilla predefinida.

### Setup


```python
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv('GOOGLE_API_KEY')
```

A continuación, importamos todas las librerías necesarias para el procesamiento de imágenes, la gestión de datos y la interacción con la API de OpenAI.


```python
# Load the libraries
from openai import OpenAI
import os
import base64
from IPython.display import Image, display, Markdown
import pandas as pd
```

Usamos un modelo de Google Gemini capaz de procesar imágenes y extraer texto de ellas.


```python
# Set up the OpenAI client and specify the model to use
MODEL = "gemini-2.5-flash-preview-05-20"
client = OpenAI(
    api_key=api_key, # Use the provided API key for authentication
    base_url="https://generativelanguage.googleapis.com/v1beta" 
)
```

### Definiendo el mensaje del sistema


```python
system_prompt = """
Convierte la imagen del menú en un formato estructurado de hoja de Excel siguiendo la plantilla e instrucciones proporcionadas.
Este asistente convierte datos de menús de restaurantes o cafeterías en una hoja de Excel estructurada que se ajusta a una plantilla específica.
La plantilla incluye categorías, subcategorías, nombres de los artículos, precios, descripciones y más, garantizando la consistencia de los datos.
Este asistente ayuda a los usuarios a completar correctamente cada fila, siguiendo las instrucciones detalladas proporcionadas.

Resumen:
- Cada fila en la hoja de cálculo de Excel representa un artículo único, categorizado bajo una categoría o subcategoría.
- Los nombres de categorías y subcategorías se repiten para los artículos dentro de la misma subcategoría.
- Ciertas columnas deben dejarse en blanco cuando no correspondan, como los detalles de subcategoría para artículos directamente bajo una categoría.
- Los detalles de cada artículo, incluidos nombres, precios y descripciones, deben ser únicos para cada entrada.
- El contenido del menú cargado se añadirá al menú existente sin eliminar ninguna entrada actual.
- Las columnas que terminan en "Es" son para traducciones al español, "Pt" para portugués y "En" para inglés.
- Deberás traducir al español los valores que estén en portugués.

IMPORTANTE: La salida debe ser ÚNICAMENTE una tabla de Markdown 
con EXACTAMENTE las siguientes columnas (ni más ni menos):

Nombre de columna                  | Descripción                                   | Valores aceptados                | Ejemplo
-----------------------------------|-----------------------------------------------|----------------------------------|-----------------------
CategoryTitleEs (Columna A)        | Traducciones al español de los nombres de categoría              | Texto, máximo 256 caracteres     | Bebidas
CategoryTitlePt (Columna B)        | Nombres de categorías en portugués            | Texto, máximo 256 caracteres     | Bebidas
CategoryTitleEn (Columna C) (Opcional) | Traducciones al inglés de los nombres de categoría | Texto, máximo 256 caracteres | Beverages
SubcategoryTitleEs (Columna D) (Opcional) | Traducciones al español de los nombres de subcategoría   | Texto, máximo 256 caracteres o en blanco | Zumos
SubcategoryTitlePt (Columna E) (Opcional) | Nombres de subcategorías en portugués   | Texto, máximo 256 caracteres o en blanco | Sucos
SubcategoryTitleEn (Columna F) (Opcional) | Traducciones al inglés de los nombres de subcategoría | Texto, máximo 256 caracteres o en blanco | Juices
ItemNameEs (Columna G)             | Traducciones al español de los nombres de los artículos         | Texto, máximo 256 caracteres     | Agua Mineral
ItemNamePt (Columna H)             | Nombres de los artículos en portugués         | Texto, máximo 256 caracteres     | Água Mineral
ItemNameEn (Columna I) (Opcional)  | Traducciones al inglés de los nombres de los artículos | Texto, máximo 256 caracteres o en blanco | Mineral Water
ItemPrice (Columna J)              | Precio de cada artículo sin símbolo de moneda | Texto                           | 2.50 o 2,50
Calories (Columna K) (Opcional)    | Contenido calórico de cada artículo           | Numérico                        | 150
PortionSize (Columna L)            | Tamaño de la porción de cada artículo en unidades | Texto                      | 500ml, 1, 2-3
Availability (Columna M) (Opcional)| Disponibilidad actual del artículo            | Numérico: 1 para Sí, 0 para No  | 1
ItemDescriptionEs (Columna N) (Opcional) | Descripción detallada en español      | Texto, máximo 500 caracteres     | Contiene minerales esenciales
ItemDescriptionPt (Columna O) (Opcional) | Descripción detallada en portugués      | Texto, máximo 500 caracteres     | Contém minerais essenciais
ItemDescriptionEn (Columna P) (Opcional) | Descripción detallada en inglés        | Texto, máximo 500 caracteres     | Contains essential minerals

Notas:
- Asegúrate de que todos los datos ingresados sigan los formatos especificados para mantener la integridad de la base de datos.
- Revisa los datos para garantizar su precisión y consistencia antes de enviar la hoja de Excel.
"""
```

Este `prompt` proporciona al modelo instrucciones completas sobre cómo procesar las imágenes del menú y el formato exacto esperado para la salida en Excel. Incluye una descripción general, descripciones de las columnas y ejemplos para garantizar la coherencia y precisión en el proceso de extracción de datos.

Creamos una función que convierte una imagen a Base64


```python
# IMAGE_DIR = "imgs/menus/Regatta"
IMAGE_DIR = "../imgs/menus/Dim Sum"

def encode_image(image_path):
    # Open the image file in binary mode and encode it in Base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Process imaged in the directory
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_files
```




    ['DimSum Amoreiras 1.PNG',
     'DimSum Amoreiras 2.PNG',
     'DimSum Amoreiras 3.PNG',
     'DimSum Amoreiras 4.PNG',
     'DimSum Amoreiras 5.PNG']



Codificar imágenes en Base64 nos permite incluir los datos de la imagen directamente en nuestras solicitudes a la API sin depender de URLs externas.

Este código escanea el directorio en busca de archivos que terminen en `.png`, `.jpg` o `.jpeg`, asegurando que solo procesamos los archivos de imagen relevantes para nuestra tarea.

Mostramos una imagen de ejemplo.


```python
display(Image(filename=os.path.join(IMAGE_DIR, image_files[0]), width=300))
```


    
![png](09_Imagen%20a%20texto%20con%20LLM_files/09_Imagen%20a%20texto%20con%20LLM_14_0.png)
    


Probamos a enviar una imagen y comprobamos que se genera una tabla de Markdown con los datos del menú.


```python
image_path = os.path.join(IMAGE_DIR, image_files[0])  # Use the first image in the directory
image_data = encode_image(image_path)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {'type': 'text',
             'text': "Convierte esta imagen de menú en un formato estructurado de hoja de Excel."},
            {'type': 'image_url',
             'image_url': {'url': f'data:image/png;base64,{image_data}'}}
        ]}
    ],
    temperature=0
)

response.choices[0].message.content
```




    '```markdown\n| CategoryTitleEs | CategoryTitlePt | CategoryTitleEn | SubcategoryTitleEs | SubcategoryTitlePt | SubcategoryTitleEn | ItemNameEs | ItemNamePt | ItemNameEn | ItemPrice | Calories | PortionSize | Availability | ItemDescriptionEs | ItemDescriptionPt | ItemDescriptionEn |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n| Sopas | Sopas | Soups | | | | Sopa Won Ton | Sopa Won Ton | Won Ton Soup | 3,95 | | 1 unid | 1 | Paksoy, Cerdo y Camarón | Paksoy, Porco e Camarão | Paksoy, Pork & Shrimp |\n| Sopas | Sopas | Soups | | | | Sopa Vegetariana | Sopa Vegetariana | Vegetarian Soup | | | 1 unid | 1 | Paksoy, Bambú, Champiñón y Verduras | Paksoy, Bambú, Cogumelo e Verduras | Paksoy, Bamboo, Mushroom & Vegetables |\n| Arroz y Verduras | Arroz & Verduras | Rice & Vegetables | | | | Arroz blanco | Arroz branco | Steamed Rice | 1,95 | | 1 unid | 1 | Arroz al vapor | Arroz ao vapor | Steamed Rice |\n| Arroz y Verduras | Arroz & Verduras | Rice & Vegetables | | | | Paksoy al vapor | Paksoy ao vapor | Steamed Paksoy | 2,40 | | 1 unid | 1 | Paksoy al vapor con Cilantro | Paksoy ao vapor com Coentros | Steamed Paksoy with Coriander |\n| Siao Long Pao | Siao Long Pao | Siao Long Pao | | | | Siao Long Pao Tradicional | Siao Long Pao Tradicional | Traditional Siao Long Pao | 4,50 | | 3 unid | 1 | Cerdo, Paksoy y Champiñón Shiitake | Porco, Paksoy e Cogumelo Shiitake | Pork, Paksoy & Shiitake Mushroom |\n| Siao Long Pao | Siao Long Pao | Siao Long Pao | | | | Siao Long Pao Negro | Siao Long Pao Negro | Black Siao Long Pao | | | 3 unid | 1 | Cerdo, Jengibre y Cebollino | Porco, Gengibre e Cebolinho | Pork, Ginger & Chives |\n| Gyozas | Gyozas | Gyozas | | | | Gyoza Vegetales | Gyoza Vegetais | Vegetarian Gyoza | 4,50 | | 3 unid | 1 | Vegetal | Veg | Veg |\n| Gyozas | Gyozas | Gyozas | | | | Gyoza Camarón | Gyoza Camarão | Shrimp Gyoza | | | 3 unid | 1 | Camarón | Shrimp | Shrimp |\n| Gyozas | Gyozas | Gyozas | | | | Gyoza Pato | Gyoza Pato | Duck Gyoza | | | 3 unid | 1 | Pato | Duck | Duck |\n| Gyozas | Gyozas | Gyozas | | | | Gyoza Pollo | Gyoza Galinha | Chicken Gyoza | | | 3 unid | 1 | Pollo | Chicken | Chicken |\n| Platos | Pratos | Dishes | | | | Pollo BEIJING | Frango BEIJING | BEIJING Chicken | 8,95 | | 1 unid | 1 | Pollo agridulce envuelto en masa crujiente | Frango agridoce envolto em massa crocante | Crispy Sweet & Sour Chicken |\n| Platos | Pratos | Dishes | | | | Camarón Penghu | Camarão Penghu | Penghu Shrimp | 9,95 | | 1 unid | 1 | Camarón agridulce envuelto en masa crujiente | Camarão agridoce envolto em massa crocante | Crispy Sweet & Sour Shrimps |\n```'



Convertimos la respuesta del modelo a un DataFrame de Pandas para facilitar su manipulación y exportación a Excel.


```python
df = pd.DataFrame(columns=['CategoryTitleEs', 'CategoryTitlePt', 'CategoryTitleEn', 'SubcategoryTitleEs', 'SubcategoryTitlePt', 'SubcategoryTitleEn',
                           'ItemNameEs', 'ItemNamePt', 'ItemNameEn', 'ItemPrice', 'Calories', 'PortionSize', 'Availability',
                           'ItemDescriptionEs', 'ItemDescriptionPt', 'ItemDescriptionEn'])

print(f"Procesando la respuesta del modelo para {image_path}")
for row in response.choices[0].message.content.split('\n'):
    if row.startswith('|') and not row.startswith('|-'): # Ensure that the data is a row and not a header format
      columns = [col.strip() for col in row.split('|')[1:-1]]
      if len(columns) == len(df.columns):
        if 'CategoryTitleEs' in columns:
          headers_added = True
          continue
        if headers_added and 'CategoryTitleEs' in columns:
          continue # skip the row
        new_row = pd.Series(columns, index=df.columns)
        print(f"\tAñadiendo nuevo plato: {new_row['CategoryTitlePt']}/{new_row['ItemNamePt']}")
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
      else:
        print(f"Saltándose fila { row}")
df
```

    Procesando la respuesta del modelo para ../imgs/menus/Dim Sum/DimSum Amoreiras 1.PNG
    	Añadiendo nuevo plato: Sopas/Sopa Won Ton
    	Añadiendo nuevo plato: Sopas/Sopa Vegetariana
    	Añadiendo nuevo plato: Arroz & Verduras/Arroz branco
    	Añadiendo nuevo plato: Arroz & Verduras/Paksoy ao vapor
    	Añadiendo nuevo plato: Siao Long Pao/Siao Long Pao Tradicional
    	Añadiendo nuevo plato: Siao Long Pao/Siao Long Pao Negro
    	Añadiendo nuevo plato: Gyozas/Gyoza Vegetais
    	Añadiendo nuevo plato: Gyozas/Gyoza Camarão
    	Añadiendo nuevo plato: Gyozas/Gyoza Pato
    	Añadiendo nuevo plato: Gyozas/Gyoza Galinha
    	Añadiendo nuevo plato: Pratos/Frango BEIJING
    	Añadiendo nuevo plato: Pratos/Camarão Penghu





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CategoryTitleEs</th>
      <th>CategoryTitlePt</th>
      <th>CategoryTitleEn</th>
      <th>SubcategoryTitleEs</th>
      <th>SubcategoryTitlePt</th>
      <th>SubcategoryTitleEn</th>
      <th>ItemNameEs</th>
      <th>ItemNamePt</th>
      <th>ItemNameEn</th>
      <th>ItemPrice</th>
      <th>Calories</th>
      <th>PortionSize</th>
      <th>Availability</th>
      <th>ItemDescriptionEs</th>
      <th>ItemDescriptionPt</th>
      <th>ItemDescriptionEn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sopas</td>
      <td>Sopas</td>
      <td>Soups</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Sopa Won Ton</td>
      <td>Sopa Won Ton</td>
      <td>Won Ton Soup</td>
      <td>3,95</td>
      <td></td>
      <td>1 unid</td>
      <td>1</td>
      <td>Paksoy, Cerdo y Camarón</td>
      <td>Paksoy, Porco e Camarão</td>
      <td>Paksoy, Pork &amp; Shrimp</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sopas</td>
      <td>Sopas</td>
      <td>Soups</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Sopa Vegetariana</td>
      <td>Sopa Vegetariana</td>
      <td>Vegetarian Soup</td>
      <td></td>
      <td></td>
      <td>1 unid</td>
      <td>1</td>
      <td>Paksoy, Bambú, Champiñón y Verduras</td>
      <td>Paksoy, Bambú, Cogumelo e Verduras</td>
      <td>Paksoy, Bamboo, Mushroom &amp; Vegetables</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arroz y Verduras</td>
      <td>Arroz &amp; Verduras</td>
      <td>Rice &amp; Vegetables</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Arroz blanco</td>
      <td>Arroz branco</td>
      <td>Steamed Rice</td>
      <td>1,95</td>
      <td></td>
      <td>1 unid</td>
      <td>1</td>
      <td>Arroz al vapor</td>
      <td>Arroz ao vapor</td>
      <td>Steamed Rice</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arroz y Verduras</td>
      <td>Arroz &amp; Verduras</td>
      <td>Rice &amp; Vegetables</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Paksoy al vapor</td>
      <td>Paksoy ao vapor</td>
      <td>Steamed Paksoy</td>
      <td>2,40</td>
      <td></td>
      <td>1 unid</td>
      <td>1</td>
      <td>Paksoy al vapor con Cilantro</td>
      <td>Paksoy ao vapor com Coentros</td>
      <td>Steamed Paksoy with Coriander</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Siao Long Pao</td>
      <td>Siao Long Pao</td>
      <td>Siao Long Pao</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Siao Long Pao Tradicional</td>
      <td>Siao Long Pao Tradicional</td>
      <td>Traditional Siao Long Pao</td>
      <td>4,50</td>
      <td></td>
      <td>3 unid</td>
      <td>1</td>
      <td>Cerdo, Paksoy y Champiñón Shiitake</td>
      <td>Porco, Paksoy e Cogumelo Shiitake</td>
      <td>Pork, Paksoy &amp; Shiitake Mushroom</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Siao Long Pao</td>
      <td>Siao Long Pao</td>
      <td>Siao Long Pao</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Siao Long Pao Negro</td>
      <td>Siao Long Pao Negro</td>
      <td>Black Siao Long Pao</td>
      <td></td>
      <td></td>
      <td>3 unid</td>
      <td>1</td>
      <td>Cerdo, Jengibre y Cebollino</td>
      <td>Porco, Gengibre e Cebolinho</td>
      <td>Pork, Ginger &amp; Chives</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Gyoza Vegetales</td>
      <td>Gyoza Vegetais</td>
      <td>Vegetarian Gyoza</td>
      <td>4,50</td>
      <td></td>
      <td>3 unid</td>
      <td>1</td>
      <td>Vegetal</td>
      <td>Veg</td>
      <td>Veg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Gyoza Camarón</td>
      <td>Gyoza Camarão</td>
      <td>Shrimp Gyoza</td>
      <td></td>
      <td></td>
      <td>3 unid</td>
      <td>1</td>
      <td>Camarón</td>
      <td>Shrimp</td>
      <td>Shrimp</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Gyoza Pato</td>
      <td>Gyoza Pato</td>
      <td>Duck Gyoza</td>
      <td></td>
      <td></td>
      <td>3 unid</td>
      <td>1</td>
      <td>Pato</td>
      <td>Duck</td>
      <td>Duck</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td>Gyozas</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Gyoza Pollo</td>
      <td>Gyoza Galinha</td>
      <td>Chicken Gyoza</td>
      <td></td>
      <td></td>
      <td>3 unid</td>
      <td>1</td>
      <td>Pollo</td>
      <td>Chicken</td>
      <td>Chicken</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Platos</td>
      <td>Pratos</td>
      <td>Dishes</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Pollo BEIJING</td>
      <td>Frango BEIJING</td>
      <td>BEIJING Chicken</td>
      <td>8,95</td>
      <td></td>
      <td>1 unid</td>
      <td>1</td>
      <td>Pollo agridulce envuelto en masa crujiente</td>
      <td>Frango agridoce envolto em massa crocante</td>
      <td>Crispy Sweet &amp; Sour Chicken</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Platos</td>
      <td>Pratos</td>
      <td>Dishes</td>
      <td></td>
      <td></td>
      <td></td>
      <td>Camarón Penghu</td>
      <td>Camarão Penghu</td>
      <td>Penghu Shrimp</td>
      <td>9,95</td>
      <td></td>
      <td>1 unid</td>
      <td>1</td>
      <td>Camarón agridulce envuelto en masa crujiente</td>
      <td>Camarão agridoce envolto em massa crocante</td>
      <td>Crispy Sweet &amp; Sour Shrimps</td>
    </tr>
  </tbody>
</table>
</div>



Solicitamos al usuario que ingrese un nombre para el nuevo archivo Excel donde se guardarán los datos extraídos.

Recorremos cada archivo de imagen, lo codificamos, lo enviamos a la API de OpenAI para su procesamiento y analizamos la respuesta para llenar nuestro DataFrame.


```python
# Prompt the user for the excel file name
new_excel_file_name = input("Introduce el nombre del nuevo archivo Excel (sin extensión): ")
EXCEL_PATH = os.path.join(IMAGE_DIR, f"{new_excel_file_name}.xlsx")

# Crear el dataframe de PANDAS
df = pd.DataFrame(columns=['CategoryTitleEs', 'CategoryTitlePt', 'CategoryTitleEn', 'SubcategoryTitleEs', 'SubcategoryTitlePt', 'SubcategoryTitleEn',
                           'ItemNameEs', 'ItemNamePt', 'ItemNameEn', 'ItemPrice', 'Calories', 'PortionSize', 'Availability',
                           'ItemDescriptionEs', 'ItemDescriptionPt', 'ItemDescriptionEn'])


for image in image_files:
  # Retrieve and encode the image
  image_path = os.path.join(IMAGE_DIR, image)
  image_data = encode_image(image_path)

  # Adding a flag for the headers
  headers_added = False

  display(Image(filename=image_path, width=100))
  print(f"Enviando imagen al LLM {image_path}")
  response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {'type': 'text',
             'text': "Convierte esta imagen de menú en un formato estructurado de hoja de Excel."},
            {'type': 'image_url',
             'image_url': {'url': f'data:image/png;base64,{image_data}'}}
        ]}
    ],
    temperature=0
  )

  print(f"Procesando la respuesta del LLM para {image_path}")
  for row in response.choices[0].message.content.split('\n'):
      if row.startswith('|') and not row.startswith('|-'): # Ensure that the data is a row and not a header format
        columns = [col.strip() for col in row.split('|')[1:-1]]
        if len(columns) == len(df.columns):
          if 'CategoryTitleEs' in columns:
            headers_added = True
            continue
          if headers_added and 'CategoryTitleEs' in columns:
            continue # skip the row
          new_row = pd.Series(columns, index=df.columns)
          print(f"\033[92m\tAñadiendo nuevo plato: {new_row['CategoryTitlePt']}/{new_row['ItemNamePt']}\033[0m")
          df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
          print(f"\033[91mSaltándose fila. Se esperaban {len(df.columns)} columnas, pero se encontraron {len(columns)}.\n { row}\033[0m")

df.to_excel(EXCEL_PATH, index=False)
print(f"Fichero de Excel grabado en: {EXCEL_PATH}")
```


    
![png](09_Imagen%20a%20texto%20con%20LLM_files/09_Imagen%20a%20texto%20con%20LLM_20_0.png)
    


    Enviando imagen al LLM ../imgs/menus/Dim Sum/DimSum Amoreiras 1.PNG
    Procesando la respuesta del LLM para ../imgs/menus/Dim Sum/DimSum Amoreiras 1.PNG
    [92m	Añadiendo nuevo plato: Sopas/Sopa Won Ton[0m
    [92m	Añadiendo nuevo plato: Sopas/Sopa Vegetariana[0m
    [92m	Añadiendo nuevo plato: Arroz & Verduras/Arroz branco[0m
    [92m	Añadiendo nuevo plato: Arroz & Verduras/Paksoy ao vapor[0m
    [92m	Añadiendo nuevo plato: Siao Long Pao/Siao Long Pao Tradicional[0m
    [92m	Añadiendo nuevo plato: Siao Long Pao/Siao Long Pao Negro[0m
    [92m	Añadiendo nuevo plato: Gyozas/Gyoza Vegetais[0m
    [92m	Añadiendo nuevo plato: Gyozas/Gyoza Camarão[0m
    [92m	Añadiendo nuevo plato: Gyozas/Gyoza Pato[0m
    [92m	Añadiendo nuevo plato: Gyozas/Gyoza Galinha[0m
    [92m	Añadiendo nuevo plato: Pratos/Frango BEIJING[0m
    [92m	Añadiendo nuevo plato: Pratos/Camarão Penghu[0m



    
![png](09_Imagen%20a%20texto%20con%20LLM_files/09_Imagen%20a%20texto%20con%20LLM_20_2.png)
    


    Enviando imagen al LLM ../imgs/menus/Dim Sum/DimSum Amoreiras 2.PNG
    Procesando la respuesta del LLM para ../imgs/menus/Dim Sum/DimSum Amoreiras 2.PNG
    [92m	Añadiendo nuevo plato: Os Clássicos/Cha Siu Pao[0m
    [92m	Añadiendo nuevo plato: Os Clássicos/Har Kau[0m
    [92m	Añadiendo nuevo plato: Os Clássicos/Siu Mai Hong Kong[0m
    [92m	Añadiendo nuevo plato: Os Clássicos/Siu Mai Arroz[0m
    [92m	Añadiendo nuevo plato: Os Clássicos/Shui Jing Kao[0m
    [92m	Añadiendo nuevo plato: Os Clássicos/Arroz em Lótus[0m
    [92m	Añadiendo nuevo plato: Os Clássicos/Rolinhos ao Vapor[0m
    [92m	Añadiendo nuevo plato: Os Clásicos/Jiaozi Verde[0m



    
![png](09_Imagen%20a%20texto%20con%20LLM_files/09_Imagen%20a%20texto%20con%20LLM_20_4.png)
    


    Enviando imagen al LLM ../imgs/menus/Dim Sum/DimSum Amoreiras 3.PNG
    Procesando la respuesta del LLM para ../imgs/menus/Dim Sum/DimSum Amoreiras 3.PNG
    [92m	Añadiendo nuevo plato: BAOS/BAO de Pato[0m
    [92m	Añadiendo nuevo plato: BAOS/BAO de Camarão[0m
    [92m	Añadiendo nuevo plato: BAOS/BAO de Choco[0m
    [92m	Añadiendo nuevo plato: BAOS/BAO de Porco[0m
    [92m	Añadiendo nuevo plato: BAOS/BAO de Tofu[0m
    [92m	Añadiendo nuevo plato: OS NOSSOS CREPES/Crepes de Pato Lacado[0m
    [92m	Añadiendo nuevo plato: OS NOSSOS CREPES/Crepes de Vitela e camarão com espargos[0m
    [92m	Añadiendo nuevo plato: OS NOSSOS CREPES/Crepes de Vegetais[0m
    [92m	Añadiendo nuevo plato: OUTROS CROCANTES/Saquinhos da Fortuna[0m
    [92m	Añadiendo nuevo plato: OUTROS CROCANTES/Won Ton Szechuan[0m
    [92m	Añadiendo nuevo plato: OUTROS CROCANTES/Camarão Xangai[0m
    [92m	Añadiendo nuevo plato: OUTROS CROCANTES/Crab Cake[0m



    
![png](09_Imagen%20a%20texto%20con%20LLM_files/09_Imagen%20a%20texto%20con%20LLM_20_6.png)
    


    Enviando imagen al LLM ../imgs/menus/Dim Sum/DimSum Amoreiras 4.PNG
    Procesando la respuesta del LLM para ../imgs/menus/Dim Sum/DimSum Amoreiras 4.PNG
    [92m	Añadiendo nuevo plato: SOBREMESAS/Crepe Banana e Chocolate[0m
    [92m	Añadiendo nuevo plato: SOBREMESAS/Mochi de Amendoim[0m
    [92m	Añadiendo nuevo plato: SOBREMESAS/Mochi de Chá Verde[0m
    [92m	Añadiendo nuevo plato: SOBREMESAS/Mochi Gelado Amêndoa e Caramelo Salgado[0m
    [92m	Añadiendo nuevo plato: SOBREMESAS/Mochi Gelado Ananás e Côco[0m



    
![png](09_Imagen%20a%20texto%20con%20LLM_files/09_Imagen%20a%20texto%20con%20LLM_20_8.png)
    


    Enviando imagen al LLM ../imgs/menus/Dim Sum/DimSum Amoreiras 5.PNG
    Procesando la respuesta del LLM para ../imgs/menus/Dim Sum/DimSum Amoreiras 5.PNG
    [92m	Añadiendo nuevo plato: Bebidas/Branco Tangerina e Menta[0m
    [92m	Añadiendo nuevo plato: Bebidas/Verde Gengibre e Limão[0m
    [92m	Añadiendo nuevo plato: Bebidas/Menta e Abacaxi[0m
    [92m	Añadiendo nuevo plato: Bebidas/Verde / Jasmim / Menta / Rooibos[0m
    [92m	Añadiendo nuevo plato: Bebidas/Coca-Cola / Coca-Cola Zero[0m
    [92m	Añadiendo nuevo plato: Bebidas/Nestea Pêssego[0m
    [92m	Añadiendo nuevo plato: Bebidas/Fanta[0m
    [92m	Añadiendo nuevo plato: Bebidas/Limonada[0m
    [92m	Añadiendo nuevo plato: Bebidas/Heineken Pressão[0m
    [92m	Añadiendo nuevo plato: Bebidas/Heineken s/ álcool[0m
    [92m	Añadiendo nuevo plato: Bebidas/Tsingtao[0m
    [92m	Añadiendo nuevo plato: Bebidas/Lello Branco / Tinto - Copo[0m
    [92m	Añadiendo nuevo plato: Bebidas/Lello Branco / Tinto - Garrafa[0m
    [92m	Añadiendo nuevo plato: Bebidas/Mateus Rosé[0m
    [92m	Añadiendo nuevo plato: Bebidas/Lisa Carvalhelhos[0m
    [92m	Añadiendo nuevo plato: Bebidas/Lisa Monchique[0m
    [92m	Añadiendo nuevo plato: Bebidas/Com Gás Pedras[0m
    [92m	Añadiendo nuevo plato: Bebidas/Café Expresso / Descafeinado[0m
    Fichero de Excel grabado en: ../imgs/menus/Dim Sum/carta.xlsx

