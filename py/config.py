import os
import io
import numpy as np
import pandas as pd
import zipfile
import requests

# Dataset de alturas y pesos
file = '../datasets/height.csv'
df = pd.read_csv(file)

df['gender'] = df['gender'].map({0: 'Male', 1: 'Female'})
df['weight'] = df['weight'] * 0.453592
df['height'] = df['height'] * 2.54

#Dataset MNIST
train_url = "https://github.com/phoebetronic/mnist/raw/main/mnist_train.csv.zip"
test_url = "https://github.com/phoebetronic/mnist/raw/main/mnist_test.csv.zip"


def download_from_zip(url, file_output):
    if os.path.exists(file_output):
        return
    try:
        req = requests.get(url)
        req.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(req.content)) as zip_file:
            csv_name = next((name for name in zip_file.namelist() if name.endswith('.csv')), None)

            if csv_name is None:
                print("No se encontró ningún archivo CSV dentro del ZIP.")
                return

            with zip_file.open(csv_name) as archivo_csv:
                with open(file_output, 'wb') as output:
                    output.write(archivo_csv.read())

    except requests.exceptions.RequestException as e:
        print(f"Error de red: {e}")
    except zipfile.BadZipFile:
        print("El archivo descargado no es un ZIP válido.")

download_from_zip(test_url, "../datasets/mnist_test.csv")
download_from_zip(train_url, "../datasets/mnist_train.csv")