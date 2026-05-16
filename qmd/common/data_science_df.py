import pandas as pd

people_df = pd.DataFrame({
    'nombre': ['Leo', 'Sara', 'Javier', 'Elena', 'Marta'],
    'horas_estudio': [0, 5, 4, 6, 7],
    'notas': [2, 4, 6, 8, 7],
    'sexo': ['M', 'F', 'M', 'F', 'F']
})

people_df.set_index('nombre', inplace=True)