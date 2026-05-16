import pandas as pd

people_df = pd.DataFrame({
    'nombre': ['Leo', 'Sara', 'Marta', 'Elena', 'Javier'],
    'horas_estudio': [0, 5, 4, 6, 7],
    'notas': [2, 4, 6, 8, 7],
    'sexo': ['M', 'F', 'F', 'F', 'M']
})

people_df.set_index('nombre', inplace=True)