import pandas as pd
from scipy.stats import ranksums
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import os

instancias = ['F1','F5','F8','F10','F11']


for instancia in instancias:
    # Crear un diccionario para almacenar los datos de cada columna de "MH"
    data = pd.read_csv('./Resultados/fitness_'+instancia+'.csv')
    mh_data = {}

# Iterar sobre los valores únicos de la columna "MH"
    for mh_value in data['MH'].unique():
        # Filtrar el DataFrame original para cada valor único de "MH"
        mh_subset = data[data['MH'] == mh_value]
        
        # Almacenar los datos en el diccionario usando el valor de "MH" como clave
        mh_data[mh_value] = mh_subset['FITNESS'].tolist()

    # Crear un nuevo DataFrame con los datos reducidos
    df = pd.DataFrame.from_dict(mh_data)

    resultados = pd.DataFrame(columns=["MH Target", "MH comparar", "Valor U", "Valor p", "Resultado"])

    target_column = "GO"

    # Realizar comparaciones y guardar los resultados en el nuevo DataFrame
    for column in df.columns:
        if column != target_column:
            stat, p = ranksums(df[target_column], df[column])
            alpha = 0.05
            resultado = "Se rechaza" if p < alpha else "No se rechaza"
        
            resultados = pd.concat([resultados, pd.DataFrame({
                "MH Target": [target_column],
                "MH comparar": [column],
                "Valor U": [stat],
                "Valor p": [p],
                "Resultado": [resultado]
            })], ignore_index=True)

# Guardar los resultados en un archivo CSV
    resultados.to_csv('./Resultados/TestHipotesis/Wilcoxon.Ranksum_'+instancia+'.csv', index=False)

    print(resultados)
    del data, df, resultados, target_column, stat, p, alpha, resultado
    print("Resultados guardados en resultados_comparacion.csv")

# Ruta al directorio que contiene los archivos CSV
directorio_csv = './Resultados/TestHipotesis/'

# Obtiene la lista de archivos CSV en el directorio
archivos_csv = [archivo for archivo in os.listdir(directorio_csv) if archivo.endswith('.csv')]

# Itera sobre cada archivo CSV y crea un PDF correspondiente
for archivo_csv in archivos_csv:
    # Lee el archivo CSV en un DataFrame de pandas
    data = pd.read_csv(os.path.join(directorio_csv, archivo_csv))

    # Crea un objeto SimpleDocTemplate para el PDF
    pdf_filename = archivo_csv.replace('.csv', '.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

    # Convierte los datos del DataFrame en una lista de listas
    data_table = [list(data.columns)]  # Encabezados de columna
    data_table.extend(data.values.tolist())  # Filas de datos

    # Crea una tabla con reportlab
    table = Table(data_table)

    # Aplica estilo a la tabla (opcional)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        # Agrega aquí más estilos si es necesario
    ])
    table.setStyle(style)

    # Construye el PDF
    elements = [table]
    doc.build(elements)

    print(f'El archivo CSV "{archivo_csv}" se ha convertido en "{pdf_filename}"')





