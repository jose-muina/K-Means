import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Datos simulados de clientes
estudiantes = [
 {'edad': 42, 'horas_estudio': 2, 'promedio_academico': 8.87},
 {'edad': 26, 'horas_estudio': 8, 'promedio_academico': 9.11},
 {'edad': 56, 'horas_estudio': 5, 'promedio_academico': 7.29},
 {'edad': 29, 'horas_estudio': 14, 'promedio_academico': 7.27},
 {'edad': 43, 'horas_estudio': 8, 'promedio_academico': 8.5},
 {'edad': 34, 'horas_estudio': 10, 'promedio_academico': 4.45},
 {'edad': 51, 'horas_estudio': 3, 'promedio_academico': 6.12},
 {'edad': 22, 'horas_estudio': 12, 'promedio_academico': 4.33},
 {'edad': 47, 'horas_estudio': 6, 'promedio_academico': 7.0},
 {'edad': 38, 'horas_estudio': 9, 'promedio_academico': 4.75},
 {'edad': 60, 'horas_estudio': 2, 'promedio_academico': 9.0},
 {'edad': 31, 'horas_estudio': 11, 'promedio_academico': 2.0},
 {'edad': 45, 'horas_estudio': 4, 'promedio_academico': 8.0},
 {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 3.2},
 {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
 {'edad': 36, 'horas_estudio': 7, 'promedio_academico': 5.0},
 {'edad': 40, 'horas_estudio': 6, 'promedio_academico': 6.0},
 {'edad': 24, 'horas_estudio': 15, 'promedio_academico': 3.0},
 {'edad': 55, 'horas_estudio': 3, 'promedio_academico': 9.0},
 {'edad': 33, 'horas_estudio': 10, 'promedio_academico': 3.0},
 {'edad': 48, 'horas_estudio': 4, 'promedio_academico': 8.0},
 {'edad': 28, 'horas_estudio': 12, 'promedio_academico': 2.0},
 {'edad': 52, 'horas_estudio': 5, 'promedio_academico': 7.0},
 {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
 {'edad': 46, 'horas_estudio': 6, 'promedio_academico': 6.0},
 {'edad': 35, 'horas_estudio': 9, 'promedio_academico': 4.0},
 {'edad': 41, 'horas_estudio': 7, 'promedio_academico': 5.0},
 {'edad': 23, 'horas_estudio': 14, 'promedio_academico': 1.0},
 {'edad': 53, 'horas_estudio': 3, 'promedio_academico': 8.0},
 {'edad': 32, 'horas_estudio': 10, 'promedio_academico': 3.0},
 {'edad': 49, 'horas_estudio': 4, 'promedio_academico': 8.0},
 {'edad': 25, 'horas_estudio': 13, 'promedio_academico': 2.0},
 {'edad': 54, 'horas_estudio': 5, 'promedio_academico': 7.0},
 {'edad': 37, 'horas_estudio': 8, 'promedio_academico': 4.0},
 {'edad': 39, 'horas_estudio': 7, 'promedio_academico': 5.0},
 {'edad': 21, 'horas_estudio': 15, 'promedio_academico': 9.0},
 {'edad': 58, 'horas_estudio': 2, 'promedio_academico': 9.0},
 {'edad': 44, 'horas_estudio': 6, 'promedio_academico': 6.0},
 {'edad': 29, 'horas_estudio': 12, 'promedio_academico': 2.0},
 {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
 {'edad': 34, 'horas_estudio': 9, 'promedio_academico': 3.0},
 {'edad': 42, 'horas_estudio': 7, 'promedio_academico': 6.0},
 {'edad': 26, 'horas_estudio': 14, 'promedio_academico': 3.0},
 {'edad': 57, 'horas_estudio': 3, 'promedio_academico': 9.0},
 {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
 {'edad': 47, 'horas_estudio': 4, 'promedio_academico': 7.0},
 {'edad': 31, 'horas_estudio': 10, 'promedio_academico': 3.0},
 {'edad': 45, 'horas_estudio': 6, 'promedio_academico': 6.0},
 {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 1.0},
 {'edad': 55, 'horas_estudio': 5, 'promedio_academico': 8.0}
]

# Convertir a DataFrame
df = pd.DataFrame(estudiantes)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['edad', 'horas_estudio', 'promedio_academico']])

# Aplicar K-Means con 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar los resultados en 2D (edad vs gasto)
plt.figure(figsize=(10, 6))

# Colores para los clusters
colores = ['red', 'blue', 'green', 'orange']

# Recorremos cada fila del DataFrame y graficamos
#df.loc es una forma de acceder a datos dentro de un DataFrame de pandas 
# utilizando etiquetas (nombres de filas y columnas), no posiciones numéricas.
for i in range(len(df)):
    edad = df.loc[i, 'edad']
    promedio = df.loc[i, 'promedio_academico']
    grupo = df.loc[i, 'cluster']
    plt.plot(edad, promedio, marker='o', linestyle='', color=colores[grupo])

# Etiquetas y formato
plt.xlabel('Edad')
plt.ylabel('Monto Promedio Academico')
plt.title('Segmentación de Estudiantes con K-Means')
plt.grid(True)
plt.tight_layout()
plt.show()

