import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Datos reales de viviendas particulares habitadas por jurisdiccion (INDEC 2022)
jurisdicciones = [
    {'jurisdiccion': 'Ciudad Autonoma de Buenos Aires', 'viviendas_habitadas': 1391258},
    {'jurisdiccion': 'Buenos Aires', 'viviendas_habitadas': 5970702},
    {'jurisdiccion': 'Catamarca', 'viviendas_habitadas': 131978},
    {'jurisdiccion': 'Chaco', 'viviendas_habitadas': 368728},
    {'jurisdiccion': 'Chubut', 'viviendas_habitadas': 213317},
    {'jurisdiccion': 'Cordoba', 'viviendas_habitadas': 1378237},
    {'jurisdiccion': 'Corrientes', 'viviendas_habitadas': 370958},
    {'jurisdiccion': 'Entre Rios', 'viviendas_habitadas': 494473},
    {'jurisdiccion': 'Formosa', 'viviendas_habitadas': 194689},
    {'jurisdiccion': 'Jujuy', 'viviendas_habitadas': 238141},
    {'jurisdiccion': 'La Pampa', 'viviendas_habitadas': 140879},
    {'jurisdiccion': 'La Rioja', 'viviendas_habitadas': 124149},
    {'jurisdiccion': 'Mendoza', 'viviendas_habitadas': 639467},
    {'jurisdiccion': 'Misiones', 'viviendas_habitadas': 420101},
    {'jurisdiccion': 'Neuquen', 'viviendas_habitadas': 254545},
    {'jurisdiccion': 'Rio Negro', 'viviendas_habitadas': 276371},
    {'jurisdiccion': 'Salta', 'viviendas_habitadas': 404504},
    {'jurisdiccion': 'San Juan', 'viviendas_habitadas': 241436},
    {'jurisdiccion': 'San Luis', 'viviendas_habitadas': 182886},
    {'jurisdiccion': 'Santa Cruz', 'viviendas_habitadas': 118047},
    {'jurisdiccion': 'Santa Fe', 'viviendas_habitadas': 1273460},
    {'jurisdiccion': 'Santiago del Estero', 'viviendas_habitadas': 311361},
    {'jurisdiccion': 'Tierra del Fuego, Antartida e Islas del Atlantico Sur', 'viviendas_habitadas': 65535},
    {'jurisdiccion': 'Tucuman', 'viviendas_habitadas': 493794}
]

# Convertir a DataFrame
df = pd.DataFrame(jurisdicciones)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['viviendas_habitadas']])

# Aplicar K-Means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 3. Asignar etiquetas a los clusters
cluster_promedio_viviendas = df.groupby('cluster')['viviendas_habitadas'].mean().sort_values()
cluster_ordenado_por_tamaño = cluster_promedio_viviendas.index.tolist()

etiquetas = {
    cluster_ordenado_por_tamaño[0]: 'Pocas viviendas',
    cluster_ordenado_por_tamaño[1]: 'Cantidad normal de viviendas',
    cluster_ordenado_por_tamaño[2]: 'Muchas viviendas'
}
df['segmento'] = df['cluster'].map(etiquetas)

# Grafico
plt.figure(figsize=(12, 6))

colores = {'Pocas viviendas': 'orange', 'Cantidad normal de viviendas': 'blue', 'Muchas viviendas': 'red'}

for segmento, color in colores.items():
    subset = df[df['segmento'] == segmento]
    plt.scatter(subset['jurisdiccion'], subset['viviendas_habitadas'], color=color, label=segmento, s=80)

# Etiquetas y formato del grafico
plt.xlabel('Jurisdicciones')
plt.ylabel('Viviendas Habitadas')
plt.title('Segmentacion de Jurisdicciones Argentinas por Viviendas Habitadas con K-Means')
plt.xticks(rotation=90, ha='right') # Para que no este todo encimado
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()