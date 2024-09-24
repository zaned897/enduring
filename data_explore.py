#%% Imports
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#%% Load data
csv_data = Path('data') / 'soil_data_12_sep_24.csv'
if not csv_data.exists():
    raise FileNotFoundError(f'{csv_data} not found')

data = pd.read_csv(csv_data)
data.head()
#%% Data filtering
features = ['pH [1:1 soil:water] <H+ ISE>', 'Organic Matter [Loss-on-Ignition] <Gravimetric> (%)',
            'Nitrate-Nitrogen [2 M KCl] <Spectrophotometric> (ppm)', 
            'Phosphorus [Mehlich 3 ICP] <ICP> (ppm)', 
            'Potassium [Mehlich 3 ICP] <ICP, AAS> (ppm)']

X = data[features]
X = X.dropna()

X_scaled = StandardScaler().fit_transform(X)

#%%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%%
# Aplicación de K-means con el número óptimo de clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Añadir los clusters al dataframe original
data['Cluster'] = clusters
#%%
from mpl_toolkits.mplot3d import Axes3D

# Si deseas utilizar PCA para reducir dimensionalidad
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_component = pca.fit_transform(X_scaled)
data['Principal Component'] = principal_component

# Visualización 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Latitude'], data['Longitude'], data['Principal Component'], 
                     c=data['Cluster'], cmap='viridis')

ax.set_xlabel('Latitud')
ax.set_ylabel('Longitud')
ax.set_zlabel('Componente Principal de Fertilidad')
plt.title('Clusters de Muestras de Suelo')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()

#%%
# Calcular estadísticas descriptivas por cluster
cluster_stats = data.groupby('Cluster')[features].mean()
print(cluster_stats)

#%%
import folium

# Crear un mapa centrado en las coordenadas promedio
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
soil_map = folium.Map(location=map_center, zoom_start=12, max_zoom=24)

# Añadir puntos al mapa
for idx, row in data.iterrows():
    folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                        radius=5,
                        color='blue' if row['Cluster'] == 0 else 'green' if row['Cluster'] == 1 else 'red',
                        fill=True).add_to(soil_map)

# Mostrar el mapa 
soil_map.save('soil_clusters.html')

#%%
!pip install folium