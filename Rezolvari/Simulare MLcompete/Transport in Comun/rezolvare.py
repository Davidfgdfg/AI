import pandas as pd

df = pd.read_csv(r"Rezolvari\Simulare MLcompete\Transport in Comun\dataset.csv")
num_vehicule = df['id'].nunique()
num_tipuri_vehicule = df['vehicle_type'].nunique()
subtask_1 = pd.DataFrame({
    'subtaskID': [1],
    'Value1': [num_vehicule],
    'Value2': [num_tipuri_vehicule]
})

#subtask2

import pandas as pd
import matplotlib.pyplot as plt

vehicule_coords = df.groupby('id')[['latitude', 'longitude']].mean().reset_index()


plt.figure(figsize=(10, 8))
plt.scatter(vehicule_coords['longitude'], vehicule_coords['latitude'], s=10, c='blue')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribuția vehiculelor – posibilitatea clusterelor (orașelor)")
plt.show()


from sklearn.cluster import KMeans
import numpy as np


vehicule_coords = df.groupby('id')[['latitude', 'longitude']].mean().reset_index()


num_orase = 3

kmeans = KMeans(n_clusters=num_orase, random_state=42)
vehicule_coords['oras'] = kmeans.fit_predict(vehicule_coords[['latitude', 'longitude']])

subtask_2 = pd.DataFrame([[2, row['id'], row['oras']] for _, row in vehicule_coords.iterrows()],
                       columns=['subtaskID', 'Value1', 'Value2'])

#subtask 3

veh_tip10 = df[df['vehicle_type'] == 10].copy()


veh_tip10['timestamp'] = pd.to_datetime(veh_tip10['timestamp'])


veh_tip10['hour'] = veh_tip10['timestamp'].dt.hour
noapte = veh_tip10[(veh_tip10['hour'] >= 0) & (veh_tip10['hour'] <= 5)]

noapte = noapte.copy()
noapte.loc[:, 'lat_round'] = noapte['latitude'].round(3)
noapte.loc[:, 'lon_round'] = noapte['longitude'].round(3)
frequent_coords = noapte.groupby(['lat_round', 'lon_round']).size().reset_index(name='count')


depouri = frequent_coords.sort_values('count', ascending=False).head(3)[['lat_round', 'lon_round']].values


depouri = depouri[depouri[:,0].argsort()]

subtask_3 = pd.DataFrame([[3, lat, lon] for lat, lon in depouri], columns=['subtaskID', 'Value1', 'Value2'])


submision = pd.concat([subtask_1, subtask_2, subtask_3], ignore_index=True)
submision.to_csv('submision.csv', index=False)