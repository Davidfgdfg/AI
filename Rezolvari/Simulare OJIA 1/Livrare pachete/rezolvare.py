'''Problema Prezicerii Livrării Pachetelor la Timp
În logistica modernă, livrarea la timp a pachetelor este esențială pentru satisfacția clienților și eficiența operațională. Dezvoltați un model care să prezică dacă un pachet va fi livrat la timp (on_time = 1) sau întârziat (on_time = 0).

Veți antrena modelul folosind setul de date de antrenament furnizat și apoi veți genera predicții pentru un set de date ne-etichetat.

Descrierea Setului de Date
Setul de Date de Antrenament (train_data.csv) conține următoarele coloane:

id (număr natural): Numărul care identifică livrarea - număr natural unic.
distance_km (0.5-700): Distanța de la depozit până la destinația de livrare (în kilometri). Distanțele mai scurte indică, în general, o probabilitate mai mare ca livrarea să fie la timp.
package_weight_kg (0.5-150): Greutatea pachetului (în kilograme). Pachetele mai ușoare sunt, de obicei, livrate mai rapid.
traffic_level (1-14): Un nivel al condițiilor de trafic (tip intreg, 1 - trafic redus, 14 - trafic aglomerat maxim).
on_time: O variabilă binară în care 1 indică faptul că pachetul a fost livrat la timp, iar 0 indică întârzierea.
Setul de Date pentru Predicție (test_data.csv):

Conține 50 de eșantioane cu aceleași caracteristici (id, distance_km, package_weight_kg și traffic_level) ca setul de antrenament, dar fără coloana on_time.
Modelul vostru va genera predicții pentru aceste eșantioane.
Subtask-uri
Subtask 1 (20p): mean_traffic_level - reprezentând media nivelurilor de trafic din setul de date pentru predicție, precizie de 2 decimale
Subtask 2 (20p): std_traffic_level - reprezentând deviația standard a aceluiași câmp (nivelul de trafic), cu precizie de 2 decimale
Subtask 3 (60p): on_time - cu predicțiile modelului vostru, 1 pentru livrare la timp, respectiv 0 pentru întârziere.
Rezultatul așteptat:
Un fisier csv output.csv care să includă următoarele 3 coloane:

subtaskID - reprezintă numărul subtaskului (1, 2 sau 3)
datapointID - care se referă la coloana id din test_data.csv
answer - răspunsul corespunzător datapointului pentru subtaskul respectiv
Notă: Pentru subtask-urile 1 și 2, la care se cere un singur răspuns pentru tot setul de date, afișați o singură linie a cărei datapointID să fie 1.

Trimiteți un singur csv care să conțină răspunsurile pentru toate subtask-urile pe care le-ați rezolvat. Pentru a vedea un exemplu, descărcați fișierul sample_output.csv'''


import os
print("Current working directory:", os.getcwd())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train_data = pd.read_csv(r'Rezolvari\Simulare OJIA 1\Livrare pachete\train_data.csv')
test_data = pd.read_csv(r'Rezolvari\Simulare OJIA 1\Livrare pachete\test_data.csv')

mean_traffic_level = round(test_data['traffic_level'].mean(), 2)
std_traffic_level = round(test_data['traffic_level'].std(), 2)


X = train_data.drop(columns=['id', 'on_time'])
Y = train_data['on_time']

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

predict = model.predict(x_val)
accuracy = accuracy_score(y_val, predict)
print(f'Validation Accuracy: {accuracy:.2f}')

test_id = test_data['id']
test_data = test_data.drop(columns=['id'])

predict_test = model.predict(test_data)

subtask1 = pd.DataFrame({
    'subtaskID': [1],
    'datapointID': [1],
    'answer': [mean_traffic_level]
})


subtask2 = pd.DataFrame({
    'subtaskID': [2],
    'datapointID': [1],
    'answer': [std_traffic_level]
})

subtask3 = pd.DataFrame({
    'subtaskID': 3,
    'datapointID': test_id,
    'answer': predict_test
})

output = pd.concat([subtask1, subtask2,subtask3], ignore_index=True)


output.to_csv(r'Rezolvari\Simulare OJIA 1\Livrare pachete\submission.csv', index=False)