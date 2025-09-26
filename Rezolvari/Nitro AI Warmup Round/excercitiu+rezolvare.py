"""
Enunț
Implementați un model de AI pentru a prezice prețurile din jocurile de pe Steam având la dispoziție un set de antrenare train_data.csv și unul pe care trebuie să faceți doar predicția test_data.csv.

Setul de date conține următoarele câmpuri având semnificațiile:

-Name: nume aplicație
-AppID: identificatorul jocului
-Metacritic score: nota medie a jocului de pe platforma de recenzii Metacritic
-Genres: o lista separata prin virgule conținând tag-uri referitoare la tipul de joc
-Publishers: numele producătorilor
-Estimated Owners: de forma min – max, însemnând așteptările (producătorilor jocului) minime și maxime de jocuri vândute
-Negatives: numărul de dislike-uri
-Positives: numărul de like-uri
-Recommendations: câte persoane recomandă acest joc
Note despre setul de date:
Câmpul-țintă este „Price”. Date fiind celelalte feature-uri, scopul este de a prezice „Price”. Setul de date de evaluare vizibil pentru candidați nu conține acest câmp. Metrica de evaluare folosita este MAE
Câmpul Genres, fiind o listă, trebuie transformat în date numerice într-un fel sau altul pentru a putea fi folosit in algoritmii de ML.
Asemănător pentru câmpul Estimated Owners
Unele câmpuri pot fi inutile in predicția câmpului-țintă. Încercați să analizați datele și să selectați doar ce este nevoie sau ar putea explica predicția.
Cerințe
Subtask 1 (40p). "Avg owners": pentru fiecare camp din coloana "Estimated Owners" de forma min – max, extrageți media valorilor considerând pentru fiecare linie int ((min + max) / 2).

Subtask 2(60p) "Price": Implementați un model de AI și rulați predicția pentru fiecare rând din test_data.csv.

Format de ieșire
Fișierul de ieșire încărcat de tip .csv trebuie să conțină 3 coloane:

subtaskID - reprezintă numărul subtaskului (1 sau 2)
datapointID - care se referă la coloana id din test_data.csv
answer - răspunsul corespunzător datapointului pentru subtaskul respectiv
"""

import numpy as np
import pandas as pd
import csv
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

def medie_interval(interval):
    min_val, max_val = interval.split(' - ')
    min_val = int(min_val)
    max_val = int(max_val)
    return int((int(min_val) + int(max_val)) / 2)


train_data['avg_owners'] = train_data['Estimated owners'].apply(medie_interval)
test_data['avg_owners'] = test_data['Estimated owners'].apply(medie_interval)

subtask1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': test_data['AppID'].astype(int),
    'answer': test_data['avg_owners']
})

features = ['Metacritic score', 'Negative', 'Positive', 'Recommendations', 'avg_owners']
X_train = train_data[features]
y_train = train_data['Price']
X_test = test_data[features]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


preds_price = model.predict(X_test)
preds_avgowners = test_data['avg_owners']



subtask1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': test_data['AppID'],
    'answer': preds_avgowners.astype(int)
}).sort_values('datapointID')


subtask2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': test_data['AppID'],
    'answer': preds_price
}).sort_values('datapointID')



output = pd.concat([subtask1, subtask2], ignore_index=True)


output.to_csv("submission.csv", index=False, float_format='%.0f')

