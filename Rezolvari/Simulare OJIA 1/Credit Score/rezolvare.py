'''
Prezentare generală
Setul de date utilizat pentru această provocare provine de la o bancă și conține informații despre utilizatori, având ca scop prezicerea dacă un utilizator este un bun platnic. Problema este una de clasificare cu 3 posible etichete asociate unui client: POOR CREDIT SCORE, codificat prin valoarea -1, STANDARD CREDIT SCORE, codificat prin valoarea 0 și GOOD CREDIT SCORE, codificat prin valoarea 1.

Prezentarea setului de date
Setul de date din fișierul train_data.csv conține următoarele coloane:

ID: cod de identificare unic în hexazecimal a fiecărei linii
Customer_ID: cod de identificare unic al clientului
Month: luna din an
Name: numele clientului
Age: vârsta clientului exprimată în ani
SSN: cod numeric personal al clientului
Occupation: ocupația clientului
Annual_Income: venitul anul al clientului
Monthly_Inhand_Salary: salariul lunar net al clientului
Num_Bank_Accounts: numărul de conturi bancare ale clientului
Num_Credit_Card: numărul de carduri de credit pe care le deține clientul
Interest_Rate: dobânda
Num_of_Loan: numărul de împrumuturi ale clientului
Type_of_Loan: o listă care conține tipurile acestor împrumuturi
Delay_from_due_date: întârzierea față de data scadentă
Num_of_Delayed_Payment: numărul de plați efectuate cu întârziere
Changed_Credit_Limit
Num_Credit_Inquiries
Credit_Mix
Outstanding_Debt
Credit_Utilization_Ratio
Credit_History_Age
Payment_of_Min_Amount
Total_EMI_per_month
Amount_invested_monthly
Payment_Behaviour
Monthly_Balance
Credit_Score: una dintre cele 3 etichete posibile asociate unui client
Setul de date din fișierul test_data.csv va conține aceleași coloane fără coloana Credit_Score.

Cerință
Subtask 1:
Câte linii de intrare sunt in fișierul train_data.csv?

Subtask 2:
Bazat pe datele fișierul train_data.csv , care este media pentru "Salariul în mână" (Monthly_Inhand_Salary) al clienților care au un Credit_Utilization_Ratio mai mare sau egal cu 25? Afișați partea întreagă inferioară a acestei medii.

Subtask 3:
Bazat pe datele fișierul train_data.csv , câte valori unice sunt înregistrate pentru atributul Month?

Subtask 4:
Bazat pe datele fișierul train_data.csv , câte valori unice ale atributului SSN care se termină în 20 există?

Subtask 5:
Construiește un model de învățare automată pentru a prezice scorul de credit pentru fiecare înregistrare din setul de date de test (test_data.csv).

Format de ieșire
Fișierul de ieșire încărcat de tip .csv trebuie să conțină 3 coloane:

subtaskID - reprezintă numărul subtaskului (1, 2, 3, 4 sau 5)
datapointID - care se referă la coloana id din test_data.csv
answer - răspunsul corespunzător datapointului pentru subtaskul respectiv
Notă: Pentru subtask-urile 1-4, la care se cere un singur răspuns pentru tot setul de date, afișați o singură linie a cărei datapointID să fie 1. Pentru subtask-ul 5, răspunsurile trebuie să fie valori din mulțimea { -1, 0, 1 } corespunzătoare etichetei atribuite clientului.

Trimiteți un singur csv care să conțină răspunsurile pentru toate subtask-urile pe care le-ați rezolvat. Pentru a vedea un exemplu, descărcați fișierul sample_output.csv

Scor
Subtask 1: 4 puncte
Subtask 2: 5 puncte
Subtask 3: 5 puncte
Subtask 4: 6 puncte
La subtask-ul 5, veți fi punctați în funcție de acuratețe (notată mai jos acc) după cum urmează:

acc < 0.4 => 0 puncte
0.4 <= acc < 0.5 => 10 puncte
0.5 <= acc < 0.6 => 25 puncte
0.6 <= acc < 0.7 => 55 puncte
0.7 <= acc < 0.75 => 65 puncte
0.75 <= acc => 80 puncte'''




import os
print("Current working directory:", os.getcwd())
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
le_month = LabelEncoder()

train_data = pd.read_csv(r'Rezolvari\Simulare OJIA 1\Credit Score\train_data.csv')
test_data = pd.read_csv(r'Rezolvari\Simulare OJIA 1\Credit Score\test_data.csv')

# Subtask 1
length = len(train_data)
print(f'Length of test data: {length}')
subtask1 = pd.DataFrame({
    'subtaskID': [1],
    'datapointID': [1],
    'answer': length
})

# Subtask 2
filtru = train_data[train_data['Credit_Utilization_Ratio']>25]
media_salariu = filtru['Monthly_Inhand_Salary'].mean()
rezultat = math.floor(media_salariu)
subtask2 = pd.DataFrame({
    'subtaskID': [2],
    'datapointID': [1],
    'answer': rezultat
})

# Subtask 3

valori_unice = train_data['Month'].nunique()
subtask3 = pd.DataFrame({
    'subtaskID': [3],
    'datapointID': [1],
    'answer': valori_unice
})

#subtask4 

ssn_filtrat = train_data[train_data["SSN"].astype(str).str.endswith("20")]
numar_unice = ssn_filtrat["SSN"].nunique()
subtask4 = pd.DataFrame({
    'subtaskID': [4],
    'datapointID': [1],
    'answer': numar_unice
})



# Subtask 5
le_month = LabelEncoder()
train_data['Month'] = le_month.fit_transform(train_data['Month'])
test_data["Month"] = le_month.transform(test_data["Month"])
IDs = test_data['ID']


num_object_cols = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                   "Changed_Credit_Limit", "Outstanding_Debt", "Credit_History_Age",
                   "Amount_invested_monthly", "Monthly_Balance"]
for col in num_object_cols:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

train_data = train_data.fillna(-1)
test_data = test_data.fillna(-1)

cat_cols = ["Occupation", "Type_of_Loan", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
for col in cat_cols:
    train_data[col], uniques = pd.factorize(train_data[col].astype(str))
    test_data[col] = pd.Categorical(test_data[col], categories=uniques).codes


train_x = train_data.drop(columns=['ID', 'Customer_ID' ,"Name", "SSN", "Type_of_Loan",'Credit_Score'])
train_y = train_data['Credit_Score']
print(train_x.dtypes)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy}')

predictions = model.predict(test_data.drop(columns=['ID', 'Customer_ID' ,"Name", "SSN", "Type_of_Loan"]))
print(test_data.dtypes)

subtask5 = pd.DataFrame({
    'subtaskID': [5] * len(IDs),
    'datapointID': IDs,
    'answer': predictions
})

output = pd.concat([subtask1,subtask2,subtask3,subtask4,subtask5], ignore_index=True)
output.to_csv(r'Rezolvari\Simulare OJIA 1\Credit Score\submission.csv', index=False)