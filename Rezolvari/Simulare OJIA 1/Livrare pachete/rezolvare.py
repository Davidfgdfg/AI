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