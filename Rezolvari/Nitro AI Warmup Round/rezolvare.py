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

