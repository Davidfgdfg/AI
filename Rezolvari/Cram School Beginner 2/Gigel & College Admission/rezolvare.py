import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(r'Cram School Beginner 2\Gigel & College Admission\train_data.csv')
test = pd.read_csv(r'Cram School Beginner 2\Gigel & College Admission\test_data.csv')

train = train.drop(columns=['Name'])
test_ids = test['ID']
test_features = test.drop(columns=['Name', 'ID'])

feat_le = LabelEncoder()
for col in ['Gender', 'City']:
    train[col] = feat_le.fit_transform(train[col])
    test_features[col] = test_features[col].map(lambda s: feat_le.transform([s])[0] if s in feat_le.classes_ else -1)


target_le = LabelEncoder()
train['Admission Status'] = target_le.fit_transform(train['Admission Status'])

X = train.drop(columns=['ID', 'Admission Status'])
y = train['Admission Status']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

predictions_numeric = model.predict(test_features)

predictions_labels = target_le.inverse_transform(predictions_numeric)

output = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': test_ids,
    'answer': predictions_labels
})

output.to_csv(r'Cram School Beginner 2\Gigel & College Admission\submisie.csv', index=False)

