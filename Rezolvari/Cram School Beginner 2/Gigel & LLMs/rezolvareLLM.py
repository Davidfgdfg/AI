import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

train_data = pd.read_csv(r'Cram School Beginner 2\Gigel & LLMs\train_data.csv')
test_data = pd.read_csv(r'Cram School Beginner 2\Gigel & LLMs\test_data.csv')

train_data = train_data.drop(columns = ['date'])
test_ids = test_data['ID']
test_features = test_data['text']

mapping = {0 : 'daily_life', 1 : 'pop_culture', 2 : 'sports_&_gaming', 3 : 'arts_&_culture' , 4 : 'business_&_entrepreneurs', 5: 'science_&_technology'}
reverse_mapping = {v: k for k, v in mapping.items()}

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)

x_train = tfidf.fit_transform(train_data['text'])
y_train = train_data['label_name'].map(reverse_mapping)

X_test = tfidf.transform(test_data['text'])


model = LinearSVC(random_state=42)
model.fit(x_train,y_train)

predict_numeric = model.predict(X_test)
predict_cuvinte = [mapping[p] for p in predict_numeric]

subtask1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': test_ids,
    'answer': predict_cuvinte
})

subtask1.to_csv('Cram School Beginner 2\Gigel & LLMs\submisie.csv',index= False)





