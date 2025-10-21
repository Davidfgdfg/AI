import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import csv
import re
import nltk
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
train_data = pd.read_csv(r'Rezolvari\Nitro NLP\Mickey si Donald\train_data.csv')
test_data = pd.read_csv(r'Rezolvari\Nitro NLP\Mickey si Donald\test_data.csv')

stop_words = set(stopwords.words('romanian'))
def clean_text(text):
    text = text.lower()                          # litere mici
    text = re.sub(r'\$NE\$', ' ', text)          # elimină $NE$
    text = re.sub(r'[^a-zăîâșț ]', ' ', text)    # elimină orice nu e literă
    text = re.sub(r'\s+', ' ', text).strip()     # spații multiple
    words = [w for w in text.split() if w not in stop_words]  # scoate stopwords
    return " ".join(words)
x_train = train_data['sample'].astype(str).apply(clean_text)
y_train = train_data['dialect']
x_test  = test_data['sample'].astype(str).apply(clean_text)


vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)
X_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

model = LinearSVC(random_state=42, max_iter=10000)
model.fit(X_train, y_train)
predictions = model.predict(x_test)

subtask1 = pd.DataFrame({
    'subtaskID':[1]*len(test_data),
    'datapointID': test_data['datapointID'],
    'answer': predictions
})

SB2_x_train = train_data['sample'].astype(str).apply(clean_text)
SB2_y_test = train_data['category']
SB2_x_train = vectorizer.fit_transform(SB2_x_train)

model1 = LinearSVC(C=2.0, max_iter=10000, random_state=42,class_weight='balanced')
model1.fit(SB2_x_train, SB2_y_test)
SB2_predictions = model1.predict(x_test)

subtask2 = pd.DataFrame({
    'subtaskID':[2]*len(test_data), 
    'datapointID': test_data['datapointID'],
    'answer': SB2_predictions
})



output = pd.concat([subtask1, subtask2], ignore_index=True)
output.to_csv(r'Rezolvari\Nitro NLP\Mickey si Donald\submission.csv', index=False)

