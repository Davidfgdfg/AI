import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# 1. Curățare Avansată
def super_clean(text):
    # Lowercase
    text = text.lower()
    # Elimină sursele de știri între paranteze (ex: (Reuters), (AP)) - foarte comun în dataset-uri de știri
    text = re.sub(r'\([^)]*\)', '', text)
    # Elimină caracterele speciale și cifrele
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Elimină spațiile multiple
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Încărcare date
train = pd.read_csv(r'Probleme_MLCompete\Redacția de știri\train.csv')
test = pd.read_csv(r'Probleme_MLCompete\Redacția de știri\test.csv')

train['text'] = train['text'].apply(super_clean)
test['text'] = test['text'].apply(super_clean)

# 2. Configurare Vectorizator (Ajustat pentru precizie)
# Am crescut la un amestec de 1, 2 și 3 cuvinte (trigrame)
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=200000, 
    sublinear_tf=True,
    min_df=2,
    max_df=0.5 # Elimină cuvintele care apar în mai mult de jumătate din articole
)

# 3. Model de tip Ensemble (Vot între doi experți)
# Combinăm LinearSVC (precizie mare) cu LogisticRegression (stabilitate)
clf1 = LinearSVC(C=0.2, class_weight='balanced', max_iter=3000)
clf2 = LogisticRegression(C=5.0, solver='lbfgs', max_iter=1000, class_weight='balanced')

ensemble_model = Pipeline([
    ('tfidf', tfidf),
    ('voting', VotingClassifier(
        estimators=[('svc', clf1), ('lr', clf2)],
        voting='hard' # Hard voting: dacă ambele sunt de acord, e sigur. Dacă nu, alegerea e bazată pe pondere.
    ))
])

# 4. Antrenare și Predicție
print("Antrenare Ensemble...")
ensemble_model.fit(train['text'], train['label'])

print("Predicție...")
preds = ensemble_model.predict(test['text'])

# Salvare
pd.DataFrame({'id': test['id'], 'label': preds}).to_csv('submission.csv', index=False)
print("Submisie generată!")