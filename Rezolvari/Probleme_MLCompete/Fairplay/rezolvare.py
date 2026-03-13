import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

df = pd.read_csv(r"Probleme_MLCompete\Fairplay\train.csv")
test = pd.read_csv(r"Probleme_MLCompete\Fairplay\test.csv")


get_set = set()

team_styles_series = df['TeamStyles'].str.replace('[', '', regex=False).str.replace(']', '', regex=False).str.replace("'", "", regex=False)
for row in team_styles_series.str.split(', '):
    for style in row:
        get_set.add(style)


agresive_cols = ['AggressiveTackler', 'RiskTaker', 'HighPressure', 'ChaosInducer']

def preprocess_data(data):

    s = data['TeamStyles'].str.replace('[', '', regex=False).str.replace(']', '', regex=False).str.replace("'", "", regex=False)
    

    for style in get_set:
        data[style] = s.apply(lambda x: 1 if style in x else 0)
    

    present_agresive = [c for c in agresive_cols if c in get_set]
    agresive_count = data[present_agresive].sum(axis=1)
    total_styles = s.str.split(', ').str.len().replace(0, 1)
    
    data['StyleAggressionScore'] = (agresive_count / total_styles).round(2)
    return data

df = preprocess_data(df)
test = preprocess_data(test)


features = ['MatchWeek', 'Goals', 'Shots', 'Corners', 'YellowCards', 'RedCards', 'StyleAggressionScore'] + list(get_set)
cat_features = ['HomeTeam', 'AwayTeam']

X = df[features + cat_features]
y = df['chaos_label']


model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.025,
    depth=6,
    eval_metric='F1',
    random_seed=42,
    verbose=100
)


model.fit(X, y, cat_features=cat_features)

test_preds = model.predict(test[features + cat_features]).astype(int)


chelsea_matches = ((test['HomeTeam'] == 'Chelsea') | (test['AwayTeam'] == 'Chelsea')).sum()
sub1 = pd.DataFrame({
    'subtaskID': [1],
    'datapointID': ['1'],
    'answer': [int(chelsea_matches)]
})


sub2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': test['MatchID'].astype(str),
    'answer': test['StyleAggressionScore']
})


sub3 = pd.DataFrame({
    'subtaskID': 3,
    'datapointID': test['MatchID'].astype(str),
    'answer': test_preds
})


submission = pd.concat([sub1, sub2, sub3], ignore_index=True)
submission.to_csv("submission.csv", index=False)

print("Fișier generat: submission.csv cu", len(submission), "rânduri.")