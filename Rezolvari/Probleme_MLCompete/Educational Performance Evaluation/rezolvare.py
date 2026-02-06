import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv(r"Rezolvari\Probleme_MLCompete\Educational Performance Evaluation\train.csv")
test = pd.read_csv(r"Rezolvari\Probleme_MLCompete\Educational Performance Evaluation\test.csv")

#subtask 1
subtask_1_answer = test[
    (test["County"] == "ANDERSON") &
    (test["School Type"] == "Elementary")
].shape[0]

print(subtask_1_answer)


# Subtask 3 - Model

target = "Relative Performance Rating"

X = train.drop(columns=[target])
y = train[target]


categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


pipeline.fit(X, y)


test_predictions = pipeline.predict(test)






subtask_1 = pd.DataFrame({
    'subtaskID': [1],
    'datapointID': [1],
    'answer': [subtask_1_answer]
})

submission.to_csv("submission.csv", index=False)

print("submission.csv generated successfully")
