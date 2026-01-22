import pandas as pd


test = pd.read_csv(r'Rezolvari\Simulare MLcompete\Analiza și clasificarea riscurilor și simptomelor Parkinson\test.csv')
#subtask 1

cardio_score = (
    (test["Hypertension"] == 1).astype(int) +
    (test["Diabetes"] == 1).astype(int) +
    (test["BMI"] > 30).astype(int)
)


Subtask_1 = pd.DataFrame({
    "PatientID": test["PatientID"],
    "subtaskID": "Task1",
    "Answer": cardio_score
})

#subtask 2
lifestyle_score = (
    (test["Smoking"] == 1).astype(int) +
    (test["AlcoholConsumption"] > 2).astype(int) +
    (test["PhysicalActivity"] < 1).astype(int)
)


subtask_2 = pd.DataFrame({
    "PatientID": test["PatientID"],
    "subtaskID": "Task2",
    "Answer": lifestyle_score
})

#subtask 3
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
train_data = pd.read_csv(r'Rezolvari\Simulare MLcompete\Analiza și clasificarea riscurilor și simptomelor Parkinson\train.csv')
test_data = pd.read_csv(r'Rezolvari\Simulare MLcompete\Analiza și clasificarea riscurilor și simptomelor Parkinson\test.csv')


y = train_data["Diagnosis"]
x = train_data.drop(columns=["Diagnosis", "PatientID","EyeColor", "DoctorInCharge"])
test_ids = test_data["PatientID"]
x_test = test_data.drop(columns=["PatientID", "EyeColor", "DoctorInCharge"])
model = XGBClassifier(n_estimators=300,max_depth=6, random_state=42)
model.fit(x, y)
predictions = model.predict(x_test)


subtask_3 = pd.DataFrame({
    "PatientID": test_ids,
    "subtaskID": "Task3",
    "Answer": predictions
})
print(x_test.dtypes)
submission = pd.concat([Subtask_1, subtask_2,subtask_3], ignore_index=True)
submission.to_csv("submission.csv", index=False)