import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

train = pd.read_csv(r'Rezolvari\Probleme_MLCompete\Clasificarea stărilor echipamentelor\train.csv')
test = pd.read_csv(r'Rezolvari\Probleme_MLCompete\Clasificarea stărilor echipamentelor\test.csv')

train_ids = train["SampleID"]
test_ids = test["SampleID"]

X_train = train.drop(columns=["SampleID"])
X_test = test.drop(columns=["SampleID"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
test_labels = kmeans.predict(X_test_scaled)

submission = pd.DataFrame({
    "SampleID": test_ids,
    "Label": test_labels
})

submission.to_csv(r"Rezolvari\Probleme_MLCompete\Clasificarea stărilor echipamentelor\submission.csv", index=False)
