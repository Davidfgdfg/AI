import pandas as pd

train_df = pd.read_csv(r'Locala_nitro_2026\pb_1_location\train_data.csv')
test_df = pd.read_csv(r'Locala_nitro_2026\pb_1_location\train_data.csv')
# --- SUBTASK 1 ---
test_df['Nutrient_Index'] = (0.4 * test_df['Nitrogen'] + 0.3 * test_df['Phosphorus'] + 0.3 * test_df['Potassium'])


res1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': test_df['ID'],
    'answer': test_df['Nutrient_Index']
})

print(res1.head())

# --- SUBTASK 2 ---

def classify_ph(ph_value):
    if ph_value < 6.0:
        return 'Acid'
    elif 6.0 <= ph_value <= 7.5:
        return 'Neutral'
    else:
        return 'Alkaline'

test_df['pH_Category'] = test_df['pH'].apply(classify_ph)

res2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': test_df['ID'],
    'answer': test_df['pH_Category']
})

print(res2.head())

# --- SUBTASK 3 ---

median_train = train_df['Moisture'].median()

def compare_moisture(value, median):
    if pd.isna(value):
        return 0
    

    return 1 if value > median else 0


test_df['Moisture_Comparison'] = test_df['Moisture'].apply(lambda x: compare_moisture(x, median_train))


res3 = pd.DataFrame({
    'subtaskID': 3,
    'datapointID': test_df['ID'],
    'answer': test_df['Moisture_Comparison']
})

print(f"Mediana calculată din train: {median_train}")


# --- SUBTASK 4 ---

soil_counts = train_df['Soil_Type'].value_counts()

test_df['Soil_Type_Count'] = test_df['Soil_Type'].map(soil_counts).fillna(0).astype(int)

res4 = pd.DataFrame({
    'subtaskID': 4,
    'datapointID': test_df['ID'],
    'answer': test_df['Soil_Type_Count']
})


# --- SUBTASK 5 ---

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


for df in [train_df, test_df]:
    df['Nutrient_Index'] = 0.4 * df['Nitrogen'] + 0.3 * df['Phosphorus'] + 0.3 * df['Potassium']

    df['Extreme_pH'] = ((df['pH'] < 5.5) | (df['pH'] > 8.5)).astype(int)


features = ['Soil_Type', 'Region', 'Irrigation', 'pH', 'Moisture', 'Nitrogen', 
            'Phosphorus', 'Potassium', 'Organic_Matter', 'Electrical_Conductivity', 
            'Bulk_Density', 'Clay_Percent', 'Temperature', 'Rainfall_7d', 
            'Sunlight_hours', 'Slope', 'Nutrient_Index', 'Extreme_pH']

X_train = train_df[features].copy()
y_train = train_df['Suitability'].copy()
X_test = test_df[features].copy()

num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='median')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

cat_cols = ['Soil_Type', 'Region', 'Irrigation']
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = X_train[col].fillna('Missing')
    X_test[col] = X_test[col].fillna('Missing')
    le.fit(pd.concat([X_train[col], X_test[col]]))
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

model = GradientBoostingClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)


res5 = pd.DataFrame({
    'subtaskID': 5,
    'datapointID': test_df['ID'],
    'answer': predictions
})

final_result = pd.concat([res1, res2, res3, res4, res5]).sort_values(by=['datapointID', 'subtaskID'])

final_result.to_csv('submission.csv', index=False)