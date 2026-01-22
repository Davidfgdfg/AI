import pandas as pd

test = pd.read_csv(r'Rezolvari\Simulare MLcompete\Picturi celebre\test.csv')

def calculate_saa(row):
    saa = 0
    if row['stroke_density'] > 0.7:
        saa += 2
    if row['complexity'] > 0.65:
        saa += 2
    if row['uses_gold_leaf']:
        saa += 1
    if row['has_signature']:
        saa += 1
    if row['num_colors'] > 65 and row['colorfulness'] > 0.7:
        saa += 2
    if row['contrast'] < 0.4 or row['brightness'] < 0.45 or row['brightness'] > 0.75:
        saa -= 1
    
    return "Autentic" if saa >= 5 else "Incert"
test['SAA_frame'] = test.apply(calculate_saa, axis=1)
subtask1 = pd.DataFrame(
    {
        'SampleID': test['SampleID'],
        'subtaskID': 'Task1',
        'Answer': test['SAA_frame']
    }
)


#subtask 2
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
test = pd.read_csv(r'Rezolvari\Simulare MLcompete\Picturi celebre\test.csv')

numerical_cols = ['num_colors', 'colorfulness', 'complexity', 'brightness',
                  'contrast', 'stroke_density', 'complexity_x_stroke',
                  'painter_style_score', 'fake_style_score']
categorical_cols = ['brush_type', 'dominant_color']
boolean_cols = ['is_oil_painting', 'has_signature', 'dominant_warm_colors']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough') 


X = preprocessor.fit_transform(test[numerical_cols + categorical_cols + boolean_cols])


kmeans = KMeans(n_clusters=5, random_state=42)
test['cluster'] = kmeans.fit_predict(X)

cluster_means = test.groupby('cluster')['painter_style_score'].mean().sort_values()
mapping = {old: new for new, old in enumerate(cluster_means.index)}
test['painter_id'] = test['cluster'].map(mapping)


subtask2 = pd.DataFrame(
    {
        'SampleID': test['SampleID'],
        'subtaskID': 'Task2',
        'Answer': test['painter_id']
    })

#subtask 3
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_csv(r'Rezolvari\Simulare MLcompete\Picturi celebre\train.csv')
test = pd.read_csv(r'Rezolvari\Simulare MLcompete\Picturi celebre\test.csv')

for df in [train, test]:
    df[['width', 'height']] = df['canvas_size'].str.split('x', expand=True).astype(float)
    df['area'] = df['width'] * df['height']
    df['aspect_ratio'] = df['width'] / df['height']
    df['color_density'] = df['num_colors'] / df['area']
    df['visual_energy'] = df['colorfulness'] * df['contrast'] * df['stroke_density']


bool_cols = [
    'is_oil_painting', 'has_signature', 'is_framed',
    'uses_gold_leaf', 'is_restored', 'dominant_warm_colors'
]

for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)


y = train['target_price']

X_train = train.drop(columns=['target_price', 'SampleID'])
X_test = test.drop(columns=['SampleID'])


cat_cols = X_train.select_dtypes(include='object').columns.tolist()



from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

model = CatBoostRegressor(
    iterations=1200,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=10,
    bagging_temperature=0.8,
    loss_function='MAE',
    random_seed=42,
    verbose=200
)

model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    cat_features=cat_cols,
    use_best_model=True
)

val_pred = model.predict(X_val).round()
mae = mean_absolute_error(y_val, val_pred)
print("MAE local:", mae)


test_pred = model.predict(X_test).round().astype(int)


subtask3 = pd.DataFrame(
    {
        'SampleID': test['SampleID'],
        'subtaskID': 'Task3',
        'Answer': test_pred

    })



submission = pd.concat([subtask1, subtask2,subtask3], ignore_index=True)
submission.to_csv("submission.csv", index=False)