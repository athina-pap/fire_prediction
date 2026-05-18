#%%
import pandas as pd

# Load the dataset
df = pd.read_csv("fire_training_dataset_mixed_realistic.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:10], "...")
print(df.head(5))
# %%
# Identify columns to use as features and target
target_col = "y_fire"
# Exclude non-feature columns like flags used for labeling or IDs
feature_cols = ['t2m','d2m','u10','v10','tp','fwi', 'lat', 'lon', 'month', 'day_of_year', 'temp_dew_diff', 'wind_speed']

# Create new features for month and day_of_year from date, and some interactions
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

# Example interaction features: temperature-dewpoint difference (dryness) and wind speed magnitude
df['temp_dew_diff'] = df['t2m'] - df['d2m']            # higher difference -> drier air
df['wind_speed'] = (df['u10']**2 + df['v10']**2) ** 0.5  # wind speed magnitude

# Define feature matrix X and target vector y
X = df[feature_cols]
y = df[target_col]

# Impute missing values in X with median for numeric columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
print("After imputation, any remaining NaNs?", X_imputed.isna().sum().sum())

# %%
# (Optional) Oversample the minority class using SMOTE
 # install if not already available

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
print("Original class distribution:", y.value_counts().to_dict())
print("Resampled class distribution:", pd.Series(y_resampled).value_counts().to_dict())

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# %%
from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, 
                                                    test_size=0.2, random_state=42, stratify=y_resampled)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# %%
# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_clf.fit(X_train, y_train)

# 2. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)

# 3. XGBoost
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1, random_state=42)
# scale_pos_weight=1 since we already balanced classes; otherwise set = (negative_cases/positive_cases)
xgb_clf.fit(X_train, y_train)

# 4. LightGBM
from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier(class_weight='balanced', random_state=42)
lgb_clf.fit(X_train, y_train)

# 5. Multi-Layer Perceptron (Neural Network)
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_clf.fit(X_train, y_train)

# %%
from sklearn.metrics import confusion_matrix, classification_report

models = [("Logistic Regression", log_clf),
          ("Random Forest", rf_clf),
          ("XGBoost", xgb_clf),
          ("LightGBM", lgb_clf),
          ("MLP Neural Net", mlp_clf)]

for name, model in models:
    print(f"*** {name} ***")
    y_pred = model.predict(X_test)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    # Precision, Recall, F1
    print(classification_report(y_test, y_pred, digits=4))

# %%
# Ensemble of XGBoost and LightGBM
y_proba_xgb = xgb_clf.predict_proba(X_test)[:,1]
y_proba_lgb = lgb_clf.predict_proba(X_test)[:,1]
# Average the predicted probabilities
y_proba_ensemble = 0.5*y_proba_xgb + 0.5*y_proba_lgb
# Convert probabilities to binary predictions using 0.5 threshold
y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)

print("*** Ensemble: 0.5*XGB + 0.5*LGBM ***")
cm = confusion_matrix(y_test, y_pred_ensemble)
print("Confusion Matrix:\n", cm)
print(classification_report(y_test, y_pred_ensemble, digits=4))

# %%
