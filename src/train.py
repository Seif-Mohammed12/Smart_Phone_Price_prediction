import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from preprocessing import preprocessing

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

train_proc = preprocessing(train_df, is_train=True)
test_proc  = preprocessing(test_df,  is_train=False, train_df_for_encoding=train_df)

X_train = train_proc.drop('price', axis=1)
y_train = train_proc['price']
X_test  = test_proc.reindex(columns=X_train.columns, fill_value=0)

print("\n" + "="*60)
print("MODEL 1: RANDOM FOREST CLASSIFIER")
print("="*60)

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

param_grid_rf = {
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best Random Forest Params:", grid_rf.best_params_)
print("Best CV Score:", grid_rf.best_score_)


print("\n" + "="*60)
print("MODEL 2: GRADIENT BOOSTING CLASSIFIER")
print("="*60)

gbdt = GradientBoostingClassifier(random_state=42)

param_grid_gbdt = {
    'n_estimators': [200, 300, 500, 700],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5]
}

grid_gbdt = GridSearchCV(gbdt, param_grid_gbdt, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_gbdt.fit(X_train, y_train)

print("Best Gradient Boosting Params:", grid_gbdt.best_params_)
print("Best CV Score:", grid_gbdt.best_score_)

# === SIMPLE NEURAL NETWORK (MLP) ===
print("\n" + "="*60)
print("MODEL 3: NEURAL NETWORK (MLPClassifier)")
print("="*60)

mlp_clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )
)

param_grid_mlp = {
    'mlpclassifier__alpha': [1e-4, 1e-3],
    'mlpclassifier__learning_rate_init': [0.001, 0.01],
}

grid_mlp = GridSearchCV(
    estimator=mlp_clf,
    param_grid=param_grid_mlp,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_mlp.fit(X_train, y_train)

print("Best MLP Params:", grid_mlp.best_params_)
print("Best CV Score:", grid_mlp.best_score_)

# === IMPROVED ENSEMBLE WITH BEST THREE MODELS (RF, GBDT, MLP) ===
print("\n" + "="*70)
print("CREATING IMPROVED ENSEMBLE (RF + GBDT + MLP)")
print("="*70)

# Calibrate the three strongest models for better probability estimates
calibrated_rf = CalibratedClassifierCV(grid_rf.best_estimator_, method='isotonic', cv=5)
calibrated_gb = CalibratedClassifierCV(grid_gbdt.best_estimator_, method='isotonic', cv=5)
calibrated_mlp = CalibratedClassifierCV(grid_mlp.best_estimator_, method='isotonic', cv=3)

calibrated_rf.fit(X_train, y_train)
calibrated_gb.fit(X_train, y_train)
calibrated_mlp.fit(X_train, y_train)

# Small grid-search over ensemble weights for RF, GBDT, MLP
from sklearn.model_selection import cross_val_score
best_weights = None
best_ens_score = -1.0

for w_rf in [1, 2]:
    for w_gb in [1, 2, 3]:
        for w_mlp in [1, 2, 3]:
            weights = [w_rf, w_gb, w_mlp]
            ens = VotingClassifier(
                estimators=[
                    ('rf', calibrated_rf),
                    ('gb', calibrated_gb),
                    ('mlp', calibrated_mlp)
                ],
                voting='soft',
                weights=weights
            )
            scores = cross_val_score(ens, X_train, y_train, cv=5, scoring='f1_weighted')
            mean_score = scores.mean()
            if mean_score > best_ens_score:
                best_ens_score = mean_score
                best_weights = weights

print(f"\nBest ensemble weights (RF, GBDT, MLP): {best_weights} with CV F1 = {best_ens_score:.4f}")

# Final ensemble with best-found weights
final_god_model = VotingClassifier(
    estimators=[
        ('rf', calibrated_rf),
        ('gb', calibrated_gb),
        ('mlp', calibrated_mlp)
    ],
    voting='soft',
    weights=best_weights
)

final_god_model.fit(X_train, y_train)  # final fit on full train

# Evaluate ensemble with best weights
cv_scores = cross_val_score(final_god_model, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"\nEnsemble CV F1-Score (weighted) with best weights: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

joblib.dump(final_god_model, '../results/final_model.pkl')
joblib.dump(X_train.columns.tolist(), '../results/train_columns.pkl')
print("\n✓ BEST MODEL & COLUMNS SAVED CORRECTLY!")

# === FINAL VALIDATION ===
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

X_train_full = train_proc.drop('price', axis=1)
y_train_full = train_proc['price']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, random_state=42, stratify=y_train_full, test_size=0.2
)

# Test the final ensemble model
final_model = final_god_model
final_model.fit(X_tr, y_tr)
y_val_proba = final_model.predict_proba(X_val)[:, 1]

# Tune decision threshold for best weighted F1
from sklearn.metrics import f1_score, classification_report, confusion_matrix

best_thr = 0.5
best_f1 = 0.0
for thr in np.linspace(0.2, 0.8, 61):  # step 0.01
    y_tmp = (y_val_proba >= thr).astype(int)
    f1 = f1_score(y_val, y_tmp, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

y_val_pred = (y_val_proba >= best_thr).astype(int)

print("\n" + "="*70)
print("FINAL MODEL VALIDATION (20% hold-out)")
print("="*70)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Best threshold (for weighted F1): {best_thr:.2f}")
print(f"F1-Score (weighted): {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
print(f"F1-Score (macro): {f1_score(y_val, y_val_pred, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# === LABEL QUALITY CHECK: FIND SUSPICIOUS TRAINING EXAMPLES ===
print("\n" + "="*70)
print("CHECKING FOR POSSIBLE MISLABELLED TRAIN ROWS")
print("="*70)

# Use cross-validated predictions on the full training set
proba_cv = cross_val_predict(final_god_model, X_train_full, y_train_full,
                             cv=5, method="predict_proba")[:, 1]

# Flag rows where model is very confident but label disagrees
confident_wrong = (
    ((y_train_full == 1) & (proba_cv < 0.15)) |
    ((y_train_full == 0) & (proba_cv > 0.85))
)

suspect_indices = X_train_full.index[confident_wrong]

if len(suspect_indices) > 0:
    suspect_df = train_df.loc[suspect_indices].copy()
    suspect_df["label_numeric"] = y_train_full.loc[suspect_indices].values
    suspect_df["model_proba_expensive"] = proba_cv[confident_wrong]

    suspect_path = "../results/suspicious_labels.csv"
    suspect_df.to_csv(suspect_path, index=False)
    print(f"Found {len(suspect_indices)} potentially mislabelled rows.")
    print(f"Saved them to: {suspect_path}")
    print("Manually inspect these rows and fix labels in train.csv if needed, then retrain.")
else:
    print("No strongly suspicious labels found at chosen confidence thresholds.")
