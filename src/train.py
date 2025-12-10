"""
================================================================================
PHONE PRICE CLASSIFICATION - MODEL TRAINING SCRIPT
================================================================================
This script trains an ensemble of machine learning models to classify phones
as "expensive" or "non-expensive" based on their specifications.

Models Used:
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Multi-Layer Perceptron (Neural Network)
    - CatBoost Classifier

Ensemble Method: Voting Classifier with optimized weights
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

# Model imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# External libraries
from catboost import CatBoostClassifier

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Local imports
from preprocessing import preprocessing

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

# Load datasets
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Apply preprocessing
train_proc = preprocessing(train_df, is_train=True)
test_proc = preprocessing(test_df, is_train=False, train_df_for_encoding=train_df)

# Prepare features and target
X_train = train_proc.drop('price', axis=1)
y_train = train_proc['price']
X_test = test_proc.reindex(columns=X_train.columns, fill_value=0)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ============================================================================
# MODEL 1: RANDOM FOREST CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("MODEL 1: RANDOM FOREST CLASSIFIER")
print("="*70)

rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [900, 1000],      # Number of trees
    'max_depth': [15, 18],            # Maximum tree depth
    'min_samples_split': [2, 3],     # Minimum samples to split
    'min_samples_leaf': [1],          # Minimum samples in leaf
}

# Randomized search for faster hyperparameter tuning
grid_rf = RandomizedSearchCV(
    rf,
    param_grid_rf,
    n_iter=6,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

grid_rf.fit(X_train, y_train)

print("Best Random Forest Params:", grid_rf.best_params_)
print("Best CV Score:", grid_rf.best_score_)

# ============================================================================
# MODEL 2: GRADIENT BOOSTING CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("MODEL 2: GRADIENT BOOSTING CLASSIFIER")
print("="*70)

gbdt = GradientBoostingClassifier(random_state=42)

# Hyperparameter grid (using best known parameters)
param_grid_gbdt = {
    'n_estimators': [500],            # Number of boosting stages
    'max_depth': [6],                 # Maximum depth of trees
    'learning_rate': [0.05],          # Learning rate
    'subsample': [0.8],               # Fraction of samples for each tree
    'min_samples_split': [5]          # Minimum samples to split
}

# Grid search (single parameter set, so fast)
grid_gbdt = GridSearchCV(
    gbdt,
    param_grid_gbdt,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_gbdt.fit(X_train, y_train)

print("Best Gradient Boosting Params:", grid_gbdt.best_params_)
print("Best CV Score:", grid_gbdt.best_score_)

# ============================================================================
# MODEL 3: NEURAL NETWORK (MLPClassifier)
# ============================================================================

print("\n" + "="*70)
print("MODEL 3: NEURAL NETWORK (MLPClassifier)")
print("="*70)

# Create MLP pipeline with standardization
mlp_clf = make_pipeline(
    StandardScaler(),  # Standardize features for neural network
    MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Two hidden layers
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )
)

# Hyperparameter grid for MLP
param_grid_mlp = {
    'mlpclassifier__alpha': [1e-4],              # L2 regularization
    'mlpclassifier__learning_rate_init': [0.001]  # Initial learning rate
}

grid_mlp = GridSearchCV(
    estimator=mlp_clf,
    param_grid=param_grid_mlp,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)

grid_mlp.fit(X_train, y_train)

print("Best MLP Params:", grid_mlp.best_params_)
print("Best CV Score:", grid_mlp.best_score_)

# ============================================================================
# MODEL 4: CATBOOST CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("MODEL 4: CATBOOST CLASSIFIER")
print("="*70)

cat_model = CatBoostClassifier(
    random_state=42,
    verbose=False,
    thread_count=-1,
    loss_function='Logloss',
    eval_metric='F1',
    iterations=700,
    depth=6,
    learning_rate=0.08,
    l2_leaf_reg=5,
    subsample=0.8,
    border_count=128,
    bagging_temperature=1.0
)

# Train CatBoost and evaluate
cat_model.fit(X_train, y_train)
cat_score = cross_val_score(
    cat_model,
    X_train,
    y_train,
    cv=3,
    scoring='f1_weighted'
).mean()

print(f"CatBoost CV Score: {cat_score:.4f}")

# Create grid-like object for consistency
grid_bonus = type('obj', (object,), {
    'best_estimator_': cat_model,
    'best_score_': cat_score
})()

bonus_name = "CatBoost"

# ============================================================================
# ENSEMBLE CREATION: VOTING CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("CREATING VOTING ENSEMBLE (RF + GBDT + MLP + CatBoost)")
print("="*70)

# ----------------------------------------------------------------------------
# Step 1: Calibrate models for better probability estimates
# ----------------------------------------------------------------------------

print("\nCalibrating models for better probability estimates...")

calibrated_rf = CalibratedClassifierCV(
    grid_rf.best_estimator_,
    method='isotonic',
    cv=3
)
calibrated_gb = CalibratedClassifierCV(
    grid_gbdt.best_estimator_,
    method='isotonic',
    cv=3
)
calibrated_mlp = CalibratedClassifierCV(
    grid_mlp.best_estimator_,
    method='isotonic',
    cv=3
)
calibrated_bonus = CalibratedClassifierCV(
    grid_bonus.best_estimator_,
    method='isotonic',
    cv=3
)

# Fit calibrated models
calibrated_rf.fit(X_train, y_train)
calibrated_gb.fit(X_train, y_train)
calibrated_mlp.fit(X_train, y_train)
calibrated_bonus.fit(X_train, y_train)

print("✓ Model calibration complete")

# ----------------------------------------------------------------------------
# Step 2: Optimize ensemble weights
# ----------------------------------------------------------------------------

print("\nOptimizing ensemble weights (RF, GBDT, MLP, CatBoost)...")
print("Individual model CV scores:")
print(f"  RF:        {grid_rf.best_score_:.4f}")
print(f"  GBDT:      {grid_gbdt.best_score_:.4f}")
print(f"  MLP:       {grid_mlp.best_score_:.4f}")
print(f"  CatBoost:  {grid_bonus.best_score_:.4f}")

best_weights = None
best_ens_score = -1.0

# Weight combinations to test
# Format: [RF, GBDT, MLP, CatBoost]
# GBDT (best performer) gets highest weight, MLP (weakest) gets lower weight
weight_combinations = [
    [1, 5, 1, 2],  # GBDT very strong, MLP low, CatBoost medium
    [1, 6, 1, 2],  # GBDT dominant, MLP low
    [2, 5, 1, 2],  # RF and GBDT strong, MLP low
    [1, 5, 0, 3],  # Drop MLP, GBDT strong, CatBoost strong
    [1, 6, 0, 3],  # Drop MLP, GBDT dominant, CatBoost strong
    [2, 5, 0, 3],  # Drop MLP, RF and GBDT strong, CatBoost strong
]

print("\nTesting weight combinations...")
for weights in weight_combinations:
    if weights[2] == 0:
        # Ensemble without MLP (3 models)
        ens = VotingClassifier(
            estimators=[
                ('rf', calibrated_rf),
                ('gb', calibrated_gb),
                ('bonus', calibrated_bonus)
            ],
            voting='soft',
            weights=[w for i, w in enumerate(weights) if i != 2]
        )
    else:
        # Ensemble with all 4 models
        ens = VotingClassifier(
            estimators=[
                ('rf', calibrated_rf),
                ('gb', calibrated_gb),
                ('mlp', calibrated_mlp),
                ('bonus', calibrated_bonus)
            ],
            voting='soft',
            weights=weights
        )
    
    # Evaluate ensemble
    scores = cross_val_score(
        ens,
        X_train,
        y_train,
        cv=3,
        scoring='f1_weighted'
    )
    mean_score = scores.mean()
    
    if mean_score > best_ens_score:
        best_ens_score = mean_score
        best_weights = weights

# ----------------------------------------------------------------------------
# Step 3: Create final ensemble with best weights
# ----------------------------------------------------------------------------

if best_weights[2] == 0:
    print(f"\n✓ Best ensemble: 3 models (RF, GBDT, CatBoost)")
    print(f"  Best weights: {[w for i, w in enumerate(best_weights) if i != 2]}")
    print(f"  CV F1 Score: {best_ens_score:.4f}")
    
    final_god_model = VotingClassifier(
        estimators=[
            ('rf', calibrated_rf),
            ('gb', calibrated_gb),
            ('bonus', calibrated_bonus)
        ],
        voting='soft',
        weights=[w for i, w in enumerate(best_weights) if i != 2]
    )
else:
    print(f"\n✓ Best ensemble: 4 models (RF, GBDT, MLP, CatBoost)")
    print(f"  Best weights: {best_weights}")
    print(f"  CV F1 Score: {best_ens_score:.4f}")
    
    final_god_model = VotingClassifier(
        estimators=[
            ('rf', calibrated_rf),
            ('gb', calibrated_gb),
            ('mlp', calibrated_mlp),
            ('bonus', calibrated_bonus)
        ],
        voting='soft',
        weights=best_weights
    )

# Train final ensemble
final_god_model.fit(X_train, y_train)

# Final cross-validation evaluation
cv_scores = cross_val_score(
    final_god_model,
    X_train,
    y_train,
    cv=3,
    scoring='f1_weighted'
)

print(f"\nFinal Ensemble CV F1-Score (weighted): {cv_scores.mean():.4f} "
      f"(+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================================

print("\n" + "="*70)
print("SAVING MODEL AND ARTIFACTS")
print("="*70)

joblib.dump(final_god_model, '../results/final_model.pkl')
joblib.dump(X_train.columns.tolist(), '../results/train_columns.pkl')

print("✓ Model saved: ../results/final_model.pkl")
print("✓ Column names saved: ../results/train_columns.pkl")

# ============================================================================
# FINAL VALIDATION ON HELD-OUT SET
# ============================================================================

print("\n" + "="*70)
print("FINAL MODEL VALIDATION (20% Hold-Out Set)")
print("="*70)

# Split data for validation
X_train_full = train_proc.drop('price', axis=1)
y_train_full = train_proc['price']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    random_state=42,
    stratify=y_train_full,
    test_size=0.2
)

# Train on subset and predict on validation set
final_model = final_god_model
final_model.fit(X_tr, y_tr)
y_val_proba = final_model.predict_proba(X_val)[:, 1]

# ----------------------------------------------------------------------------
# Optimize decision threshold for best F1-score
# ----------------------------------------------------------------------------

print("\nOptimizing decision threshold...")

best_thr = 0.5
best_f1 = 0.0

# Search for optimal threshold
for thr in np.linspace(0.2, 0.8, 61):  # Test thresholds from 0.2 to 0.8
    y_tmp = (y_val_proba >= thr).astype(int)
    f1 = f1_score(y_val, y_tmp, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

# Make predictions with optimal threshold
y_val_pred = (y_val_proba >= best_thr).astype(int)

# ----------------------------------------------------------------------------
# Print validation results
# ----------------------------------------------------------------------------

print("\n" + "-"*70)
print("VALIDATION RESULTS")
print("-"*70)

print(f"Accuracy:              {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Best threshold:        {best_thr:.2f}")
print(f"F1-Score (weighted):   {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
print(f"F1-Score (macro):      {f1_score(y_val, y_val_pred, average='macro'):.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ============================================================================
# LABEL QUALITY CHECK: FIND SUSPICIOUS TRAINING EXAMPLES
# ============================================================================

print("\n" + "="*70)
print("CHECKING FOR POTENTIALLY MISLABELLED TRAINING EXAMPLES")
print("="*70)

# Use cross-validated predictions to find confident misclassifications
proba_cv = cross_val_predict(
    final_god_model,
    X_train_full,
    y_train_full,
    cv=3,
    method="predict_proba"
)[:, 1]

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
    print(f"Saved to: {suspect_path}")
    print("\nRecommendation: Manually inspect these rows and fix labels")
    print("in train.csv if needed, then retrain.")
else:
    print("No strongly suspicious labels found at chosen confidence thresholds.")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
