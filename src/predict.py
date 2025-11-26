# predict_final.py  ← RUN THIS TO GET YOUR FINAL SUBMISSION
import pandas as pd
import joblib
from src.preprocessing import preprocessing  # ← your exact function

print("GENERATING OFFICIAL SUBMISSION")
print("="*60)

# 1. Load test data (NO price column, NO model_name needed)
test_df = pd.read_csv('../data/test.csv')   # ← your real test file
print(f"Test data shape: {test_df.shape}")

# 2. Load training data for proper encoding
train_df = pd.read_csv('../data/train.csv')

# 3. Apply EXACT SAME preprocessing as training
#     → is_train=False so no target mapping, but pass train_df for encoding
X_test_processed = preprocessing(test_df, is_train=False, train_df_for_encoding=train_df)

print(f"After preprocessing: {X_test_processed.shape}")

# 4. Load your best trained model
#     (you should have saved it earlier with joblib.dump)
try:
    model = joblib.load('../results/final_model.pkl')
    print("Loaded saved best model")
except:
    print("Model not found! Train first and save with joblib.dump(model, 'final_model.pkl')")
    exit()

# 5. Make sure test has same columns as training (critical!)
#     Save training columns once after first training:
#     joblib.dump(X_train.columns.tolist(), '../results/train_columns.pkl')
try:
    train_columns = joblib.load('../results/train_columns.pkl')
    X_test_processed = X_test_processed.reindex(columns=train_columns, fill_value=0)
    print("Columns aligned successfully")
except:
    print("train_columns.pkl not found → assuming columns already match")

# 6. Predict
y_pred = model.predict(X_test_processed)
y_pred_labels = ['expensive' if pred == 1 else 'non-expensive' for pred in y_pred]

# 7. Save official submission
submission = pd.DataFrame({'price': y_pred_labels})
submission.to_csv('../results/submission.csv', index=False)

print("\nSUBMISSION READY!")
print(f"Total predictions: {len(y_pred_labels)}")
print("First 15 predictions:")
print(submission.head(50))

print("\nFile saved: ../results/submission.csv")
print("→ Submit this file to get your final score!")