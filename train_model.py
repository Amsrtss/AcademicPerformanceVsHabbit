import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv('df_encoded.csv')

# Pisah fitur dan target
y = df['exam_score']
X = df.drop(columns=['exam_score'])

# Ubah boolean ke float agar numeric semua
X = X.astype(float)

# Split data (72% train, 8% val, 20% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
val_size = 0.08 / (0.72 + 0.08)  # proporsi validation dari sisa data (train+val)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# StandardScaler fit di train, transform semua
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluasi di validation set
y_val_pred = model.predict(X_val_scaled)

# Hitung metrics
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"\nValidation MSE: {mse_val:.2f}")
print(f"Validation R²: {r2_val:.2f}")

# Evaluasi di test set
y_test_pred = model.predict(X_test_scaled)

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\nTest MSE: {mse_test:.2f}")
print(f"Test R²: {r2_test:.2f}")

# Save model + scaler
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and scaler saved.")

# Simpan hasil evaluasi
eval_results = {
    "validation": {
        "mse": mse_val,
        "r2": r2_val
    },
    "test": {
        "mse": mse_test,
        "r2": r2_test
    }
}


joblib.dump(eval_results, 'evaluation_results.pkl')
print("Evaluation results saved with joblib.")