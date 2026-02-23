import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cardekho_dataset.csv")

# Drop unnecessary columns
df.drop(["Unnamed: 0", "car_name"], axis=1, inplace=True)

# ---------------- FEATURES ----------------
X = df.drop("selling_price", axis=1)
y = np.log(df["selling_price"])

# ---------------- CATEGORICAL + NUMERICAL ----------------
categorical_cols = ["brand", "fuel_type", "transmission_type", "seller_type"]
numerical_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]

# ---------------- ENCODING ----------------
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_cat = encoder.fit_transform(X[categorical_cols])
X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Combine with numerical data
X_final = pd.concat([X_cat_df, X[numerical_cols].reset_index(drop=True)], axis=1)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred = model.predict(X_test)

# Convert back
y_test_actual = np.exp(y_test)
y_pred_actual = np.exp(y_pred)

# ---------------- EVALUATION ----------------
r2 = r2_score(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print("\n MODEL PERFORMANCE")
print(f"R2 Score  : {r2:.4f}")
print(f"MAE       : ₹ {mae:,.0f}")
print(f"RMSE      : ₹ {rmse:,.0f}")

# ---------------- FEATURE IMPORTANCE ----------------
feat_imp = pd.DataFrame({
    "Feature": X_final.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n Top 10 Features:")
print(feat_imp.head(10))

# ---------------- SAVE EVERYTHING ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(X_final.columns, open("columns.pkl", "wb"))

print("\nModel saved!")