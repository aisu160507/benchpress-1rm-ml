import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os

os.makedirs("results", exist_ok=True)


df = pd.read_csv("bench_press_sample_with_rpe.csv", encoding="utf-8-sig")
df["日付"] = pd.to_datetime(df["日付"])


df["総挙上量"] = df["重量(kg)"] * df["回数"] * df["セット数"]
df["前回重量"] = df["重量(kg)"].shift(1)
df["直近ボリュームMA"] = df["総挙上量"].rolling(window=3, min_periods=1).mean()
df["直近ボリューム差分"] = df["総挙上量"].diff()


df = df.dropna().reset_index(drop=True)


features = [
    "重量(kg)", "回数", "セット数", "休養日数", "RPE",
    "総挙上量", "前回重量", "直近ボリュームMA", "直近ボリューム差分"
]
X = df[features]
y = df["実測1RM(kg)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.3
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}


results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    results.append({"Model": name, "MAE": mae, "RMSE": rmse})

   
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual 1RM")
    plt.ylabel("Predicted 1RM")
    plt.title(f"{name}: Prediction vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{name}_scatter.png")
    plt.close()


baseline_pred = y_test.shift(1).fillna(method="bfill")
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
results.append({"Model": "Baseline(prev 1RM)", "MAE": baseline_mae, "RMSE": baseline_rmse})


lin = LinearRegression().fit(X_train, y_train)
coef = pd.Series(lin.coef_, index=X.columns).sort_values()

plt.figure(figsize=(8, 5))
coef.plot(kind="barh")
plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("results/LinearRegression_coefficients.png")
plt.close()


residuals = y_test - predictions["LinearRegression"]
plt.figure(figsize=(8, 4))
plt.plot(y_test.index, residuals, marker="o", linestyle="--")
plt.axhline(0, color="red", linestyle="dashed")
plt.title("Residuals (Linear Regression)")
plt.xlabel("Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig("results/LinearRegression_residuals.png")
plt.close()

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("results/results_with_rpe_full_metrics.csv", index=False, encoding="utf-8-sig")
