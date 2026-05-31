"""
Linear models predicting consumer sentiment.

Training data : up to and including 2022 Q2
Test data     : 2022 Q3 onwards

Model 1: CPI, cpi-cpi5, cpi-cpi10, mortgage_30yr, mortgage-mortgage5, mortgage-mortgage10
Model 2: Model 1 + real_wages
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── Load data ──────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "economic_sentiment.csv")
df = pd.read_csv(DATA_PATH, index_col="period_end", parse_dates=True)

FEATURES_M1 = [
    "cpi",
    "cpi-cpi5",
    "cpi-cpi10",
    "mortgage_30yr",
    "mortgage-mortgage5",
    "mortgage-mortgage10",
    "unemployment_u3",
]
FEATURES_M2 = FEATURES_M1 + ["real_wages"]
TARGET = "consumer_sentiment"

# ── Train / test split ─────────────────────────────────────────────────────────

train = df[df["quarter"] <= "2022Q2"]
test  = df[df["quarter"] >= "2022Q3"]

X_train_m1, y_train = train[FEATURES_M1], train[TARGET]
X_test_m1,  y_test  = test[FEATURES_M1],  test[TARGET]

X_train_m2 = train[FEATURES_M2]
X_test_m2  = test[FEATURES_M2]

# ── Fit models ─────────────────────────────────────────────────────────────────

m1 = LinearRegression().fit(X_train_m1, y_train)
m2 = LinearRegression().fit(X_train_m2, y_train)

# ── Predictions (full series for plotting) ─────────────────────────────────────

df["pred_m1"] = np.concatenate([
    m1.predict(train[FEATURES_M1]),
    m1.predict(test[FEATURES_M1]),
])
df["pred_m2"] = np.concatenate([
    m2.predict(train[FEATURES_M2]),
    m2.predict(test[FEATURES_M2]),
])

# ── Evaluate on test set ───────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}")

print("=" * 40)
print("Test-set performance (2022 Q3 onwards)")
print("=" * 40)
evaluate("Model 1 (no real wages)", y_test, m1.predict(X_test_m1))
evaluate("Model 2 (+ real wages)",  y_test, m2.predict(X_test_m2))

# ── Coefficients ───────────────────────────────────────────────────────────────

print("\n\nCoefficients")
print("-" * 40)
for model, features, label in [
    (m1, FEATURES_M1, "Model 1"),
    (m2, FEATURES_M2, "Model 2"),
]:
    print(f"\n{label}  (intercept: {model.intercept_:.2f})")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat:<22} {coef:+.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────

split_date = pd.Timestamp("2022-07-01")

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
titles = ["Model 1: CPI + Mortgage predictors", "Model 2: + Real Wages"]

for ax, pred_col, title in zip(axes, ["pred_m1", "pred_m2"], titles):
    ax.axvspan(split_date, df.index.max(), alpha=0.08, color="steelblue", label="Test period")
    ax.axvline(split_date, color="steelblue", linewidth=1.2, linestyle="--")

    ax.plot(df.index, df[TARGET],   color="black",      linewidth=1.5, label="Actual")
    ax.plot(df.index, df[pred_col], color="firebrick",  linewidth=1.5, linestyle="-", label="Predicted")

    # Highlight test-period predictions more boldly
    ax.plot(test.index, df.loc[test.index, pred_col],
            color="firebrick", linewidth=2.5, linestyle="-")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Consumer Sentiment Index")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)

    # Annotate split
    ax.text(split_date, ax.get_ylim()[0] + 2, " 2022 Q3\n split", fontsize=8,
            color="steelblue", va="bottom")

axes[-1].set_xlabel("Quarter")
fig.suptitle("Consumer Sentiment: Actual vs Predicted", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "data", "model_predictions.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out}")
plt.show()
