import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error,
    roc_auc_score, classification_report, mean_squared_error, r2_score,
    )
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import numpy as np
import joblib

# if __name__ == "__main__":

# load datasets
df = load_dataset("matbench_expt_gap")
# Convert formulas to Composition objects
df['composition'] = df["composition"].apply(Composition)
# Instantiate the featurizer
ep = ElementProperty.from_preset("magpie")
# Generating the feature matrix
feat_array = ep.featurize_many(df["composition"], ignore_errors=True)
feat_df = pd.DataFrame(feat_array,
                    columns = ep.feature_labels(),
                    index = df.index)
df = pd.concat([df, feat_df], axis=1)
# renaming "gap expt" to "Eg" and setting it as index
df.rename(columns={"gap expt":"Eg"}, inplace=True)
df.set_index('composition', inplace=True)
# Splitting dataset
x = df.drop('Eg', axis=1)
y = df['Eg']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify= y==0) # keeps metal/non‐metal ratio the same in both sets but on y it is not working correctly
# Label metals (0) vs. non-metals (1) based on experimental band gap
y_train_clf = y_train.apply(lambda x: 0 if x==0 else 1)
y_test_clf = y_test.apply(lambda x: 0 if x==0 else 1)
# Train XGBoost classifier
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(x_train, y_train_clf)
# Predict
y_pred_clf = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]  # Probabilities for ROC-AUC
# Evaluate classifier
print("📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("ROC-AUC:", roc_auc_score(y_test_clf, y_proba))
# Optional: More detailed report
print("\nClassification Report:\n", classification_report(y_test_clf, y_pred_clf, target_names=['Non-metal', 'Metal']))
# Get confusion matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
# Label positions and annotations
labels = np.array([["TP", "FP"], ["FN", "TN"]])
counts = cm
# Create a string matrix with count and label
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n{labels[i, j]}"
# Plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=annot, fmt="", cmap="Reds", cbar=False,
            xticklabels=["non-metal", "metal"],
            yticklabels=["non-metal", "metal"])
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("(c)")
plt.tight_layout()
plt.show()
# Regression on non-metals only
# Dataset used for regression (0nly non-metals)
x_reg = df[df['Eg']>0].drop('Eg', axis=1)
y_reg = df[df['Eg']>0]['Eg']
# splitting the data
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.2, random_state=42)
# Training Regression model
reg = XGBRegressor(random_state=42)
reg.fit(x_train_reg,y_train_reg)
# predicting
y_pred_reg = reg.predict(x_test_reg)
# Evaluating Regression Model
print("📊 Evaluating Regression Model:")
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)
mad = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"RMSE [eV]: {rmse:.3f}")
print(f"R² Score : {r2:.3f}")
print(f"MAD: {mad:.3f}")
# Saving the models
joblib.dump(model, './XGBClassifier.joblib')
joblib.dump(reg, './XGBRegressor.joblib')
# --- Plot experimental test set with ideal and fit lines ---
# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, label='Predictions')
# Ideal line (y = x)
lims = [min(y_test_reg.min(), y_pred_reg.min()), max(y_test_reg.max(), y_pred_reg.max())]
plt.plot(lims, lims, 'r--', label='Ideal (y = x)')
# Labels and legend
plt.xlabel('Measured $E_g$ (exp)')
plt.ylabel('Predicted $E_g$ (XGB)')
plt.title('Predicted vs Measured (Experimental)')
plt.legend()
plt.tight_layout()
plt.show()