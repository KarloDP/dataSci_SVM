import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

train_df = pd.read_csv('student_academic_performance_train.csv')
test_df = pd.read_csv('student_academic_performance_testing.csv')
unseen_df = pd.read_csv('student_academic_performance_unseen.csv')

SCORE_COLS = ["Math_Score", "Reading_Score", "Writing_Score"]

def add_target(df):
    df = df.copy()
    avg = df[SCORE_COLS].mean(axis=1)
    df["Performance"] = pd.cut(
        avg,
        bins=[0,60,75,100],
        labels=["Low", "Medium", "High"],
        right=True
   )
    return df
train_df = add_target(train_df)
test_df = add_target(test_df)
unseen_df = add_target(unseen_df)

DROP_COLS = ["Student_ID"] + SCORE_COLS + ["Performance"]

FEATURE_COLS = [c for c in train_df.columns if c not in DROP_COLS]

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Performance"].values

X_test = test_df[FEATURE_COLS].values
y_test = test_df["Performance"].values

X_unseen = unseen_df[FEATURE_COLS].values
y_unseen = unseen_df["Performance"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_unseen_scaled = scaler.transform(X_unseen)

print("\n--- Training SVM ---")
svm_model = SVC(kernel="rbf", C=10.0, gamma="scale",
                class_weight="balanced", random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("Training complete.")

y_pred = svm_model.predict(X_test_scaled)

print("\n" + "=" * 55)
print("  TEST SET EVALUATION")
print("=" * 55)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["High", "Low", "Mid"]))

y_unseen_pred = svm_model.predict(X_unseen_scaled)

print("=" * 55)
print("  UNSEEN DATA EVALUATION")
print("=" * 55)
print(f"Accuracy : {accuracy_score(y_unseen, y_unseen_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_unseen, y_unseen_pred, target_names=["High", "Low", "Mid"]))

# ──────────────────────────────────────────────
# CONFUSION MATRIX PLOT (Test Set)
# ──────────────────────────────────────────────
labels = ["High", "Low", "Mid"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SVM — Student Academic Performance", fontsize=14, fontweight="bold")

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Confusion Matrix (Test Set)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Class distribution comparison
class_order = ["Low", "Mid", "High"]
train_counts = train_df["Performance"].value_counts().reindex(class_order)
test_counts = test_df["Performance"].value_counts().reindex(class_order)
unseen_counts = unseen_df["Performance"].value_counts().reindex(class_order)

x = np.arange(len(class_order))
width = 0.25
axes[1].bar(x - width, train_counts, width, label="Train", color="#4C72B0")
axes[1].bar(x, test_counts, width, label="Test", color="#55A868")
axes[1].bar(x + width, unseen_counts, width, label="Unseen", color="#C44E52")
axes[1].set_xticks(x)
axes[1].set_xticklabels(class_order)
axes[1].set_title("Class Distribution Across Splits")
axes[1].set_xlabel("Performance Grade")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.savefig("svm_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to svm_results.png")

sample = X_unseen_scaled[[0]]
prediction = svm_model.predict(sample)
print(f"\nSample Unseen Student Prediction: {prediction[0]}")
print(f"Actual Label                    : {y_unseen[0]}")
