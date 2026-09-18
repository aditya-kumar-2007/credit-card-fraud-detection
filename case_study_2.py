"""
Case Study 2: Credit Card Fraud Detection
-----------------------------------------
XGBoost on a heavily imbalanced transaction dataset.

Expected dataset:
Kaggle Credit Card Fraud Detection dataset
CSV filename: data/creditcard.csv

The dataset normally contains:
- Time
- V1 ... V28
- Amount
- Class (0 = legitimate, 1 = fraud)

This implementation:
1. Loads and validates the dataset.
2. Splits data using stratification.
3. Uses SMOTE on the TRAINING data only.
4. Trains XGBoost.
5. Evaluates ROC-AUC and PR-AUC.
6. Tests multiple decision thresholds.
7. Prints feature importance.
8. Explains why threshold selection matters for fraud detection.

IMPORTANT:
SMOTE must be applied only to training data. Applying it before
the train/test split causes data leakage.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/creditcard.csv")
RANDOM_STATE = 42


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            "Download the Kaggle Credit Card Fraud Detection dataset "
            "and place creditcard.csv inside the data/ folder.\n"
        )

    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' was not found.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    return X, y


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n========== THRESHOLD {threshold:.2f} ==========")
    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    return predictions


def main():
    X, y = load_data(DATA_PATH)

    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())
    print(f"\nFraud percentage: {y.mean():.4%}")

    # Stratification preserves the rare fraud class in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    # SMOTE is applied ONLY to training data.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train, y_train
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Calculate scale_pos_weight from the original training data.
    # SMOTE already balances the training set, so we do not additionally
    # use scale_pos_weight here.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_resampled, y_train_resampled)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\n========== MODEL RESULT ==========")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    # Default threshold.
    evaluate_threshold(y_test, probabilities, 0.50)

    # Lower thresholds can catch more fraud but may create more false alarms.
    evaluate_threshold(y_test, probabilities, 0.30)
    evaluate_threshold(y_test, probabilities, 0.10)

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Credit Card Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=160)
    plt.show()

    # Precision-Recall curve.
    precision, recall, _ = precision_recall_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=160)
    plt.show()

    # Confusion matrix at threshold 0.30.
    predictions = (probabilities >= 0.30).astype(int)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    )
    disp.plot()
    plt.title("Confusion Matrix at Threshold 0.30")
    plt.tight_layout()
    plt.savefig("confusion_matrix_threshold_30.png", dpi=160)
    plt.show()

    # Feature importance.
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(15)

    print("\n========== TOP 15 FEATURES ==========")
    print(importance)

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top 15 XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=160)
    plt.show()

    print("\n========== INTERPRETATION ==========")
    print(
        "Credit-card fraud datasets are highly imbalanced: legitimate "
        "transactions greatly outnumber fraudulent ones."
    )
    print(
        "Accuracy alone can therefore be misleading. ROC-AUC and especially "
        "Precision-Recall AUC provide more useful information for this task."
    )
    print(
        "SMOTE creates synthetic minority-class training examples. It is "
        "applied only to the training split to avoid test-set leakage."
    )
    print(
        "Lowering the decision threshold generally catches more fraud "
        "(higher recall), but it can also flag more legitimate transactions "
        "as suspicious (more false positives)."
    )
    print(
        "The final threshold should be selected according to the financial "
        "cost of missed fraud, false alarms, and the review capacity of the "
        "fraud-detection system."
    )


if __name__ == "__main__":
    main()
