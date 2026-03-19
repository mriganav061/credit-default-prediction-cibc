import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def compute_ks(y_true, y_prob):
    """
    Compute the Kolmogorov-Smirnov (KS) statistic.

    KS measures the maximum separation between the cumulative
    distributions of positive and negative classes. It is widely
    used in credit risk modeling to evaluate discriminatory power.

    Parameters:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities for the positive class

    Returns:
        float: KS statistic
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(tpr - fpr)


def get_models(y_train):
    """
    Define candidate models for comparison.

    Parameters:
        y_train (array-like): Training target used to compute class imbalance ratio

    Returns:
        dict: Dictionary of model name -> estimator
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    }

    return models


def train_and_evaluate(models, preprocessor, X_train, X_val, y_train, y_val):
    """
    Train multiple models, evaluate them using cross-validation and
    holdout validation metrics, and return the best-performing model.

    Evaluation metrics:
    - CV ROC-AUC: robustness across folds
    - Validation ROC-AUC: overall ranking performance
    - PR-AUC: useful for imbalanced classification
    - KS: credit risk discriminatory power

    Parameters:
        models (dict): Dictionary of model name -> estimator
        preprocessor: Preprocessing pipeline or transformer
        X_train (DataFrame): Training feature set
        X_val (DataFrame): Validation feature set
        y_train (Series/array): Training target
        y_val (Series/array): Validation target

    Returns:
        best_model: Best fitted pipeline based on validation ROC-AUC
        results_df (DataFrame): Summary table of model metrics
        pipelines (dict): Dictionary of model name -> fitted pipeline
    """
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    pipelines = {}
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        # Create end-to-end modeling pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Cross-validation on the training set only
        cv_auc_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        # Fit model on full training split
        pipeline.fit(X_train, y_train)

        # Predict probabilities on validation set
        val_prob = pipeline.predict_proba(X_val)[:, 1]

        # Compute evaluation metrics
        val_auc = roc_auc_score(y_val, val_prob)
        pr_auc = average_precision_score(y_val, val_prob)
        ks = compute_ks(y_val, val_prob)

        results.append({
            "model": name,
            "cv_auc_mean": cv_auc_scores.mean(),
            "cv_auc_std": cv_auc_scores.std(),
            "val_auc": val_auc,
            "pr_auc": pr_auc,
            "ks": ks
        })

        pipelines[name] = pipeline

        print(
            f"{name} -> "
            f"CV AUC: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std():.4f}), "
            f"Val AUC: {val_auc:.4f}, "
            f"PR AUC: {pr_auc:.4f}, "
            f"KS: {ks:.4f}"
        )

        # Track best model based on validation ROC-AUC
        if val_auc > best_score:
            best_score = val_auc
            best_model = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="val_auc",
        ascending=False
    ).reset_index(drop=True)

    return best_model, results_df, pipelines