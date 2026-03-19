from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_models():
    """
    Define candidate models for comparison.

    Returns:
        dict: Model name -> model object
    """

    return {
        "logistic": LogisticRegression(max_iter=1000),
        
        "rf": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"  # helps with imbalance
        ),
        
        "gb": GradientBoostingClassifier(),

        "xgb": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",   # required to avoid warning
            use_label_encoder=False
        )
    }



def train_and_evaluate(models, preprocessor, X_train, X_val, y_train, y_val):
    """
    Train multiple models and evaluate using ROC-AUC.

    Selects the best-performing model.

    Parameters:
        models (dict): Dictionary of models
        preprocessor: Preprocessing pipeline
        X_train, X_val, y_train, y_val: Split data

    Returns:
        best_model: Best trained pipeline
        results: Dictionary of model performance
    """

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():

        # Combine preprocessing + model into single pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predict probabilities (needed for ROC-AUC)
        probs = pipeline.predict_proba(X_val)[:, 1]

        # Evaluate model
        auc = roc_auc_score(y_val, probs)
        print(f"{name} AUC: {auc:.4f}")

        results[name] = auc

        # Track best model
        if auc > best_score:
            best_score = auc
            best_model = pipeline

    return best_model, results