import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    """
    Load training and test datasets and perform initial cleanup.

    Parameters:
        train_path (str): Path to training dataset
        test_path (str): Path to test dataset

    Returns:
        train (DataFrame): Cleaned training dataset
        test (DataFrame): Cleaned test dataset
        test_ids (Series): IDs for test dataset
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop ID column (not useful for modeling)
    train = train.drop(columns=['Unnamed: 0'])
    test_ids = test['Unnamed: 0']
    test = test.drop(columns=['Unnamed: 0'])

    return train, test, test_ids


def split_data(train):
    """
    Split training data into train and validation sets.

    Parameters:
        train (DataFrame): Full training dataset

    Returns:
        X_train, X_val, y_train, y_val
    """

    # Separate features and target
    X = train.drop(columns=['SeriousDlqin2yrs'])
    y = train['SeriousDlqin2yrs']

    # Stratified split to preserve class distribution
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

def build_preprocessor(X):
    """
    Create preprocessing pipeline for numeric features.

    Steps:
    - Handles missing values using median imputation
    - Scales features using StandardScaler

    Parameters:
        X (DataFrame): Feature dataset

    Returns:
        preprocessor (ColumnTransformer)
    """

    # All features are numeric in this dataset
    numeric_features = X.columns

    # Pipeline: Imputation → Scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Apply transformation to numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    return preprocessor
