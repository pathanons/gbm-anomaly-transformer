from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.gbm.baselines.window_features import extract_window_features
from src.gbm.data import JointWindowDataset


def build_feature_matrix(dataset: JointWindowDataset) -> tuple[pd.DataFrame, pd.Series]:
    feature_rows = []
    labels = []
    meta_rows = []

    for index in range(len(dataset)):
        batch = dataset[index]
        x_window = batch["x"].numpy()
        returns_window = batch["returns"].numpy()
        feature_rows.append(extract_window_features(x_window, returns_window))
        labels.append(int(batch["y"].item()))
        meta_rows.append(batch["meta"])

    feature_names = [f"f{i}" for i in range(len(feature_rows[0]))]
    X = pd.DataFrame(feature_rows, columns=feature_names)
    y = pd.Series(labels, name="y_true")
    meta = pd.DataFrame(meta_rows)
    meta["y_true"] = y
    return meta.join(X), y


def fit_random_forest(
    train_meta_x: pd.DataFrame,
    train_y: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    feature_cols = [col for col in train_meta_x.columns if col.startswith("f")]
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_meta_x[feature_cols])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, train_y.to_numpy())
    model.feature_cols_ = feature_cols  # type: ignore[attr-defined]
    model.imputer_ = imputer  # type: ignore[attr-defined]
    return model


def fit_isolation_forest(
    train_meta_x: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    feature_cols = [col for col in train_meta_x.columns if col.startswith("f")]
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_meta_x[feature_cols])
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    model.feature_cols_ = feature_cols  # type: ignore[attr-defined]
    model.imputer_ = imputer  # type: ignore[attr-defined]
    return model


def score_isolation_forest(model: IsolationForest, meta_x: pd.DataFrame) -> pd.Series:
    feature_cols = model.feature_cols_  # type: ignore[attr-defined]
    imputer = model.imputer_  # type: ignore[attr-defined]
    X = imputer.transform(meta_x[feature_cols])
    # Higher score = more anomalous (negate sklearn decision_function).
    return pd.Series(-model.decision_function(X), index=meta_x.index)


def score_random_forest(model: RandomForestClassifier, meta_x: pd.DataFrame) -> pd.Series:
    feature_cols = model.feature_cols_  # type: ignore[attr-defined]
    imputer = model.imputer_  # type: ignore[attr-defined]
    X = imputer.transform(meta_x[feature_cols])
    if hasattr(model, "predict_proba") and len(model.classes_) > 1:
        class_index = list(model.classes_).index(1)
        return pd.Series(model.predict_proba(X)[:, class_index], index=meta_x.index)
    return pd.Series(model.predict(X), index=meta_x.index, dtype=float)
