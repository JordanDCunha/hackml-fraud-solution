import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from catboost import CatBoostClassifier


def feature_engineering(df):
    df = df.copy()

    df['orig_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_diff'] = df['oldbalanceDest'] - df['newbalanceDest']

    df['orig_error'] = df['amount'] - df['orig_diff']
    df['dest_error'] = df['amount'] - df['dest_diff']

    df['is_orig_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['is_dest_zero'] = (df['oldbalanceDest'] == 0).astype(int)

    return df


def main():
    # ==============================
    # Load Data
    # ==============================
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print("Original Train shape:", train.shape)
    print("Test shape:", test.shape)

    # ==============================
    # Sample (faster)
    # ==============================
    train = train.sample(300000, random_state=42)
    print("Sampled Train shape:", train.shape)

    # ==============================
    # Feature Engineering
    # ==============================
    train_fe = feature_engineering(train)
    test_fe = feature_engineering(test)

    # Extra features
    for df in [train_fe, test_fe]:
        df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
        df['is_cashout'] = (df['type'] == 'CASH_OUT').astype(int)

    # ==============================
    # Encode for LightGBM
    # ==============================
    le = LabelEncoder()
    train_fe['type_encoded'] = le.fit_transform(train_fe['type'])
    test_fe['type_encoded'] = le.transform(test_fe['type'])

    drop_cols = ['nameOrig', 'nameDest', 'id']

    # LightGBM
    X_lgb = train_fe.drop(columns=drop_cols + ['urgency_level', 'type'])
    X_test_lgb = test_fe.drop(columns=drop_cols + ['type'])

    # CatBoost
    X_cat = train_fe.drop(columns=drop_cols + ['urgency_level'])
    X_test_cat = test_fe.drop(columns=drop_cols)

    y = train_fe['urgency_level']

    # ==============================
    # Split
    # ==============================
    X_train_lgb, X_val_lgb, y_train, y_val = train_test_split(
        X_lgb, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_cat, X_val_cat, _, _ = train_test_split(
        X_cat, y, test_size=0.2, stratify=y, random_state=42
    )

    # ==============================
    # Class weights
    # ==============================
    class_counts = y.value_counts().to_dict()
    total = len(y)
    class_weights = {cls: total / count for cls, count in class_counts.items()}

    # ==============================
    # LightGBM
    # ==============================
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        class_weight=class_weights,
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )

    lgb_model.fit(X_train_lgb, y_train)
    lgb_preds = lgb_model.predict(X_val_lgb)

    print("\n=== LightGBM ===")
    print("F1:", f1_score(y_val, lgb_preds, average='macro'))

    # ==============================
    # CatBoost
    # ==============================
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        verbose=0
    )

    cat_model.fit(X_train_cat, y_train, cat_features=['type'])

    cat_preds = cat_model.predict(X_val_cat)
    cat_preds = np.array([int(x[0]) for x in cat_preds])

    print("\n=== CatBoost ===")
    print("F1:", f1_score(y_val, cat_preds, average='macro'))

    # ==============================
    # Ensemble
    # ==============================
    final_val_preds = [
        Counter([lgb_preds[i], cat_preds[i]]).most_common(1)[0][0]
        for i in range(len(lgb_preds))
    ]

    print("\n=== Ensemble ===")
    print("F1:", f1_score(y_val, final_val_preds, average='macro'))

    # ==============================
    # Train full
    # ==============================
    lgb_model.fit(X_lgb, y)
    cat_model.fit(X_cat, y, cat_features=['type'])

    # ==============================
    # Test predictions
    # ==============================
    lgb_test = lgb_model.predict(X_test_lgb)

    cat_test = cat_model.predict(X_test_cat)
    cat_test = np.array([int(x[0]) for x in cat_test])

    final_test = [
        Counter([lgb_test[i], cat_test[i]]).most_common(1)[0][0]
        for i in range(len(lgb_test))
    ]

    # ==============================
    # Submission
    # ==============================
    submission = pd.DataFrame({
        'id': test['id'],
        'urgency_level': final_test
    })

    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created!")


if __name__ == "__main__":
    main()
