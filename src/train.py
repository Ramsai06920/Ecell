import joblib
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import get_features


def train_models():
    # Load features from features.py
    X_train, X_test, y_train, y_test = get_features()

    # Convert string labels to numbers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Train XGBoost    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train_enc)
    print("XGBoost training complete.")

    # Train CatBoost
    print("\nTraining CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train_enc)
    print("CatBoost training complete.")

    # Train AdaBoost
    print("\nTraining AdaBoost...")
    ada_model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    ada_model.fit(X_train, y_train_enc)
    print("AdaBoost training complete.")

    # Save models and label encoder 
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    joblib.dump(cat_model, 'models/cat_model.pkl')
    joblib.dump(ada_model, 'models/ada_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')

    return xgb_model, cat_model, ada_model, le, X_test, y_test_enc

if __name__ == "__main__":
    train_models()