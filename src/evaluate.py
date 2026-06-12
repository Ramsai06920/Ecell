import joblib
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from src.train import train_models

def evaluate_models():
    xgb_model, ada_model, cat_model, le, X_test, y_test_enc = train_models()

    models = {
        'XGBoost':  xgb_model,
        'AdaBoost': ada_model,
        'CatBoost': cat_model
    }

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred_enc = model.predict(X_test)

        accuracy  = accuracy_score(y_test_enc, y_pred_enc)
        precision = precision_score(y_test_enc, y_pred_enc, average='weighted')
        recall    = recall_score(y_test_enc, y_pred_enc, average='weighted')
        f1        = f1_score(y_test_enc, y_pred_enc, average='weighted')

        results[name] = {
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f1_score':  f1,
            'model':     model
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        cm = confusion_matrix(y_test_enc, y_pred_enc)
        print(f"\nConfusion Matrix:")
        print(f"Classes: {list(le.classes_)}")
        print(cm)

        print(f"\nDetailed Report:")
        print(classification_report(y_test_enc, y_pred_enc,
                                    target_names=le.classes_))


    print(f"\n{'='*50}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)

    for name, metrics in results.items():
        print(f"{name:<12} {metrics['accuracy']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1_score']:>10.4f}")

    best_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_name]['model']

    print(f"\nBest model: {best_name} "
          f"(F1={results[best_name]['f1_score']:.4f})")

    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Best model saved to models/best_model.pkl")

    return best_name, best_model


if __name__ == "__main__":
    evaluate_models()
