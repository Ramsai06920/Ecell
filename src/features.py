import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

def get_features():
    df = pd.read_csv('data/clean_data.csv')
    df = df.dropna(subset=['clean_text'])
    print(f"Loaded {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    X = df['clean_text']
    y = df['label']
    print("\nApplying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_vectorized = vectorizer.fit_transform(X)
    print(f"Feature matrix shape: {X_vectorized.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")

    # Save vectorizer for use in API later
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("\nVectorizer saved to models/vectorizer.pkl")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    get_features()