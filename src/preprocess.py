import sys

from matplotlib.pylab import sample
import pandas as pd 
import os
import re #for cleaning the text
from datasets import load_dataset

HIGH_RISK_WORDS = {
    'risk','loss','debt','lawsuit','bankruptcy','decline', 'failure', 'fraud', 'penalty', 'litigation'
}

LOW_RISK_WORDS = {
    'growth','profit','revenue','expansion','strong','increase','improvement', 'gain', 'successful', 'record'
}

# def clean_text(text):
#     text = str(text)
#     text = text.lower()
#     text = re.sub(r'http\S+|www\S+', '', text)
#     #text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'[^a-z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Keep only lowercase letters and spaces
    text = re.sub(r'[^a-z ]', '', text)   # changed \s to literal space
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def assign_label(text):
    words = text.split()
    total = len(words)

    if total == 0:
        return 'medium_risk'

    high_count = sum(1 for w in words if w in HIGH_RISK_WORDS)
    low_count = sum(1 for w in words if w in LOW_RISK_WORDS)

    high_score = high_count / total
    low_score = low_count / total


    if high_score > 0.003:
        return 'high_risk'
    elif low_score > 0.003:
        return 'low_risk'
    else:
        return 'medium_risk'
    

def load_and_preprocess():
    dataset = load_dataset(
        "winterForestStump/10-K_sec_filings",
        name="default",
        split="001",
        streaming=True,
        verification_mode="no_checks"
    )

    rows = []
    for i, row in enumerate(dataset):
        rows.append(row)
        if i >= 4999:
            break

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows")

    mda_col = "Management\u2019s Discussion and Analysis of Financial Condition and Results of Operations"

    df['combined_text'] = (
        df['Risk Factors'].fillna('') + ' ' + df['Business'].fillna('') + ' ' + df[mda_col].fillna('')
    )
    print("Cleaning text...")
    df['clean_text'] = df['combined_text'].apply(clean_text)

    print("Assigning labels...")
    df['label'] = df['clean_text'].apply(assign_label)

    df = df[['company_name', 'filing_date', 'clean_text', 'label']]

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/clean_data.csv', index=False)

    print(f"Rows saved: {len(df)}")
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    return df

if __name__ == "__main__":
    load_and_preprocess()



