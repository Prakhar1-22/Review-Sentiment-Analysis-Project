import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from utils import clean_text, preprocess_nlp

warnings.filterwarnings('ignore', category=FutureWarning)

# Create models folder if not exists
if not os.path.exists('models'):
    os.makedirs('models')

print("Loading dataset...")
df = pd.read_csv('data/amazonreviews.csv', on_bad_lines='skip', engine='python')

df = df.dropna(subset=['reviews.rating', 'reviews.text'])
df['cleaned_text'] = df['reviews.text'].apply(clean_text)
df['processed_text'] = df['cleaned_text'].apply(preprocess_nlp)

def map_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating <= 2:
        return 'Negative'
    return None

df['sentiment'] = df['reviews.rating'].apply(map_sentiment)
df = df[df['sentiment'].notna()]
print('Sentiment distribution:')
print(df['sentiment'].value_counts())

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), min_df=1, stop_words='english')
X = tfidf.fit_transform(df['processed_text'])

X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])
model = LogisticRegression(class_weight='balanced', max_iter=3000, solver='liblinear')
model.fit(X_train, y_train)

# Evaluate Sentiment Model
y_pred = model.predict(X_test)
print(f"Sentiment Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Train Rating Model
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, df['reviews.rating'], test_size=0.2, random_state=42)
rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
rating_model.fit(X_train_r, y_train_r)

# Evaluate Rating Model
y_pred_r = rating_model.predict(X_test_r)
print(f"Rating Model MAE: {mean_absolute_error(y_test_r, y_pred_r):.2f}")
print(f"Rating Model RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.2f}")

print("Saving models...")
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/rating_model.pkl', 'wb') as f:
    pickle.dump(rating_model, f)
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("All models trained and saved successfully!")