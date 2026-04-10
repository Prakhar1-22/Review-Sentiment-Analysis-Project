# Amazon Sentiment Analysis Project

This project provides an automated feedback and sentiment analysis system for Amazon reviews using machine learning.

## Project Structure
- `app.py`: Streamlit web application for interactive analysis
- `train_models.py`: Script to train the ML models
- `utils.py`: Utility functions for text preprocessing
- `data/`: Contains the dataset
- `models/`: Saved trained models (generated after training)

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk streamlit
   ```

2. **Download NLTK Data**
   The script will automatically download required NLTK data, but you can do it manually:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('punkt_tab')
   ```

3. **Train the Models**
   Run the training script:
   ```bash
   python train_models.py
   ```
   This will:
   - Load and preprocess the Amazon reviews dataset
   - Train a sentiment classification model (Positive/Neutral/Negative)
   - Train a rating prediction model (1-5 stars)
   - Save the models to the `models/` folder

   Note: Training may take 10-30 minutes depending on your hardware.

4. **Run the Interactive App**
   ```bash
   streamlit run app.py
   ```
   This launches a web interface where you can:
   - Enter customer reviews
   - Get real-time sentiment analysis
   - See predicted ratings
   - Receive AI suggestions for negative feedback

## Features

- **Sentiment Analysis**: Classifies reviews as Positive, Neutral, or Negative
- **Rating Prediction**: Predicts star ratings (1-5) based on text
- **Text Preprocessing**: Includes cleaning, tokenization, stopword removal, and lemmatization
- **Interactive UI**: User-friendly web interface built with Streamlit
- **Model Calibration**: Uses calibrated classifiers for probability estimates

## Model Details

- **Vectorizer**: TF-IDF with up to 10,000 features, n-grams (1-3), min_df=2
- **Sentiment Model**: LinearSVC with hyperparameter tuning and calibration
- **Rating Model**: RandomForest Regressor
- **Evaluation**: Accuracy ~85-90% for sentiment, MAE ~0.5-0.7 for ratings

## Usage

1. Train models once
2. Run the Streamlit app
3. Enter review text in the text area
4. Click "Run Analysis" to see results

For development, you can modify the code in VS Code and rerun as needed.
=======
# Review-Sentiment-Analysis-Project
End-to-end NLP pipeline that classifies Amazon product review sentiment (Positive/Negative) and predicts star ratings using TF-IDF + Logistic Regression + Random Forest — deployed as an interactive Streamlit web app.

