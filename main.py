# src/main.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Data Aggregation and Integration
class DataAggregator:
    def __init__(self, data_sources):
        self.data_sources = data_sources
    
    def load_data(self):
        """Load and concatenate data from multiple CSV files."""
        data_frames = [pd.read_csv(source) for source in self.data_sources]
        return pd.concat(data_frames, ignore_index=True)

    def clean_data(self, df):
        """Clean and standardize the data."""
        df.dropna(inplace=True)
        df['booking_date'] = pd.to_datetime(df['booking_date'])
        return df

# Machine Learning Models
class CustomerSegmentation:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def segment_customers(self, df):
        """Segment customers using K-Means clustering."""
        kmeans = KMeans(n_clusters=self.n_clusters)
        df['segment'] = kmeans.fit_predict(df[['age', 'travel_frequency', 'spending_score']])
        return df

class PredictiveAnalytics:
    def __init__(self):
        self.model = RandomForestRegressor()
    
    def train_model(self, X_train, y_train):
        """Train the predictive model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

# Natural Language Processing (NLP)
class SentimentAnalysis:
    def __init__(self):
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, comments):
        """Analyze sentiment of customer feedback."""
        return [self.sia.polarity_scores(comment) for comment in comments]

# Real-Time Personalization
class RealTimePersonalization:
    def personalize_content(self, user_data):
        """Generate personalized content based on user data."""
        personalized_content = f"Welcome back! Based on your previous travels to {user_data['last_destination']}, we recommend..."
        return personalized_content

def main():
    # Step 1: Data Aggregation and Integration
    data_sources = ['data/passenger_profiles.csv', 
                    'data/booking_history.csv', 
                    'data/flight_info.csv']
    
    aggregator = DataAggregator(data_sources)
    raw_data = aggregator.load_data()
    cleaned_data = aggregator.clean_data(raw_data)

    # Step 2: Customer Segmentation
    segmentation_model = CustomerSegmentation(n_clusters=5)
    segmented_data = segmentation_model.segment_customers(cleaned_data)

    # Step 3: Predictive Analytics
    X = segmented_data[['age', 'travel_frequency', 'spending_score']]
    y = segmented_data['future_spending']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    predictive_model = PredictiveAnalytics()
    predictive_model.train_model(X_train, y_train)
    
    predictions = predictive_model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')

    # Step 4: Sentiment Analysis on Feedback
    feedback_comments = ["Great service!", "Flight was delayed.", "Loved the lounge access!"]
    sentiment_analyzer = SentimentAnalysis()
    sentiments = sentiment_analyzer.analyze_sentiment(feedback_comments)
    
    print(sentiments)

if __name__ == "__main__":
    main()