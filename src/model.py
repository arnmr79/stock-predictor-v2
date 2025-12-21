import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def prepare_data(df):
    """
    Prepares data for training.
    Target = 1 if Tomorrow Close > Today Close, else 0.
    """
    df = df.copy()
    # Create Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaNs created by indicators or shift
    df = df.dropna()
    return df

def train_model(X, y):
    """
    Trains a Random Forest Classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """
    Evaluates the model.
    """
    preds = model.predict(X)
    print("Classification Report:\n", classification_report(y, preds))
    return accuracy_score(y, preds)
