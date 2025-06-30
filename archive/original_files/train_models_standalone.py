#!/usr/bin/env python3
"""
Standalone training script for all ML models using the datasets
"""

import os
import pandas as pd
import numpy as np
import zipfile
import pickle
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Import our preprocessor
from email_preprocessor import EmailPreprocessor


def load_datasets(dataset_dir: str = "./dataset") -> pd.DataFrame:
    """Load and combine all email datasets"""
    all_data = []
    
    dataset_files = [
        "SpamAssasin.csv.zip",
        "Enron.csv.zip",
        "Ling.csv.zip",
        "CEAS_08.csv.zip",
        "Nazario.csv.zip",
        "phishing_email.csv.zip"
    ]
    
    for dataset_file in dataset_files:
        file_path = os.path.join(dataset_dir, dataset_file)
        if os.path.exists(file_path):
            print(f"Loading {dataset_file}...")
            try:
                # Extract and read CSV
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    csv_filename = zip_ref.namelist()[0]
                    with zip_ref.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file)
                        
                        # Find email content column
                        content_cols = ['text', 'email', 'message', 'content', 'body', 'Body', 'Text', 'text_combined']
                        content_col = None
                        subject_col = None
                        
                        # Check for separate subject column
                        if 'subject' in df.columns:
                            subject_col = 'subject'
                        
                        for col in content_cols:
                            if col in df.columns:
                                content_col = col
                                break
                        
                        # Find label column
                        label_cols = ['label', 'spam', 'class', 'category', 'Label', 'Spam', 'Class']
                        label_col = None
                        for col in label_cols:
                            if col in df.columns:
                                label_col = col
                                break
                        
                        if content_col and label_col:
                            # Create standardized dataframe
                            standardized_df = pd.DataFrame()
                            
                            # Handle content and subject
                            if subject_col:
                                standardized_df['content'] = df[content_col].astype(str)
                                standardized_df['subject'] = df[subject_col].astype(str)
                            else:
                                standardized_df['content'] = df[content_col].astype(str)
                                standardized_df['subject'] = ''
                            
                            # Convert labels
                            labels = df[label_col]
                            if labels.dtype == 'object':
                                spam_indicators = ['spam', '1', 'yes', 'phishing']
                                standardized_df['is_spam'] = labels.str.lower().isin(spam_indicators).astype(int)
                            else:
                                standardized_df['is_spam'] = (labels == 1).astype(int)
                            
                            # Add sender if available
                            if 'sender' in df.columns:
                                standardized_df['sender'] = df['sender'].astype(str)
                            else:
                                standardized_df['sender'] = ''
                            
                            all_data.append(standardized_df)
                            print(f"  Loaded {len(standardized_df)} emails ({standardized_df['is_spam'].sum()} spam)")
                
            except Exception as e:
                print(f"  Error loading {dataset_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal emails loaded: {len(combined_df)}")
        print(f"Spam emails: {combined_df['is_spam'].sum()} ({combined_df['is_spam'].mean():.1%})")
        print(f"Ham emails: {len(combined_df) - combined_df['is_spam'].sum()} ({1 - combined_df['is_spam'].mean():.1%})")
        return combined_df
    else:
        return pd.DataFrame()


def prepare_training_data(df: pd.DataFrame, max_samples: int = 10000):
    """Prepare data for training"""
    # Sample data if too large
    if len(df) > max_samples:
        # Stratified sampling to maintain class balance
        df = df.groupby('is_spam').sample(n=max_samples//2, random_state=42)
    
    # Prepare emails and labels
    emails = []
    labels = []
    
    for idx, row in df.iterrows():
        content = str(row['content'])
        subject = str(row.get('subject', ''))
        sender = str(row.get('sender', 'unknown@unknown.com'))
        
        # Handle missing or nan values
        if subject == 'nan' or not subject:
            # Try to extract from content
            if content.lower().startswith('subject:'):
                lines = content.split('\n', 1)
                subject = lines[0].replace('subject:', '').strip()
                content = lines[1] if len(lines) > 1 else content
            else:
                subject = content[:50] + "..." if len(content) > 50 else content
        
        if sender == 'nan' or not sender:
            sender = 'noreply@unknown.com' if row['is_spam'] else 'user@company.com'
        
        emails.append((content, subject, sender))
        labels.append(row['is_spam'])
    
    return emails, labels


def train_naive_bayes(emails, labels, test_emails, test_labels):
    """Train Naive Bayes model"""
    print("\n" + "="*60)
    print("Training Naive Bayes Model")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = EmailPreprocessor(max_features=5000)
    preprocessor.fit(emails, labels)
    
    # Transform training data
    X_train = []
    for content, subject, sender in emails:
        features = preprocessor.transform(content, subject, sender)
        X_train.append(features['tfidf'])
    
    X_train = np.array(X_train)
    y_train = np.array(labels)
    
    # Train model
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    X_test = []
    for content, subject, sender in test_emails:
        features = preprocessor.transform(content, subject, sender)
        X_test.append(features['tfidf'])
    
    X_test = np.array(X_test)
    y_test = np.array(test_labels)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✓ Training completed")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Save model
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor,
            'model_type': 'multinomial'
        }, f)
    print("✓ Model saved to naive_bayes_model.pkl")
    
    return accuracy


def train_svm(emails, labels, test_emails, test_labels):
    """Train SVM model"""
    print("\n" + "="*60)
    print("Training SVM Model")
    print("="*60)
    
    # Initialize preprocessor and scaler
    preprocessor = EmailPreprocessor(max_features=2000)
    preprocessor.fit(emails, labels)
    scaler = StandardScaler()
    
    # Transform training data
    X_train = []
    for content, subject, sender in emails:
        features = preprocessor.transform(content, subject, sender)
        X_train.append(features['combined'])
    
    X_train = np.array(X_train)
    X_train = scaler.fit_transform(X_train)
    y_train = np.array(labels)
    
    # Train model
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    X_test = []
    for content, subject, sender in test_emails:
        features = preprocessor.transform(content, subject, sender)
        X_test.append(features['combined'])
    
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = np.array(test_labels)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✓ Training completed")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Support vectors: {np.sum(model.n_support_)}")
    
    # Save model
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor,
            'scaler': scaler
        }, f)
    print("✓ Model saved to svm_model.pkl")
    
    return accuracy


def train_random_forest(emails, labels, test_emails, test_labels):
    """Train Random Forest model"""
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = EmailPreprocessor(max_features=3000)
    preprocessor.fit(emails, labels)
    
    # Transform training data
    X_train = []
    for content, subject, sender in emails:
        features = preprocessor.transform(content, subject, sender)
        X_train.append(features['combined'])
    
    X_train = np.array(X_train)
    y_train = np.array(labels)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    X_test = []
    for content, subject, sender in test_emails:
        features = preprocessor.transform(content, subject, sender)
        X_test.append(features['combined'])
    
    X_test = np.array(X_test)
    y_test = np.array(test_labels)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✓ Training completed")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Number of trees: {model.n_estimators}")
    
    # Save model
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor
        }, f)
    print("✓ Model saved to random_forest_model.pkl")
    
    return accuracy


def train_logistic_regression(emails, labels, test_emails, test_labels):
    """Train Logistic Regression model"""
    print("\n" + "="*60)
    print("Training Logistic Regression Model")
    print("="*60)
    
    # Initialize preprocessor and scaler
    preprocessor = EmailPreprocessor(max_features=3000)
    preprocessor.fit(emails, labels)
    scaler = StandardScaler()
    
    # Transform training data
    X_train = []
    for content, subject, sender in emails:
        features = preprocessor.transform(content, subject, sender)
        X_train.append(features['combined'])
    
    X_train = np.array(X_train)
    X_train = scaler.fit_transform(X_train)
    y_train = np.array(labels)
    
    # Train model
    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    X_test = []
    for content, subject, sender in test_emails:
        features = preprocessor.transform(content, subject, sender)
        X_test.append(features['combined'])
    
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = np.array(test_labels)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✓ Training completed")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Converged: {model.n_iter_ < model.max_iter}")
    
    # Save model
    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor,
            'scaler': scaler
        }, f)
    print("✓ Model saved to logistic_regression_model.pkl")
    
    return accuracy


def train_neural_network(emails, labels, test_emails, test_labels):
    """Train Neural Network model"""
    print("\n" + "="*60)
    print("Training Neural Network Model")
    print("="*60)
    
    # Initialize preprocessor and scaler
    preprocessor = EmailPreprocessor(max_features=2000)
    preprocessor.fit(emails, labels)
    scaler = StandardScaler()
    
    # Transform training data
    X_train = []
    for content, subject, sender in emails:
        features = preprocessor.transform(content, subject, sender)
        X_train.append(features['combined'])
    
    X_train = np.array(X_train)
    X_train = scaler.fit_transform(X_train)
    y_train = np.array(labels)
    
    # Train model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    X_test = []
    for content, subject, sender in test_emails:
        features = preprocessor.transform(content, subject, sender)
        X_test.append(features['combined'])
    
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = np.array(test_labels)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✓ Training completed")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Iterations: {model.n_iter_}")
    
    # Save model
    with open('neural_network_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor,
            'scaler': scaler
        }, f)
    print("✓ Model saved to neural_network_model.pkl")
    
    return accuracy


def main():
    """Main training pipeline"""
    print("="*60)
    print("MCP Model Training Pipeline")
    print("="*60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    df = load_datasets()
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Prepare data
    print("\n2. Preparing training data...")
    emails, labels = prepare_training_data(df, max_samples=10000)
    print(f"Prepared {len(emails)} samples")
    
    # Split data
    print("\n3. Splitting data...")
    train_emails, test_emails, train_labels, test_labels = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training samples: {len(train_emails)}")
    print(f"Test samples: {len(test_emails)}")
    
    # Train all models
    print("\n4. Training models...")
    
    results = {}
    results['naive_bayes'] = train_naive_bayes(train_emails, train_labels, test_emails, test_labels)
    results['svm'] = train_svm(train_emails, train_labels, test_emails, test_labels)
    results['random_forest'] = train_random_forest(train_emails, train_labels, test_emails, test_labels)
    results['logistic_regression'] = train_logistic_regression(train_emails, train_labels, test_emails, test_labels)
    results['neural_network'] = train_neural_network(train_emails, train_labels, test_emails, test_labels)
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nModel Performance Summary:")
    for model, accuracy in results.items():
        print(f"  {model}: {accuracy:.2%}")
    
    print("\nAll models have been trained and saved.")
    print("The MCP servers will automatically load these pre-trained models.")


if __name__ == "__main__":
    main()