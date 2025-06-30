#!/usr/bin/env python3
"""
Train all MCP model servers using datasets from /dataset directory
Based on spam email classification best practices
"""

import os
import pandas as pd
import numpy as np
import zipfile
import asyncio
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import sys

# Import our model servers to use their training functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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
                        
                        # Standardize column names
                        # Common patterns: 'text', 'email', 'message', 'content' for email body
                        # 'label', 'spam', 'class', 'category' for labels
                        
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
                                # Separate subject and body columns
                                standardized_df['content'] = df[content_col].astype(str)
                                standardized_df['subject'] = df[subject_col].astype(str)
                            else:
                                # Combined content - will extract subject later
                                standardized_df['content'] = df[content_col].astype(str)
                                standardized_df['subject'] = ''
                            
                            # Convert labels to binary (1 for spam, 0 for ham)
                            labels = df[label_col]
                            if labels.dtype == 'object':
                                # String labels
                                spam_indicators = ['spam', '1', 'yes', 'phishing']
                                standardized_df['is_spam'] = labels.str.lower().isin(spam_indicators).astype(int)
                            else:
                                # Numeric labels (0 = ham, 1 = spam)
                                standardized_df['is_spam'] = (labels == 1).astype(int)
                            
                            # Add sender if available
                            if 'sender' in df.columns:
                                standardized_df['sender'] = df['sender'].astype(str)
                            else:
                                standardized_df['sender'] = ''
                            
                            # Add source dataset
                            standardized_df['source'] = dataset_file.replace('.csv.zip', '')
                            
                            all_data.append(standardized_df)
                            print(f"  Loaded {len(standardized_df)} emails ({standardized_df['is_spam'].sum()} spam)")
                        else:
                            print(f"  Warning: Could not find required columns in {dataset_file}")
                            print(f"  Available columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"  Error loading {dataset_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal emails loaded: {len(combined_df)}")
        print(f"Spam emails: {combined_df['is_spam'].sum()} ({combined_df['is_spam'].mean():.1%})")
        print(f"Ham emails: {len(combined_df) - combined_df['is_spam'].sum()} ({1 - combined_df['is_spam'].mean():.1%})")
        return combined_df
    else:
        print("No datasets could be loaded!")
        return pd.DataFrame()


def preprocess_emails(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe to format expected by MCP servers"""
    samples = []
    
    for idx, row in df.iterrows():
        content = str(row['content'])
        
        # Handle subject
        if row.get('subject') and str(row['subject']).strip():
            subject = str(row['subject'])
            email_content = content
        else:
            # Try to extract subject from content if present
            lines = content.split('\n', 1)
            
            if len(lines) > 1 and lines[0].lower().startswith('subject:'):
                subject = lines[0].replace('Subject:', '').replace('subject:', '').strip()
                email_content = lines[1].strip()
            else:
                # Use first 50 chars as subject if not found
                subject = content[:50] + "..." if len(content) > 50 else content
                email_content = content
        
        # Handle sender
        if row.get('sender') and str(row['sender']).strip() and str(row['sender']) != 'nan':
            sender = str(row['sender'])
        else:
            # Generate synthetic sender based on spam/ham status
            if row['is_spam']:
                sender_domains = ['spam.com', 'phishing.net', 'suspicious.tk', 'alert.ml']
                sender = f"noreply@{np.random.choice(sender_domains)}"
            else:
                sender_domains = ['company.com', 'gmail.com', 'outlook.com', 'yahoo.com']
                sender = f"user{np.random.randint(1000, 9999)}@{np.random.choice(sender_domains)}"
        
        samples.append({
            'email_content': email_content,
            'subject': subject,
            'sender': sender,
            'is_spam': bool(row['is_spam'])
        })
    
    return samples


async def train_naive_bayes_model(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]]):
    """Train and save Naive Bayes model"""
    print("\n" + "="*60)
    print("Training Naive Bayes Model")
    print("="*60)
    
    from mcp_naive_bayes import train_naive_bayes, naive_bayes_model, preprocessor
    
    # Mock context object
    class MockContext:
        pass
    
    ctx = MockContext()
    
    # Train the model
    result = await train_naive_bayes(ctx, train_samples, model_type_choice="multinomial")
    
    if result['success']:
        print(f"✓ Training completed successfully")
        print(f"  Training accuracy: {result['training_accuracy']:.2%}")
        print(f"  Features: {result['feature_count']}")
        
        # Save the model
        with open('naive_bayes_model.pkl', 'wb') as f:
            pickle.dump({
                'model': naive_bayes_model,
                'preprocessor': preprocessor,
                'model_type': 'multinomial'
            }, f)
        print("✓ Model saved to naive_bayes_model.pkl")
    else:
        print(f"✗ Training failed: {result.get('error', 'Unknown error')}")


async def train_svm_model(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]]):
    """Train and save SVM model"""
    print("\n" + "="*60)
    print("Training SVM Model")
    print("="*60)
    
    from mcp_svm import train_svm, svm_model, preprocessor, scaler
    
    class MockContext:
        pass
    
    ctx = MockContext()
    
    # Train with RBF kernel
    result = await train_svm(ctx, train_samples, kernel="rbf", C=1.0, gamma="scale")
    
    if result['success']:
        print(f"✓ Training completed successfully")
        print(f"  Training accuracy: {result['training_accuracy']:.2%}")
        print(f"  Support vectors: {result['n_support_vectors']}")
        
        # Save the model
        with open('svm_model.pkl', 'wb') as f:
            pickle.dump({
                'model': svm_model,
                'preprocessor': preprocessor,
                'scaler': scaler
            }, f)
        print("✓ Model saved to svm_model.pkl")
    else:
        print(f"✗ Training failed: {result.get('error', 'Unknown error')}")


async def train_random_forest_model(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]]):
    """Train and save Random Forest model"""
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)
    
    from mcp_random_forest import train_random_forest, rf_model, preprocessor
    
    class MockContext:
        pass
    
    ctx = MockContext()
    
    # Train with 100 trees
    result = await train_random_forest(ctx, train_samples, n_estimators=100, max_depth=20)
    
    if result['success']:
        print(f"✓ Training completed successfully")
        print(f"  Training accuracy: {result['training_accuracy']:.2%}")
        print(f"  Number of trees: {result['n_estimators']}")
        
        # Save the model
        with open('random_forest_model.pkl', 'wb') as f:
            pickle.dump({
                'model': rf_model,
                'preprocessor': preprocessor
            }, f)
        print("✓ Model saved to random_forest_model.pkl")
    else:
        print(f"✗ Training failed: {result.get('error', 'Unknown error')}")


async def train_logistic_regression_model(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]]):
    """Train and save Logistic Regression model"""
    print("\n" + "="*60)
    print("Training Logistic Regression Model")
    print("="*60)
    
    from mcp_logistic_regression import train_logistic_regression, lr_model, preprocessor, scaler
    
    class MockContext:
        pass
    
    ctx = MockContext()
    
    # Train with L2 regularization
    result = await train_logistic_regression(ctx, train_samples, penalty="l2", C=1.0, solver="lbfgs")
    
    if result['success']:
        print(f"✓ Training completed successfully")
        print(f"  Training accuracy: {result['training_accuracy']:.2%}")
        print(f"  Converged: {result.get('converged', 'N/A')}")
        
        # Save the model
        with open('logistic_regression_model.pkl', 'wb') as f:
            pickle.dump({
                'model': lr_model,
                'preprocessor': preprocessor,
                'scaler': scaler
            }, f)
        print("✓ Model saved to logistic_regression_model.pkl")
    else:
        print(f"✗ Training failed: {result.get('error', 'Unknown error')}")


async def train_neural_network_model(train_samples: List[Dict[str, Any]], test_samples: List[Dict[str, Any]]):
    """Train and save Neural Network model"""
    print("\n" + "="*60)
    print("Training Neural Network Model")
    print("="*60)
    
    from mcp_neural_network import train_neural_network, nn_model, preprocessor, scaler
    
    class MockContext:
        pass
    
    ctx = MockContext()
    
    # Train with 3 hidden layers
    result = await train_neural_network(
        ctx, 
        train_samples, 
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate=0.001,
        max_iter=500
    )
    
    if result['success']:
        print(f"✓ Training completed successfully")
        print(f"  Training accuracy: {result['training_accuracy']:.2%}")
        print(f"  Total parameters: {result['network_info']['total_parameters']}")
        
        # Save the model
        with open('neural_network_model.pkl', 'wb') as f:
            pickle.dump({
                'model': nn_model,
                'preprocessor': preprocessor,
                'scaler': scaler
            }, f)
        print("✓ Model saved to neural_network_model.pkl")
    else:
        print(f"✗ Training failed: {result.get('error', 'Unknown error')}")


async def main():
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
    
    # Convert to samples format
    print("\n2. Preprocessing emails...")
    all_samples = preprocess_emails(df)
    print(f"Preprocessed {len(all_samples)} samples")
    
    # Split into train/test
    print("\n3. Splitting data...")
    train_samples, test_samples = train_test_split(
        all_samples, 
        test_size=0.2, 
        random_state=42,
        stratify=[s['is_spam'] for s in all_samples]
    )
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Limit training size for faster training (you can remove this for full training)
    if len(train_samples) > 5000:
        print(f"\nLimiting training set to 5000 samples for faster training...")
        train_samples = train_samples[:5000]
    
    # Train all models
    print("\n4. Training models...")
    
    await train_naive_bayes_model(train_samples, test_samples)
    await train_svm_model(train_samples, test_samples)
    await train_random_forest_model(train_samples, test_samples)
    await train_logistic_regression_model(train_samples, test_samples)
    await train_neural_network_model(train_samples, test_samples)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nAll models have been trained and saved.")
    print("The MCP servers will automatically load these pre-trained models when started.")
    print("\nTo use the trained models, start any MCP server:")
    print("  $ fastmcp dev mcp_naive_bayes.py")
    print("  $ fastmcp dev mcp_svm.py")
    print("  etc.")


if __name__ == "__main__":
    asyncio.run(main())