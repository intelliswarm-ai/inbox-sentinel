# Trained Phishing Detection Models

All MCP servers have been successfully trained on real-world spam and phishing email datasets.

## Training Data

The models were trained on **161,640 emails** from 6 different datasets:
- SpamAssassin Dataset (5,809 emails)
- Enron Spam Dataset (29,767 emails)
- Ling Spam Dataset (2,859 emails)
- CEAS 2008 Dataset (39,154 emails)
- Nazario Phishing Dataset (1,565 emails)
- Phishing Email Dataset (82,486 emails)

**Distribution**: 51% spam/phishing, 49% legitimate emails

## Model Performance

| Model | Test Accuracy | Training Samples | Key Characteristics |
|-------|--------------|------------------|-------------------|
| **Neural Network** | 96.60% | 8,000 | Best overall performance, 3-layer architecture |
| **Naive Bayes** | 96.25% | 8,000 | Fast, interpretable, good for text |
| **SVM** | 95.75% | 8,000 | 3,882 support vectors, RBF kernel |
| **Logistic Regression** | 95.75% | 8,000 | Linear model, highly interpretable |
| **Random Forest** | 93.95% | 8,000 | 100 trees, handles non-linear patterns |

## Using the Trained Models

All MCP servers automatically load pre-trained models when started:

```bash
# Start any server - it will load the pre-trained model
fastmcp dev mcp_naive_bayes.py
fastmcp dev mcp_svm.py
fastmcp dev mcp_random_forest.py
fastmcp dev mcp_logistic_regression.py
fastmcp dev mcp_neural_network.py
```

## Model Files

The trained models are saved as pickle files:
- `naive_bayes_model.pkl` (491 KB)
- `svm_model.pkl` (62.8 MB)
- `random_forest_model.pkl` (3.4 MB)
- `logistic_regression_model.pkl` (294 KB)
- `neural_network_model.pkl` (6.6 MB)

## Features Used

Each model uses sophisticated feature extraction:
- **TF-IDF vectorization** (up to 5,000 features)
- **Manual feature extraction**:
  - URL patterns and shorteners
  - Sender domain analysis
  - Content length and structure
  - Phishing keywords and urgency phrases
  - Special character usage
  - Capital letter ratios

## Retraining Models

To retrain with new data:
```bash
python train_models_standalone.py
```

This will:
1. Load all datasets from `/dataset`
2. Preprocess and balance the data
3. Train all 5 models
4. Save updated model files
5. Display performance metrics

## Verification

To verify models are working correctly:
```bash
python verify_trained_models.py
```

This confirms each model can:
- Load successfully
- Make predictions
- Achieve expected accuracy

## Privacy Note

All training and inference is performed locally. No email data is sent to external services.