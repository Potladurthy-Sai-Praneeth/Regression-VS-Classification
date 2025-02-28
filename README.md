# Comparative Analysis of Regression and Classification for Rating Prediction

## Overview
This project explores the effectiveness of regression and classification approaches in predicting discrete user ratings. Using the Rent the Runway dataset, we analyze multiple models to determine the best-suited technique for forecasting ratings, considering evaluation metrics such as accuracy, F1 score, MSE, RMSE, and MAE.

## Dataset
- **Source**: Rent the Runway dataset
- **Size**: 192,462 interactions, 105,508 users, 5,850 items
- **Features**:
  - User attributes: age, height, weight, body type, rented for, review text, etc.
  - Item attributes: fit, rating, size, category, review date, etc.
- **Preprocessing**:
  - Imputation of missing values with median/mode where applicable.
  - Addressing class imbalance without distorting regression-based results.
  - Feature engineering:
    - **BMI Calculation**: A combination of weight and height to provide a more meaningful user representation.
    - **Review Polarity Score**: Sentiment analysis of review text using NLP techniques.
    - **Temporal Features**: Extracting year and month from review date to identify trends over time.

## Models Implemented
### Regression-Based Models
1. **Heuristic Cosine Similarity** (Baseline) - Computes similarity between users and items based on feature vectors.
2. **Factorization Machines (FM)** - Captures interactions between users and items effectively.
3. **Shallow Neural Network** - A simple multi-layer perceptron for learning non-linear relationships.
4. **Random Forest Regressor** - Ensemble learning model based on decision trees.
5. **Decision Tree Regressor** - Simple tree-based model for non-linear predictions.
6. **XGBoost Regressor** - Gradient boosting model optimized for structured data.

### Classification-Based Models
1. **Factorization Machines (FM) Classifier** - Adapts FM for categorical predictions.
2. **Shallow Neural Network Classifier** - Utilizes softmax activation for class predictions.
3. **Random Forest Classifier** - Uses ensemble decision trees for classification.
4. **Decision Tree Classifier** - Basic decision tree model.
5. **XGBoost Classifier** - Gradient-boosted trees for classification.

## Evaluation Metrics
- **Regression**:
  - **MSE (Mean Squared Error)**: Measures the average squared difference between actual and predicted ratings.
  - **RMSE (Root Mean Squared Error)**: Evaluates the square root of MSE to reduce error amplification.
  - **MAE (Mean Absolute Error)**: Captures absolute differences between actual and predicted ratings.
- **Classification**:
  - **Accuracy**: Measures overall correctness of predictions.
  - **F1 Score**: Balances precision and recall, useful for imbalanced datasets.

## Key Findings
- **Factorization Machines** performed best among regression models, capturing user-item interactions effectively.
- **Shallow Neural Network** achieved the highest accuracy in classification but struggled with imbalanced data.
- **Regression approaches** were more reliable for rating prediction due to the discrete nature of ratings.
- **RMSE** was the most suitable metric for evaluating model performance, minimizing large prediction errors.

## Future Work
- Address class imbalance in classification models using advanced resampling techniques.
- Implement regularization techniques such as dropout and L1/L2 penalties to improve model generalization.
- Experiment with deeper neural networks and transformer-based architectures for enhanced performance.
- Fine-tune hyperparameters for boosting models (e.g., XGBoost) to optimize predictive accuracy.
