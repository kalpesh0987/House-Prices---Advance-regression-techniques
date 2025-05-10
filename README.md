# House Price Prediction using Regression and Ensemble Models

This project focuses on building and evaluating multiple regression models to predict house prices. The work involves model experimentation, hyperparameter tuning, feature selection, ensemble modeling, and performance evaluation using real-world datasets.

## Objective

The goal of this project is to develop a machine learning pipeline that accurately predicts house prices. This involves:

- Implementing various regression algorithms
- Performing hyperparameter tuning to optimize model performance
- Applying feature selection techniques to reduce overfitting and enhance interpretability
- Using ensemble learning (stacking) to combine model strengths
- Evaluating model performance using appropriate regression metrics

## Dataset Overview

The dataset includes various features related to housing properties, such as lot size, number of rooms, neighborhood, year built, and more. The target variable is `SalePrice`.

Preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Scaling and normalization (if required)
- Splitting the dataset into training and test sets

## Models Implemented

### Regression Algorithms

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Ensemble Learning

A one-layer stacking ensemble model was developed using the three base regressors above. The meta-model was trained on the outputs of these base models to improve predictive performance.

## Hyperparameter Tuning

Hyperparameter optimization was performed using:
- GridSearchCV
- RandomizedSearchCV

Parameters tuned include:
- `max_depth`
- `min_samples_split`
- `n_estimators`
- `learning_rate`
- `criterion`

These tuning strategies helped balance model complexity and computation time.

## Feature Selection Techniques (Bonus Exercise)

Three advanced feature selection methods were explored:

1. **SelectFromModel**  
   A model-based approach that selects features with high importance scores.

2. **SequentialFeatureSelector**  
   A forward search strategy that adds features one at a time based on performance improvement.

3. **GeneticSelectionCV**  
   Uses genetic algorithms to evolve optimal feature subsets over multiple generations.

These techniques reduced dimensionality, enhanced performance, and improved model interpretability.

## Evaluation Metrics

Models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Cross-validation scores

Final model predictions were also submitted to Kaggle for external validation.

## Key Learnings

- Stacking ensemble models provided better performance than individual models.
- Hyperparameter tuning improved the accuracy and robustness of the models.
- Feature selection helped avoid overfitting and simplified the model.
- Understanding the trade-offs between different algorithms and tuning strategies was essential for building an optimal pipeline.
