# 🧠 Insurance Fraud Detection Using Decision Trees and Random Forests

This project explores the application of regression and ensemble learning techniques—specifically Decision Trees, Random Forests, and Gradient Boosting—to detect fraudulent patterns in insurance data. Additionally, advanced feature selection techniques and stacked ensemble models are employed to enhance model performance and generalization.

## 📌 Overview

- **Topic**: Experimentation with Regression, Ensemble Learning, and Feature Selection  
- **Bonus**: Feature Selection using Genetic Algorithms, SelectFromModel, and Sequential Feature Selector

## 🚀 Objectives

- Implement and compare multiple regression models
- Optimize model performance using hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Explore model-based and algorithmic feature selection methods
- Apply ensemble learning through stacking
- Evaluate model performance using metrics such as RMSE, R², and Kaggle leaderboard scores

## 🧰 Technologies Used

- Python (Jupyter Notebook)
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `xgboost`, `mlxtend`, `deap`
- Scikit-learn based models: DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor
- Hyperparameter tuning: GridSearchCV, RandomizedSearchCV
- Feature selection: SelectFromModel, SequentialFeatureSelector, GeneticSelectionCV

## 🔍 Key Concepts

### 1. Regression Models Used
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Stacked Model (Ensemble)**

### 2. Feature Selection (Bonus)
- **SelectFromModel**: Selects features based on model importance scores
- **SequentialFeatureSelector**: Greedy forward selection strategy
- **GeneticSelectionCV**: Uses evolutionary algorithms for optimal subset selection

### 3. Ensemble Learning
- **One-layer Stacking**:
  - Base Learners: Decision Tree, Random Forest, Gradient Boosting
  - Meta Learner: Regressor (trained on outputs of base models)
- **Performance Optimization** via cross-validation and hyperparameter tuning

## 📈 Performance Metrics

- RMSE (Root Mean Squared Error)
- R² Score

## 📊 Highlights

- Demonstrated effective use of feature selection to reduce dimensionality and improve performance
- Leveraged ensemble techniques to boost accuracy and robustness
- Achieved significant improvements in AUC and prediction accuracy compared to prior assignments

## 🧠 Key Learnings

- Applied advanced techniques like SMOTE, stacking, and evolutionary feature selection
- Balanced model interpretability with performance via tuning and feature reduction
- Gained deeper insights into ensemble synergy and feature importance in high-dimensional datasets

## 🏁 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/kalpesh0987/House-Prices---Advance-regression-techniques
   cd insurance-fraud-detection
