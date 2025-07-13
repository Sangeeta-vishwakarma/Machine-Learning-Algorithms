# Machine-Learning-Algorithms
This repository includes basic algorithms of machine learning. It contains juyter notebook, idle(python) and dataset(csv/excel format).

# Regression
## 1. Simple Linear Regression
This project demonstrates how to implement a Simple Linear Regression model using Python and the scikit-learn library. The goal is to predict salaries based on years of experience.

### Steps
- Importing Libraries
- Reading the Dataset
- Splitting the Dataset
- Training the Model
- Making Predictions
- Visualizing Results

### Key Learnings
- How to preprocess and split a dataset for machine learning.
- Understanding and applying Simple Linear Regression.
- Visualizing regression results using scatter plots and line graphs.
- Using scikit-learn for training and prediction.
- Importance of train/test split for model validation.

### Sample Output
The regression line fits a linear trend showing how salary increases with experience. The model generalizes well with clear predictions on test data.

## 2. Multiple Linear Regression
This notebook implements Multiple Linear Regression using the sklearn library. It demonstrates:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Model training
- Predictions
- Performance evaluation using common regression metrics

### Key Learnings:
- Pandas & NumPy are essential for handling structured data.
- Matplotlib & Seaborn are used for data visualization.
- scikit-learn provides out-of-the-box tools for model building and evaluation.
- Pairplot visualizes relationships between numeric features.
- Heatmap shows feature correlation. Helps in selecting impactful variables for regression.
- Coefficients tell how much the target variable changes with a unit change in the feature.
- Intercept is the predicted value when all features are zero.
- Mean Absolute Error (MAE): Average of absolute errors.
- Mean Squared Error (MSE): Punishes large errors more.
- Root Mean Squared Error (RMSE): Same unit as target; easy to interpret.

### Final Thought
- Multiple Linear Regression is a good baseline for regression problems.
- Always check multicollinearity and feature importance before deploying the model.
- Preprocessing, visualization, and interpretation are key in real-world applications.


## 3. Polynomial Linear Regression 
### Predicting Salaries Based on Position Level
This project demonstrates the application of Polynomial Regression to model and predict salaries based on position levels. While simple linear regression may not capture complex trends in data, polynomial regression introduces non-linearity, making it more suitable for datasets with curved relationships.

The goal is to predict an employee's salary based on their position level in an organization. Given a dataset of position levels and corresponding salaries, the objective is to build both Linear and Polynomial Regression models, compare their performance, and visualize the difference in predictions.

### Steps:
1. Importing Libraries and Dataset
2. Data Preparation
3. Model Building
   - Linear Regression Model:  Fitted a simple linear regression to the dataset to serve as a baseline.
   -  Polynomial Regression Model: Generated polynomial features up to degree 4.
     Fitted a linear regression model on these polynomial features.
4. Visualization
Plotted:
- Linear regression results (straight line) showing underfitting.
- Polynomial regression results (curved line) demonstrating a better fit to the data.
- High-resolution plots to visualize smooth polynomial curves for better interpretation.
  
5. Prediction
   Predicted the salary for a position level of 6.5

### Key Learnings
- Linear vs. Polynomial Regression:
  - Linear regression fails to capture the nonlinear nature of real-world data trends.
  - Polynomial regression, by adding powers of the features, provides flexibility and improved accuracy for nonlinear data.
- Model Selection:
  - Choosing the right degree of the polynomial is crucial — too low may underfit, too high may overfit.

- Visualization Importance:
  - Graphical comparisons between model fits provide intuitive understanding and assist in selecting the most appropriate model.
- Practical Usage:
  - Polynomial regression is ideal for problems where the relationship between variables is not strictly linear, such as growth curves, salary prediction based on experience, etc.

### Conclusion
This project effectively showcases the limitations of linear regression in capturing complex trends and how polynomial regression can overcome them. It also highlights the importance of feature engineering and model evaluation through visualization.


## 4. Support Vector Regression
This contains an implementation of Support Vector Regression (SVR) using the RBF kernel to model and predict salaries based on position levels in a company. Predict an employee's salary based on their position level using a Support Vector Machine (SVM) regression algorithm. This helps demonstrate how non-linear models can capture complex trends in small datasets.

### Steps
- Data Preprocessing
- Model Training
- Prediction
- Visualization

### Learning Outcomes
- Understanding how Support Vector Regression works
- Applying feature scaling correctly
- Visualizing regression results in 2D
- Importance of inverse transforming scaled results
  
## 5. Decision Tree Regression
This contains a Jupyter Notebook demonstrating the implementation of a Decision Tree Regression model using the scikit-learn library. 

### Dataset Used
The dataset used appears to be a Position Level Salary dataset, commonly used in regression tutorials.
It includes:
- Position (categorical)
- Level (numerical)
- Salary (target variable)

### Workflow Overview
1. Data Preprocessing
- Load dataset using pandas.
- Extract features (X) and target (y).
2. Model Training
- Fit a DecisionTreeRegressor on the dataset.
3. Prediction
- Predict salary for a specific level (e.g., 6.5).
4. Visualization
- Plot decision tree predictions over a high-resolution X_grid.

### Output
- The plot demonstrates how decision trees create stepwise prediction lines.
- Salary is predicted for non-integer levels, e.g., Level 6.5.
### Learnings
- Understanding how Decision Tree Regression works.
- Differences between linear and stepwise predictions.
- Importance of using high-resolution input for smooth visualization.

## 6. Random Forest Regression

# Classification
## 1. Logistic Regression
## 2. K-Nearest Neighbors( KNN )
## 3. Support Vector Machine
