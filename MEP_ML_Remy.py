import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Generate synthetic data
np.random.seed(0)  # for reproducibility

data = {
    'Age': np.random.randint(20, 70, 100), 
    'Handedness': np.random.choice(['Left', 'Right'], 100),
    'Propofol_Dosage': np.random.uniform(5.0, 50.0, 100),
    'MEP_Amplitude': 3.0 - 0.05 * np.random.choice([1.5, 1], 100) * np.random.uniform(5.0, 50.0, 100) + np.random.normal(0, 0.1, 100)
}

df = pd.DataFrame(data)

# Data Overview
print(df.describe())
sns.pairplot(df, hue='Handedness')
plt.show()

# Regression Model
X = df[['Age', 'Handedness', 'Propofol_Dosage']]
y = df['MEP_Amplitude']

# One-hot encoding for the categorical variable 'Handedness'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Age', 'Propofol_Dosage']),
        ('cat', OneHotEncoder(drop='first'), ['Handedness'])
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
coefficients = pipeline.named_steps['classifier'].coef_

# Display Coefficients
features = ['Age', 'Propofol_Dosage', 'Handedness_Right']
for coef, feature in zip(coefficients, features):
    print(f"Coefficient for {feature}: {coef:.4f}")

# Visualize the effects
plt.figure(figsize=(12,6))
sns.boxplot(x='Handedness', y='MEP_Amplitude', data=df)
plt.title('Effect of Handedness on MEP Amplitude')
plt.show()

plt.figure(figsize=(12,6))
sns.regplot(x='Propofol_Dosage', y='MEP_Amplitude', data=df, line_kws={'color':'red'}, ci=None)
plt.title('Effect of Propofol Dosage on MEP Amplitude')
plt.show()

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Prediction vs Actual Visualization
plt.figure(figsize=(12,6))
sns.scatterplot(y_test, y_pred)
plt.xlabel('Actual MEP Amplitude')
plt.ylabel('Predicted MEP Amplitude')
plt.title('Actual vs Predicted MEP Amplitude')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # A reference line for perfect predictions
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Coefficient Heatmap
coef_df = pd.DataFrame(coefficients, index=features, columns=["Coefficient Value"])
plt.figure(figsize=(8, 4))
sns.heatmap(coef_df, annot=True, cmap='coolwarm', vmin=-0.1, vmax=0.1)
plt.title('Regression Coefficients')
plt.show()

# Decision Tree Regression for visualization
tree_reg = DecisionTreeRegressor(max_depth=3)  # restricting depth for visualization purpose
tree_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', tree_reg)
])
tree_pipeline.fit(X_train, y_train)

# Plotting the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_reg, filled=True, feature_names=features, rounded=True)
plt.title('Decision Tree Regression')
plt.show()

# Use this tree for prediction (just as an additional model)
y_tree_pred = tree_pipeline.predict(X_test)
mse_tree = mean_squared_error(y_test, y_tree_pred)
print(f"Mean Squared Error (Decision Tree): {mse_tree:.4f}")

