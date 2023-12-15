# Simple Linear Regression Example
# Dataset found on Kaggle: https://www.kaggle.com/datasets/himanshunakrani/student-study-hours/

# Objective: Build linear regression model to examine the affect of hours studied has on grade received in student scores.

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Upload dataset
df = pd.read_csv('score_updated.csv')

# Explore dataset
df.info()
df.describe()

# View top 5 rows
print(df.head())

# Check for outliers in boxplot
df.boxplot(column=['Scores'])

# Plot data in Seaborn & Matplotlib
sns.pairplot(df,x_vars=['Hours'],y_vars=['Scores'],height=7,kind='scatter')
plt.xlabel('Study Hours')
plt.ylabel('Student Scores')
plt.title('Student Score Prediction')
plt.show()

# Plot data in Matplotlib only
df.plot(x='Hours', y='Scores', style='o')
plt.title('Student Score Prediction')
plt.xlabel('Hours Studied')
plt.ylabel('Student Scores')
plt.show()

# Define predictor variable
X = df[['Hours']]
print(X.head())
print(type(X))

# Define response variable
y = df['Scores']
print(y.head())
print(type(y))

# Model building

# Split the data for train and test 
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.2,random_state=0)

# Fitting the model
lr = LinearRegression()
lr.fit(X_train,y_train)

# Predicting the scores for the test values
y_pred = lr.predict(X_test)

# Plotting the actual vs predicted values

c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Scores')
plt.ylabel('index')
plt.title('Prediction')
plt.show()

# Plotting the error
c = [i for i in range(1,len(y_test)+1,1)]
plt.plot(c,y_test-y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()

# Accuracy metrics from scikit learn
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r square :' , metrics.r2_score(y_test, y_pred))

# Plot actual and predicted values
plt.figure(figsize=(12,6))
plt.scatter(y_test,y_pred, color='r',linestyle='-')
plt.title('Actual and Predicted Scores')
plt.show()

# Intercept and coefficient of the line
print('Intercept of the model:',lr.intercept_)
print('Coefficient of the line:',lr.coef_)