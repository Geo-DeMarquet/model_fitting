import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('linear_data.csv') #reads data from csv file

x = df[['X']].values #assigns x values to x variable
y = df['Y'].values #assigns y values to y variable

model = LinearRegression() #creates an instance of the LinearRegression class from sklearn.linear_model

model.fit(x, y) #fits the x and y variables to the model

y_pred = model.predict(x) #applies y = mx + c to calculate line of best fit

slope = model.coef_[0] #calculates coefficient using linear regression model
intercept = model.intercept_#calculates y intercept using linear regression model
equation = f"y = {slope:.2f}x + {intercept:.2f}" #puts these in y =mx + c formula

plt.text(x.min(), y.min() + 20, equation, fontsize=12, color='black') #assigns equation at top left of graph

plt.scatter(x, y, color='blue', label='Original data')
#plots original data points
plt.plot(x, y_pred, color='orange', label='Fitted line')
#plots line of best fit
plt.xlabel('X') #labels axes
plt.ylabel('Y')
plt.title('Linear Regression Fit') #gives plot a title
plt.legend() #gives plot a legend/key
plt.show()

slope = model.coef_[0] #calculates coefficient using linear regression model
intercept = model.intercept_#calculates y intercept using linear regression model
print(f"Equation of the fitted line: y = {slope:.2f}x + {intercept:.2f}") #prints in y=mx + c formula
