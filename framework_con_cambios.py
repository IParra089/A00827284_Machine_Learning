# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#Read dataset
df = pd.read_csv('MC.csv')
#Create new dataset with relevant variables
df.columns=["calories","fat"] #Changing the names for simplicity

#split into x and y
x=df.fat
y=df.calories


# IQR
Q1 = df.calories.quantile(0.25)
Q3 = df.calories.quantile(0.75)
IQR=Q3-Q1
 
print("Old Shape: ", df.shape)
 
# Upper bound
upper = np.where(df.calories >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df.calories <= (Q1-1.5*IQR))
 
''' Removing the Outliers '''
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
 
print("New Shape: ", df.shape)

#Standard Scaler
scaler=StandardScaler()
scaler.fit(df)
x=scaler.transform(df)

y=df.calories
#splitting data into train and test sets (80-20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2



#Variable for linear regression
model=LinearRegression()

#Training the model with the train sets
#x_train=np.array(x_train).reshape(-1,1) #Reshaping due to an error, for use without scaler
model.fit(x_train, y_train)
print("Coefficients: \n", model.coef_)

#Prediction on train
y_hat_train=model.predict(x_train)

#Calculate metrics
mse_train=mean_squared_error(y_hat_train,y_train)
mae_train=mean_absolute_error(y_hat_train,y_train)
r2_train=r2_score(y_hat_train,y_train)
var_train=np.var(y_hat_train)
sse_train= np.mean((np.mean(y_hat_train) - y)** 2) # Where y refers to dependent variable. # sse : Sum of squared errors.
bias_train = sse_train - var_train

print("MSE=", mse_train)
print("RMSE=", np.sqrt(mse_train))
print("MAE=", mae_train)
print("R2=" ,r2_train)
print("Varianza=",var_train)
print("Bias=",bias_train)

#Prediction on test
#x_test=np.array(x_test).reshape(-1,1)
y_hat_test=model.predict(x_test)

#Calculate metrics
mse_test=mean_squared_error(y_hat_test,y_test)
mae_test=mean_absolute_error(y_hat_test,y_test)
r2_test=r2_score(y_hat_test,y_test)
var_test=np.var(y_hat_test)
sse_test= np.mean((np.mean(y_hat_test) - y)** 2) # Where y refers to dependent variable. # sse : Sum of squared errors.
bias_test = sse_test - var_test

print("MSE=", mse_test)
print("RMSE=", np.sqrt(mse_test))
print("MAE=", mae_test)
print("R2=" ,r2_test)
print("Varianza=",var_test)
print("Bias=",bias_test)

#Prediction on validation
#x_val=np.array(x_val).reshape(-1,1)
y_hat_val=model.predict(x_val)

#Calculate metrics
mse_val=mean_squared_error(y_hat_val,y_val)
mae_val=mean_absolute_error(y_hat_val,y_val)
r2_val=r2_score(y_hat_val,y_val)
var_val=np.var(y_hat_val)
sse_val= np.mean((np.mean(y_hat_val) - y)** 2) # Where y refers to dependent variable. # sse : Sum of squared errors.
bias_val = sse_val - var_val

print("MSE=", mse_val)
print("RMSE=", np.sqrt(mse_val))
print("MAE=", mae_val)
print("R2=" ,r2_val)
print("Varianza=",var_val)
print("Bias=",bias_val)



plt.figure()
plt.scatter(x,y)
plt.plot(x_test,y_hat_test,color="black")
plt.xlabel("Total fat")
plt.ylabel("Calories")
plt.title("Modelo de regresión")
plt.show()

plt.figure()
plt.scatter(x_train,y_train,label="data")
plt.xlabel("Total fat")
plt.ylabel("Calories")
plt.title("Train set")
plt.show()



plt.figure()
plt.scatter(x_test,y_test,label="data")
plt.xlabel("Total fat")
plt.ylabel("Calories")
plt.title("Test set")
plt.show()

plt.figure()
plt.scatter(x_val,y_val,label="data")
plt.xlabel("Total fat")
plt.ylabel("Calories")
plt.title("Validation set")
plt.show()

sse= np.mean((np.mean(prediction) - y)** 2) # Where y refers to dependent variable. # sse : Sum of squared errors.

bias = sse - variance