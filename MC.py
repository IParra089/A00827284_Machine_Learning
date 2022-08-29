import pandas as pd
import numpy as np

def coeff(xtrain,ytrain):
    #calculate mean values
    xmean=xtrain.mean()
    ymean=ytrain.mean()
    
    #Least squares method
    #calculate m
    m= sum((xtrain-xmean)*(ytrain-ymean))/sum(((xtrain-xmean)**2))
    
    #calculate b
    b= (ymean-(m*xmean))
    
    return m, b, xmean, ymean

def lin_reg(data):
    
    #Read data set
    df = pd.read_csv("MC.csv")
    df.columns=["calories","fat"] #Changing the names for simplicity

    #split into x and y
    x=df.fat
    y=df.calories

    #split into train and test using pandas
    xtrain=x.sample(frac=0.8, random_state=300)
    xtest=x.drop(xtrain.index)
    ytrain=y.sample(frac=0.8, random_state=300)
    ytest=y.drop(xtrain.index)

    #Convert to numpy array
    xtrain=xtrain.to_numpy()
    xtest=xtest.to_numpy()
    ytrain=ytrain.to_numpy()
    ytest=ytest.to_numpy()
    
    #obtain variables from coeff function
    mb=coeff(xtrain, ytrain)
    m=mb[0]
    b=mb[1]
    xmean=mb[2]
    ymean=mb[3]
    
    print(f"The equation is: Y = {m} * x + {b}")
    
    #Prediction on train and MSE
    y_pred_train=(m*xtrain)+b
    MSE_train= sum((ytrain-y_pred_train)**2)/len(ytrain)

    #Prediction on test and MSE
    y_pred_test=(m*xtest)+b
    MSE_test=sum((ytest-y_pred_test)**2)/len(ytest)
    
    print(f"Train MSE={MSE_train}, Test MSE={MSE_test}")
    
    #calculate R^2
    R2 = sum((y_pred_train-ymean)**2)/sum((ytrain-ymean)**2)
    print(f"R^2 = {R2}")
    
    return m, b, MSE_train, MSE_test, R2
    
    
    

lin_reg("MC.csv")
