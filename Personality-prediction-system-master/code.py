import pandas as pd
from numpy import *
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors

data =pd.read_csv('train dataset.csv')    #reading the data from train dataset.csv


array = data.values     # this is used to accesing the data in train dataset

for i in range(len(array)):   # This is used for binary labelling(String to binary numbers)
	if array[i][0]=="Male":
		array[i][0]=1
	else:
		array[i][0]=0


df=pd.DataFrame(array)  #for simplicity we are converting the data into array

maindf =df[[0,1,2,3,4,5,6]]
mainarray=maindf.values    # accessing the values from the dataframe or using head function
print (mainarray)


temp=df[7]
train_y =temp.values    # accessing the values from the train dataset
# print(train_y)
# print(mainarray)
train_y=temp.values         # accessing the values from the train dataset

for i in range(len(train_y)):     # for loop is used to interate over all the values
	train_y[i] =str(train_y[i])


# This is where we create the Logistic Regresssion Model
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)  #These are hyperparameters which work the best for Logistic regression
mul_lr.fit(mainarray, train_y)  # Here the training process starts

# Here the name of our model is 'mul_lur'

testdata =pd.read_csv('test dataset.csv')  # Read the test csv file from the directory
test = testdata.values

for i in range(len(test)):   # Binary labelling of string parameters/attributes
	if test[i][0]=="Male":
		test[i][0]=1
	else:
		test[i][0]=0


df1=pd.DataFrame(test)  # Getting the dataframe from the csv values (cdv -> dataframe -> values)

testdf =df1[[0,1,2,3,4,5,6]]
maintestarray=testdf.values   # Accpeting the values of the rows from the dataframe
print(maintestarray)

y_pred = mul_lr.predict(maintestarray)  # predicting the values one by one
for i in range(len(y_pred)) :     # for loop is used to interate over all the values and get the predictions
	y_pred[i]=str((y_pred[i]))
DF = pd.DataFrame(y_pred,columns=['Predicted Personality'])
DF.index=DF.index+1
DF.index.names = ['Person No']   # This variable cosists of the predicted personlaity names
DF.to_csv("output.csv")     # This is used for converting it into output.csv

