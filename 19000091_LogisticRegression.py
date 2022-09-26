import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data imported to the file. Data gathering
car_evaluvation = pd.read_csv("car.csv")

#if there are any missing values remove them 
car_evaluvation = car_evaluvation.dropna()

#data preprocessing
buyingAndmaintDataClassification ={
    'vhigh' : 3,
    'high' : 2,
    'med' : 1,
    'low' : 0
    }

car_evaluvation['buying'] = car_evaluvation['buying'].apply( lambda id : buyingAndmaintDataClassification [id])

car_evaluvation['maint'] = car_evaluvation['maint'].apply( lambda id : buyingAndmaintDataClassification [id])


doorsDataClassification ={
    '2' : 2,
    '3' : 3,
    '4' : 4,
    '5more' : 5
    }

car_evaluvation['doors'] = car_evaluvation['doors'].apply( lambda id : doorsDataClassification [id])


personsDataClassification ={
    '2' : 2,
    '4' : 4,
    'more' : 5
    }

car_evaluvation['persons'] = car_evaluvation['persons'].apply( lambda id : personsDataClassification [id])


lug_bootDataClassification ={
    'small' : 1,
    'med' : 2,
    'big' : 3
    }

car_evaluvation['lug_boot'] = car_evaluvation['lug_boot'].apply( lambda id : lug_bootDataClassification [id])


safetyDataClassification ={
    'low' : 1,
    'med' : 2,
    'high' : 3
    }

car_evaluvation['safety'] = car_evaluvation['safety'].apply( lambda id : safetyDataClassification [id])


car_classDataClassification ={
    'unacc' : 1,
    'acc' : 2,
    'good' : 3,
    'vgood' : 4
    }

car_evaluvation['car_class'] = car_evaluvation['car_class'].apply( lambda id : car_classDataClassification [id])


x = car_evaluvation.iloc[ : , : 6 ]
y = car_evaluvation.iloc[ : , 6 ]


#seperating traning and testing data randomly from the data frame

from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )


#Training the Logistic regression Model using training data
from sklearn.linear_model import SGDClassifier

trainedDataModel = SGDClassifier( ).fit( xTrain , yTrain )


#Predicting results for xTest variables using trained data model
predictedResults = trainedDataModel.predict( xTest )


#Testing accuracy of trained model
meanAccuracy = trainedDataModel.score( xTest , yTest )
print( "Accuracy of model : " , meanAccuracy * 100 , " %" )


#Generating prediction report
from sklearn.metrics import classification_report
predictionReport = classification_report( yTest , predictedResults )
print( "Prediction report : " )
print( predictionReport )