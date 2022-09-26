import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data imported to the file. Data gathering
concrete_compression_strength = pd.read_excel("Concrete_Data.xls")

#if there are any missing values remove them 
concrete_compression_strength = concrete_compression_strength.dropna()

#Data preprocessing is not required because all the attribute values have real/integer values

x = concrete_compression_strength.iloc[ : , : 8 ]
y = concrete_compression_strength.iloc[ : , 8 ]


#seperating traning and testing data randomly from the data frame
from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )


#Training the Linear regression Model using training data
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

trainedDataModel = make_pipeline( StandardScaler() , SGDRegressor( alpha = 0.0001 ) ).fit( xTrain , yTrain )


#Predicting results for xTest variables using trained data model
predictedResults = trainedDataModel.predict( xTest )

#Testing accuracy of trained model
meanAccuracy = trainedDataModel.score( xTest , yTest )
print( "Accuracy of model : " , meanAccuracy * 100 )


#Calculating mean squared error
from sklearn.metrics import mean_squared_error
meanSquaredError = mean_squared_error( yTest , predictedResults )
print( "Root mean squared error : " , meanSquaredError** ( 0.5 ) )

maxValue = max(concrete_compression_strength['concrete_compre_stren'])
minValue = min(concrete_compression_strength['concrete_compre_stren'])
print( "Normalized Root mean squared error : " , (meanSquaredError** ( 0.5 ))/(maxValue-minValue)  )

