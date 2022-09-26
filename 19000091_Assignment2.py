#19000091
#SCS3201 - Machine Learning and Neural Computing
#Assignment 2

import pandas as pd

#read the data file
car_evaluvation = pd.read_csv("car.csv")

#print the data head
print(car_evaluvation.head())

#if there are any missing values remove them 
car_evaluvation = car_evaluvation.dropna()

#data preprocessing
buyingAndmaintDataClassification = {
    'vhigh' : 3,
    'high' : 2,
    'med' : 1,
    'low' : 0
    }

car_evaluvation['buying'] = car_evaluvation['buying'].apply( lambda id : buyingAndmaintDataClassification [id])

car_evaluvation['maint'] = car_evaluvation['maint'].apply( lambda id : buyingAndmaintDataClassification [id])


doorsDataClassification = {
    '2' : 2,
    '3' : 3,
    '4' : 4,
    '5more' : 5
    }

car_evaluvation['doors'] = car_evaluvation['doors'].apply( lambda id : doorsDataClassification [id])


personsDataClassification = {
    '2' : 2,
    '4' : 4,
    'more' : 5
    }

car_evaluvation['persons'] = car_evaluvation['persons'].apply( lambda id : personsDataClassification [id])


lug_bootDataClassification = {
    'small' : 1,
    'med' : 2,
    'big' : 3
    }

car_evaluvation['lug_boot'] = car_evaluvation['lug_boot'].apply( lambda id : lug_bootDataClassification [id])


safetyDataClassification = {
    'low' : 1,
    'med' : 2,
    'high' : 3
    }

car_evaluvation['safety'] = car_evaluvation['safety'].apply( lambda id : safetyDataClassification [id])


car_classDataClassification = {
    'unacc' : 1,
    'acc' : 2,
    'good' : 3,
    'vgood' : 4
    }

car_evaluvation['car_class'] = car_evaluvation['car_class'].apply( lambda id : car_classDataClassification [id])

#print the data head after data preprocessing
print(car_evaluvation.head())

x = car_evaluvation.iloc[:, :6]
y = car_evaluvation.iloc[:,6]

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 4, bootstrap = True, ccp_alpha = 0.00)

rfc.fit(xTrain , yTrain)

yPrediction = rfc.predict(xTest)

from sklearn import metrics

print("Accuracy of the trained model is:",metrics.accuracy_score(yTest, yPrediction))

featureImportance = pd.Series(rfc.feature_importances_,index=['buying','maint','doors','persons','lug_boot','safety']).sort_values(ascending=False)
featureImportance

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot to see the what requirement are more important to decide the car class evaluation
sns.color_palette("Paired")
sns.barplot(x = featureImportance, y = featureImportance.index)

# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Requirements')
plt.title("Important Features for Car Evaluation")
plt.legend()
plt.show() 
 