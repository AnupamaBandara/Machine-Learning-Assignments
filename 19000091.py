#19000091

import pandas as pd

#read the data file
dataFrame = pd.read_csv("cmc.csv")

#print the data head
print(dataFrame.head())

#if there are any missing values remove them 
dataFrame = dataFrame.dropna()

#print the data head after data preprocessing
print(dataFrame.head())

x = dataFrame.iloc[:, :9]
y = dataFrame.iloc[:,9]

from sklearn.model_selection import train_test_split

x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100, bootstrap = True, ccp_alpha = 0.00)

rfc.fit(x_Train , y_Train)

y_Prediction = rfc.predict(x_Test)

from sklearn import metrics

print("Accuracy of the trained model is:",metrics.accuracy_score(y_Test, y_Prediction))

featureImportance = pd.Series(rfc.feature_importances_,index=['wife_age','wife_edu','husband_edu','no_children_ever','wife_region','wife_work','husband_occ','living_index','media_expo']).sort_values(ascending=False)
featureImportance

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot to see the what requirement are more important to decide the car class evaluation
sns.color_palette("pastel")
sns.barplot(x = featureImportance, y = featureImportance.index)

# Add labels to the graph
plt.xlabel('Characteristics Importance Score')
plt.ylabel('Characteristics')
plt.title("Important Characteristics for Contraceptive Method Choice")
plt.legend()
plt.show() 
 