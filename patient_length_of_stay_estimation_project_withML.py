# -*- coding: utf-8 -*-
# Patient Length of Stay Estimation Project With ML
# import libraries 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report,accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import pickle 

#load dataset 
data = pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")
data_ = data.head(50)
data.info()
describe = data.describe()

los = data["Length of Stay"]
data["Length of Stay"] = data["Length of Stay"].replace("120 +", 120)
data["Length of Stay"] = pd.to_numeric(data["Length of Stay"])
los = data["Length of Stay"]
data.isna().sum()

for column in data.columns:
    unique_values = len(data[column].unique())
    print(f"Number of unique values in {column}: {unique_values}")
    

data = data[data["Patient Disposition"] != "Expired" ]    

print(data["Payment Typology 1"].apply(type).value_counts())

print(data["Length of Stay"].apply(type).value_counts())



#EDA (Exploratory Data Analysis)

sns.boxplot(x = "Payment Typology 1", y = "Length of Stay", data=data,palette="Set2")
plt.xticks(rotation=60)
plt.title("Payment Typology 1 vs Length of Stay")
plt.tight_layout()
plt.show()

sns.countplot(x = "Age Group", data = data[data["Payment Typology 1"] == "Medicare"], order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"],palette="Set2")
plt.title("Medicare Partients for Age Group")

sns.boxplot(x = "Type of Admission", y="Length of Stay", data=data,palette="Set2")
plt.xticks(rotation=60)
plt.title("Type of Admission vs Length of Stay")

sns.boxplot(x = "Age Group", y="Length of Stay", data=data, order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"],palette="Set2")
plt.xticks(rotation=60)
plt.title("Age Group vs Length of Stay")

#Feature Encoding - selection: label encoding 
data = data.drop(["Hospital Service Area", "Hospital County","Operating Certificate Number",
                  "Facility Name","Zip Code - 3 digits","Patient Disposition","Discharge Year",
                  "CCSR Diagnosis Description","CCSR Procedure Description","APR DRG Description",
                  "APR MDC Description","APR Severity of Illness Description",
                  "Payment Typology 2","Payment Typology 3","Birth Weight","Total Charges","Total Costs"], axis = 1)
                  
age_group_index = {"0 to 17":1, "18 to 29":2, "30 to 49":3, "50 to 69":4, "70 or Older":5}
gender_index = {"U":0, "F":1, "M":2}
risk_and_severity_index = {np.nan:0, "Minor":1,"Moderate":2,"Major":3,"Extreme":4}

data["Age Group"] = data["Age Group"].apply(lambda x: age_group_index[x]) 
data["Gender"] = data["Gender"].apply(lambda x: gender_index[x]) 
data["APR Risk of Mortality"] = data["APR Risk of Mortality"].apply(lambda x: risk_and_severity_index[x]) 

encoder = OrdinalEncoder()
data["Race"] = encoder.fit_transform(np.asarray(data["Race"]).reshape(-1,1))
data["Ethnicity"] = encoder.fit_transform(np.asarray(data["Ethnicity"]).reshape(-1,1))
data["Type of Admission"] = encoder.fit_transform(np.asarray(data["Type of Admission"]).reshape(-1,1))
data["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(data["CCSR Diagnosis Code"]).reshape(-1,1))
data["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(data["CCSR Procedure Code"]).reshape(-1,1))
data["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(data["APR Medical Surgical Description"]).reshape(-1,1))
data["Payment Typology 1"] = encoder.fit_transform(np.asarray(data["Payment Typology 1"]).reshape(-1,1))
data["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(data["Emergency Department Indicator"]).reshape(-1,1))

# Missing Value Control
data.isna().sum()
data = data.drop("CCSR Procedure Code", axis =1)
data = data.dropna(subset=["Permanent Facility Id", "CCSR Diagnosis Code"])

#Train Test Split 

X = data.drop(["Length of Stay"], axis = 1)
y = data["Length of Stay"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Regression: Train and Test

dtree = DecisionTreeRegressor(max_depth = 10) # we used max_depth func because we wanted to block overfitting.
dtree.fit(X_train, y_train)
train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("RMSE: Train", np.sqrt(mean_squared_error(y_train, train_prediction)))
print("RMSE: Test", np.sqrt(mean_squared_error(y_test, test_prediction)))

#Solve Classification Problem with Categorization 
      
bins = [0,5,10,20,30,50,120]
labels = [5,10,20,30,50,120]

data ["Los_bin"] = pd.cut(x=data["Length of Stay"], bins = bins )
data["Los_label"] =pd.cut(x=data["Length of Stay"], bins = bins, labels = labels)
data_ = data.head(50)

data ["Los_bin"] = data["Los_bin"].apply(lambda x: str(x).replace(",","-"))
data ["Los_bin"] = data["Los_bin"].apply(lambda x: str(x).replace("120","120+"))

f, ax = plt.subplots()
sns.countplot(x = "Los_bin", data = data, palette="Set2")

new_X = data.drop(["Length of Stay","Los_bin","Los_label"], axis = 1) 
new_y = data["Los_bin"]

X_train, X_test, y_train, y_test = train_test_split(new_X,new_y,test_size = 0.2, random_state = 42)

dtree = DecisionTreeClassifier(max_depth=15)
dtree.fit(X_train,y_train)

# Model Save
with open("patient_los_model.pkl", "wb") as f:
    pickle.dump(dtree, f)

train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("Train_Accuracy: ", accuracy_score(y_train, train_prediction))
print("Test_Accuracy: ", accuracy_score(y_test, test_prediction))
print("Classification report: ", classification_report(y_test, test_prediction))

#Model Evuluation - Confusion Matrix and ROC Curve

cm = confusion_matrix(y_test, test_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix= cm)
disp.plot(cmap = "YlGnBu")
plt.title("Confusion Matrix")
plt.show()


 

















