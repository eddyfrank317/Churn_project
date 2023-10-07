Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that.So, if you were in charge of predicting customer churn, how would you go about using machine learning to make a good guess about which customers might leave? What steps would you take to create a machine learning model that can predict if someone's going to leave or not?```

Solution
Predicting customer churn is a common use case in the telecom industry, on this repo, i will take you step by step on how to accomplish this.
Step 1: Import the necessary libraries for use

import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Step 2: Load the data set
data = pd.read_csv("C:\Users\YU\Desktop\MOCK_DATA (3).csv")
data.head()
data.info()
data.describe()

#Step 3: Data preprocessing
Prepare the data for modelling by checking for missing values, outliers and duplicates and encoding categorical features

##handle missing values
data.dropna(inplace=True)
#encode categorucal variables if any using onehot encoding
one_hot_encoded_data = pd.get_dummies(data, columns = ['gender', 'last_purchase_date', 'customer_segment'])
data_encoded= pd.get_dummies(data, columns = ['gender', 'last_purchase_date', 'customer_segment'])

Step 4: Split the dataset into featres and target with 80% for training and 20% for testing
x=data_encoded.drop(columns=['churned_indicator'])
y=data_encoded['churned_indicator']

X_train: feature for training
X_train: feature for training
y_train: target_feature for training
y_train: target_feature for training

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

Step 5: Feature scalling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Step 6:  Build and train a machine learning model
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
model.fit(X_train_scaled, y_train)

Step 7: Evaluate the model
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test,predictions)
classification_rep = classification_report(y_test,predictions)



