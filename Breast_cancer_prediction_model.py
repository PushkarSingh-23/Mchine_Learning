# %% [markdown]
# Label Encoding :
# @ Converting the labels into numeric form

# %%
#importing the Dependencies 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# Label Encoding of Breast Cancer Dataset

# %%
#loading the data from csv file to pandas dataframe 
cancer_data = pd.read_csv('F:\Machine_learning\data.csv')

# %% [markdown]
# Loading First Five rows of dataset

# %%
cancer_data.head()

# %% [markdown]
# Loading Shape of Dataset

# %%
cancer_data.shape

# %% [markdown]
# Loading last five rows of dataset

# %%
cancer_data.tail()

# %% [markdown]
# Finding the count of different labels
# 

# %%
cancer_data['diagnosis'].value_counts()

# %% [markdown]
# converting B and M into numeric values

# %%
#load the label Encoder function
LabelEncoder = LabelEncoder()

# %%
label = LabelEncoder.fit_transform(cancer_data['diagnosis'])

# %%
#appending the labels to the dataframe
cancer_data['target'] = label

# %%
cancer_data.head()

# %% [markdown]
# 0 ---> Benign :
# 1 ---> Malignant

# %%
cancer_data['target'].value_counts()

# %%
x = cancer_data.drop(['target','diagnosis'],axis=1)
y = cancer_data['target']

# %%
x = cancer_data.drop(['Unnamed: 32' ,'diagnosis' ,'target'],axis=1)


# %%
x

# %%
y

# %%
y.head()

# %%
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
scaler = StandardScaler()

# %%
scaler.fit(x)

# %%
standardized_data = scaler.transform(x)

# %%
standardized_data

# %% [markdown]
# splitting the data into training data and testing data

# %%
x_train, x_test, y_train, y_test = train_test_split(standardized_data, y, test_size=0.2, random_state=5)

# %%
x_train.shape

# %%
standardized_data.shape

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
model = LogisticRegression()

# %%
model.fit(x_train, y_train)

# %%
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction , y_train)

# %%
print('Accuracy on Training Data :' , training_data_accuracy)

# %%
import numpy as np

# %%
input_data = (867739,18.45,21.91,120.2,1075,0.0943,0.09709,0.1153,0.06847,0.1692,0.05727,0.5959,1.202,3.766,68.35,0.006001,0.01422,0.02855,0.009148,0.01492,0.002205,22.52,31.39,145.6,1590,0.1465,0.2275,0.3965,0.1379,0.3109,0.0761)
# reshape the numpy array as we ase predicting for only one data point 
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)

# %%
if prediction == 1:
  print("Person ha initial stage of cancer")
elif prediction == 0:
  print("Person does have advance stage of cancer")

# %%



