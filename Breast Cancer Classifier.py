# Import required dependencies
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# Load the dataset
cancer_dataset = pd.read_csv("data.csv")
# Show the first five rows in the dataset
cancer_dataset.head()
# Show the last five rows in the dataset
cancer_dataset.tail()
# Show the dataset shape
cancer_dataset.shape
# Show some statistical info about the dataset
cancer_dataset.describe()
# Find the ralation between the output diagnosis and all the dataset input features
cancer_dataset.groupby('diagnosis').mean()


# check if there is any none(missing) values in the dataset to decide if will make a data cleaning or not
cancer_dataset.isnull().sum()
# Convert the textual diagnosis column from textual into numeric column by Replacing M with 1 and B with 0
cancer_dataset.replace({'diagnosis':{'M':1,'B':0}},inplace=True)
# Show the dataset after label encoding the ouput diagnosis column
cancer_dataset.head()


# Count the number of groups in the diagnosis column and its repetition
cancer_dataset['diagnosis'].value_counts()
# Plot the gourps with its repetions
plt.figure(figsize=(5,5))
sns.countplot(x = 'diagnosis',data=cancer_dataset)

# Plot the distribution of all columns in the dataset with different colors
colors = sns.color_palette('husl', n_colors=len(cancer_dataset.columns))
for i, column in enumerate(cancer_dataset.columns):
    if i==0 or i==cancer_dataset.shape[1]-1:
        continue
    plt.figure(figsize=(5,5))
    sns.distplot(cancer_dataset[column], color=colors[i])



# Split data into input and label data
X = cancer_dataset.drop(columns=['id','diagnosis','Unnamed: 32'],axis=1) 
Y = cancer_dataset['diagnosis'] 
print(X)
print(Y)
# split the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Create the model and train it
LRModel = LogisticRegression()
LRModel.fit(x_train,y_train)
# Make the model predict on train and test input data
predicted_train = LRModel.predict(x_train)
predicted_test = LRModel.predict(x_test)
# Avaluate the model accuracy
accuracy_predicted_train = accuracy_score(predicted_train,y_train)
accuracy_predicted_test = accuracy_score(predicted_test,y_test)
print(accuracy_predicted_train)
print(accuracy_predicted_test)



# Make a predictive system
input_data = [[13.61,24.98,88.05,582.7,0.09488,0.08511,0.08625,0.04489,0.1609,0.05871,0.4565,1.29,2.861,43.14,0.005872,0.01488,0.02647,0.009921,0.01465,0.002355,16.99,35.27,108.6,906.5,0.1265,0.1943,0.3169,0.1184,0.2651,0.07397]
    ,[9.787,19.94,62.11,294.5,0.1024,0.05301,0.006829,0.007937,0.135,0.0689,0.335,2.043,2.132,20.05,0.01113,0.01463,0.005308,0.00525,0.01801,0.005667,10.92,26.29,68.81,366.1,0.1316,0.09473,0.02049,0.02381,0.1934,0.08988]]
# convert input data into 1D numpy array
input_array = np.array(input_data)
# convert 1D input array into 2D
input_array[0] = input_array[0].reshape(1,-1)
input_array[1] = input_array[1].reshape(1,-1)
# make the model predict the output
predictions = LRModel.predict(input_array)
if predictions[0]==1:
    print("The breast cancer is Malignant")
else:
    print("The breast cancer is Benign")
if predictions[1]==1:
    print("The breast cancer is Malignant")
else:
    print("The breast cancer is Benign")

