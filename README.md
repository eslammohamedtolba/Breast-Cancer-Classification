# Breast-Cancer-Classification
This Python script is designed to classify breast cancer tumors as either benign (non-cancerous) or malignant (cancerous) using a Logistic Regression model.
The dataset is included in the repository, making it easy to get started. 
This README provides an overview of the code, its prerequisites, and how to use the model.

## Prerequisites
Before using the Breast Cancer Classification Model, you need to have the following dependencies installed:
- Python (>=3.6)
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

## Code Overview
- Import the necessary dependencies and libraries.
- Load the breast cancer dataset from a CSV file (data.csv).
- Explore the dataset by displaying its shape, statistical summary, and the relationship between the output diagnosis and input features.
- Check for missing values in the dataset and perform label encoding on the diagnosis column (Malignant as 1, Benign as 0).
- Visualize the distribution of the diagnosis groups.
- Split the data into input features (X) and labels (Y).
- Split the data into training and testing sets.
- Create a Logistic Regression model and train it using the training data.
- Make predictions on both the training and testing data.
- Calculate and print the accuracy of the model on both sets.

## Predictive System
The code includes a section for making predictions on new data. 
To predict the type of a breast cancer tumor, input features should be provided as a list of lists (2D array).
The model will classify the tumor as either malignant or benign.


## Accuracy
The model achieves an accuracy of 91% on testing data, making it a powerful tool for classifying breast cancer tumors as benign or malignant.


## Contribution
Contributions to this project are welcome. If you have ideas to enhance the model's accuracy, improve its efficiency, or add new features, feel free to make a pull request or open an issue. 
Your contributions can help in the early detection and diagnosis of breast cancer, which is crucial for patient outcomes and treatment decisions.
