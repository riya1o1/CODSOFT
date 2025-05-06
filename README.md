# CODSOFT
Tasks for CODSOFT Internship - April 2025

# Task 1 - Titanic Survival Prediction
This project is part of the CODSOFT Internship. It focuses on predicting whether a passenger survived the Titanic disaster using a machine learning model trained on historical data.

# Problem Statement
Given data on Titanic passengers (such as age, sex, ticket class, etc.), the goal is to build a model that predicts survival.

# Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn (for visualization)
- Streamlit (for deployment)
- TensorFlow/Keras (for model)
- CSV dataset

# Files Included
- app.py – Streamlit web app
- Titanic-Dataset.csv – Dataset used for training
  (https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- model.h5 – Trained model (saved using Keras)
  (https://colab.research.google.com/drive/1NtH64pSqJUoiI4qy6UH_V-_4JwLk9r3u)
- screenshots – Output screenshots of the app 

# Model Accuracy
Achieved **81.01% accuracy** on test data 

# Output
- Interface - Interface.png
  ![Interface](https://github.com/user-attachments/assets/1b11203e-3b89-4271-a66e-fcf27fc4d16f)

- Not-survived - Not-survived.png
  ![Not-survived](https://github.com/user-attachments/assets/010eeb7b-99ce-4c9f-88b2-918ac8dbe2ff)

- Survived - Survived.png
  ![Survived](https://github.com/user-attachments/assets/15ad42c7-8ed4-4580-b521-69e0c0bb8f9f)


                                                            . . . . . . .
  
# Task 2 - Movie Rating Prediction
This project aims to predict IMDb movie ratings using a machine learning model. The model is trained using features like genre, director, and actors, and it is deployed with an interactive Streamlit UI for real-time prediction.

# Problem Statement
Given the metadata of a movie (genre, director, and actors), the goal is to predict the IMDb rating of the movie using a trained neural network model.

# Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras (for neural network)
- Streamlit (for interactive UI)
- CSV dataset

# Files Included
- app.py – Streamlit web app
- Titanic-Dataset.csv – Dataset used for training
  (https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
- model.h5 – Trained model (saved using Keras)
  (https://colab.research.google.com/drive/1SfCzeE1mfzyVKkhPl0EFlat5mIDFCd2s)
- screenshots – Output screenshots of the app

# Output
- Output.png
  ![Output](https://github.com/user-attachments/assets/5163d946-6412-4697-afe4-a2f63b5b493f)


                                                              . . . . . . .

# Task 3 - Iris Flower Species Prediction
This project uses a Random Forest Classifier to predict the species of an Iris flower based on its features. The model is trained using the Iris dataset, and an interactive Streamlit UI is used to make real-time predictions.

# Problem Statement
Given the sepal length, sepal width, petal length, and petal width of an Iris flower, the goal is to predict the species of the flower using a trained machine learning model.

# Technologies Used
- Python
- Pandas: For data manipulation and cleaning.
- NumPy: For numerical operations.
- Scikit-learn: For training the machine learning model.
- Streamlit: For building the interactive user interface.
- Joblib: For saving and loading the trained model and label encoder.

# Files Included
- app.py – Streamlit web app
- IRIS.csv – Dataset used for training
  (https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- random_forest_model.joblib – Trained model (saved using RandomForestClassifier model)
  (https://colab.research.google.com/drive/1P7m6WGOxjVobLcMiO9pQMWJLJjWleDok#scrollTo=mHIjbKPv2hRz)
- screenshots – Output screenshots of the app

# Output
- Iris_prediction.png
  ![Iris_prediction](https://github.com/user-attachments/assets/10c77755-80c8-43a1-ab74-72c544f33b61)

  
                                                              . . . . . . .

# Task 4 - Sales Prediction Using Linear Regression
This project uses a Linear Regression model to predict product sales based on advertising expenditures across various media channels. The model is trained using the Advertising dataset, and predictions are made using new inputs for TV, Radio, and Newspaper budgets.

# Problem Statement
Given the advertising spend on **TV**, **Radio**, and **Newspaper**, the goal is to predict the **sales** of a product using a trained Linear Regression model.

# Technologies Used
- Python
- Pandas: For data manipulation and cleaning.
- NumPy: For numerical operations.
- Scikit-learn: For training the Linear Regression model.
- Matplotlib & Seaborn: For data visualization and analysis.
- Joblib: For saving and loading the trained model.

# Files Included
- sales_prediction.py – Main script for data loading, visualization, training, and evaluation
- advertising.csv – Dataset used for training  
  (https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input)
- sales_model.pkl – Trained model (saved using scikit-learn’s LinearRegression)
  (https://colab.research.google.com/drive/1a4hmekTOObPFxMh_pzxW-_WZH7VtUUX5#scrollTo=fi96jwKYzdqX)
- predict_sales.py – Script to load the model and make predictions on new data
- screenshots – Output visualizations and performance graphs

# Output
- Output.png
  ![Output (2)](https://github.com/user-attachments/assets/552c84f6-1834-4af2-b3dc-637d1091103e)


                                                                . . . . . . .

# Task 5 – Credit Card Fraud Detection (CODSOFT Internship Project)
This project is developed as part of the CODSOFT Internship Program. It focuses on detecting fraudulent credit card transactions using machine learning techniques. The solution uses a classification model trained on anonymized transaction data and deployed via a user-friendly Streamlit interface.

##  Problem Statement
With the rising number of digital transactions, fraud detection has become a critical challenge for financial institutions. Given features extracted from historical credit card transactions (including PCA-transformed fields like V1 to V28, Time, and Amount), the objective is to build a machine learning model that can accurately classify whether a transaction is **fraudulent (1)** or **genuine (0)**.

# Technologies & Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit (for deployment)
- imbalanced-learn (for SMOTE)
- CSV dataset

# Files Included
- app.py – Streamlit web app
- creditcard.csv – Dataset used for training
  (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- logistic_model.pkl – Trained logistic regression model
  (https://colab.research.google.com/drive/1a0RYaDVW3ICUyP6bt_9WZ-aPjx4vAaN4#scrollTo=E3ekbRvD63zR)
- screenshots – UI and output screenshots

# Model Accuracy
Achieved **97.34% accuracy** on test data 

# Output
- Credit_output
![Credit_output](https://github.com/user-attachments/assets/671e0060-b13b-4cad-81f1-9757b9d8d758)



  


