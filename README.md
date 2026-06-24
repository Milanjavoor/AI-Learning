# AIML-Learning
It contains the models built on google colab as a part of my learning in the ever changing field of AI and machine learning
project 1:
# 🛡️ Cyber Attack Detection using 1D Convolutional Neural Networks (CNN)

A deep learning-based cybersecurity project that detects malicious network activity using a 1D Convolutional Neural Network (CNN). The model is trained on network traffic and system event data to classify whether an event represents normal behavior or a potential cyber attack.

The project includes data preprocessing, feature encoding, normalization, model training, evaluation, and real-time prediction capabilities. Categorical features are transformed using label encoding, numerical features are scaled using Min-Max normalization, and the processed data is reshaped for CNN-based analysis.

By leveraging convolutional layers, the model automatically learns patterns and relationships within cybersecurity data, enabling efficient detection of suspicious activities and potential threats. The system can be used as a foundation for intrusion detection systems (IDS), threat monitoring tools, and intelligent cybersecurity applications.

## 🚀 Features

* Automated cyber attack detection
* Data preprocessing and cleaning pipeline
* Label encoding for categorical features
* Feature scaling using Min-Max normalization
* 1D CNN architecture for pattern recognition
* Binary classification (Attack vs Normal Traffic)
* Real-time event prediction function
* Model evaluation with accuracy metrics

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Google Colab

## 🧠 Deep Learning Concepts Demonstrated

* Convolutional Neural Networks (CNNs)
* Feature Extraction
* Binary Classification
* Activation Functions (ReLU, Sigmoid)
* Max Pooling
* Dropout Regularization
* Model Training and Validation
* Neural Network Evaluation

## 📊 Data Processing Pipeline

1. Load cybersecurity dataset
2. Remove irrelevant features
3. Handle missing values
4. Encode categorical features
5. Normalize feature values
6. Reshape data for CNN input
7. Train the neural network
8. Evaluate model performance
9. Predict cyber attacks on new inputs

## 🎯 Prediction Output

The trained model classifies incoming events as:

* 🚨 Attack Detected
* ✅ Normal Traffic

## 📚 Learning Outcomes

* Understand how CNNs can be applied beyond image processing
* Learn cybersecurity data preprocessing techniques
* Build an end-to-end deep learning classification pipeline
* Explore intrusion detection using neural networks
* Gain practical experience with TensorFlow and Keras
Sequential Feature Selection using Random Forest
Overview

This project demonstrates a complete machine learning preprocessing and feature selection pipeline using Python and Scikit-Learn.

The objective is to identify the most important features from a synthetic dataset by applying Sequential Forward Feature Selection with a Random Forest Classifier.

The workflow includes data sampling, missing value handling, categorical encoding, train-test splitting, feature selection, and evaluation using cross-validation.

Dataset

Dataset Used:

synthetic_soc_1000_indexed.csv

The dataset contains both numerical and categorical features along with a target label column.

A random sample of 600 records is selected from the dataset for experimentation and analysis.

Project Workflow

Dataset Loading
↓
Random Sampling
↓
Missing Value Handling
↓
Categorical Encoding
↓
Label Encoding
↓
Train-Test Split
↓
Random Forest Model
↓
Sequential Feature Selection
↓
Cross-Validation Evaluation
↓
Top Feature Identification

Data Preprocessing
Missing Value Treatment

Numerical Features:

Missing values are replaced using the median value of each column.

Categorical Features:

Missing values are replaced using the mode (most frequent value).
If no mode exists, the value "Unknown" is assigned.

This ensures that no missing values remain before model training.

Feature Encoding

Categorical features are converted into numerical form using Label Encoding.

Example:

Gender:

Male → 0
Female → 1

The target label column is also encoded into numerical values.

Train-Test Split

The dataset is divided into:

Training Set: 80%
Testing Set: 20%

Random State:

42

This ensures reproducible results.

Machine Learning Model

Model Used:

Random Forest Classifier

Configuration:

Number of Trees: 100
Random State: 42

Random Forest is chosen because it is robust, handles mixed feature types effectively, and provides strong performance for feature selection tasks.

Sequential Feature Selection

The project uses Sequential Forward Selection (SFS).

How It Works
Start with zero features.
Add one feature at a time.
Evaluate model performance using cross-validation.
Keep the feature that improves accuracy the most.
Repeat until the desired number of features is selected.

Configuration:

Direction: Forward
Features Selected: 10
Cross Validation Folds: 5
Scoring Metric: Accuracy
Output

The program produces:

Selected Features

A list of the top 10 most informative features selected by the Sequential Feature Selector.

Example:

Selected features:
['feature_1', 'feature_5', 'feature_8', ...]

Cross-Validation Score

Average classification accuracy across 5 folds.

Example:

CV Score: 0.873

Reduced Feature Matrix

The transformed training dataset containing only the selected features.

Example:

Selected shape:
(480, 10)

Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Key Concepts Demonstrated
Data Cleaning
Missing Value Imputation
Label Encoding
Feature Engineering
Feature Selection
Random Forest Classification
Cross Validation
Machine Learning Pipeline Design
Learning Outcomes

Through this project, users can learn:

How to handle missing data effectively.
How to encode categorical variables.
How feature selection improves model efficiency.
How Sequential Feature Selection works internally.
How Random Forest can be used as a feature evaluation model.
How cross-validation helps estimate model performance.
Future Improvements
Recursive Feature Elimination (RFE)
Mutual Information Feature Selection
Chi-Square Feature Selection
XGBoost Feature Importance
SHAP-Based Feature Analysis
Hyperparameter Optimization
Automated Feature Engineering
Author

Developed as a machine learning feature selection project to explore data preprocessing, feature engineering, and Sequential Forward Selection using Random Forest classifiers.
