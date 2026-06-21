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
