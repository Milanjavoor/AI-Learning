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
* 🧠 Handwritten Digit Recognition using TensorFlow (MNIST)
📌 Overview

This project implements a Multi-Layer Perceptron (MLP) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model is trained on grayscale images of handwritten digits (0–9) and learns to predict the correct digit with high accuracy.

The project also visualizes the training process by plotting both accuracy and loss curves over multiple epochs.

🚀 Features
Loads the MNIST handwritten digit dataset.
Normalizes image pixel values for faster convergence.
Builds a fully connected neural network using TensorFlow/Keras.
Trains the model using the Adam optimizer.
Evaluates model performance on unseen test data.
Generates predictions for test images.
Visualizes training and validation accuracy.
Visualizes training and validation loss.
🛠️ Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
📂 Dataset

The project uses the built-in MNIST dataset provided by TensorFlow.

Training Images: 60,000
Testing Images: 10,000
Image Size: 28 × 28 pixels
Classes: 10 (Digits 0–9)
🧠 Model Architecture
Input Layer (784 neurons)
        │
Dense Layer (128 neurons, ReLU)
        │
Dense Layer (128 neurons, ReLU)
        │
Output Layer (10 neurons)
⚙️ Training Configuration
Optimizer: Adam
Learning Rate: 0.001
Loss Function: Sparse Categorical Crossentropy
Metric: Sparse Categorical Accuracy
Epochs: 33
Batch Size: 32
📈 Results

The model achieves high classification accuracy on the MNIST test dataset while demonstrating effective learning through training and validation performance plots.

📊 Output

The project displays:

Sample handwritten digit image
Actual label of the sample image
Model summary
Test accuracy
Predicted logits for test images
Accuracy vs Epoch graph
Loss vs Epoch graph
📚 Learning Objectives

This project demonstrates:

Data preprocessing
Image normalization
Building Sequential models in TensorFlow
Dense (Fully Connected) Neural Networks
Model compilation and training
Performance evaluation
Prediction generation
Visualization of learning curves
▶️ How to Run
Clone the repository
git clone https://github.com/your-username/your-repository.git
Install dependencies
pip install tensorflow numpy matplotlib
Run the program
python main.py
📌 Future Improvements
Reshape images and implement a Convolutional Neural Network (CNN) for improved accuracy.
Add a confusion matrix and classification report.
Display predicted labels alongside input images.
Save and reload the trained model.
Build a simple GUI for real-time handwritten digit prediction.
👨‍💻 Author

Milan Javoor

Deep Learning • Artificial Intelligence • Quantum Computing • Machine Learning

Note

One small issue in your current code: your model expects flattened input of shape (784,), but you're feeding images of shape (28, 28). Before training, you should reshape the data:

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

Also, change:

plt.show

to:

plt.show()

Otherwise, the plots won't be displayed. These fixes will make the project run correctly and align the input with the model architecture.
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


Formula 1 Winner Prediction (1950–2020)
Short description
This project builds a simple binary classifier that predicts whether a driver won a Formula 1 race (winner vs non-winner) using historical data from the "Formula 1 World Championship 1950–2020" Kaggle dataset. It preprocesses and merges drivers, races, results, qualifying and constructor information and trains a 1D CNN (Conv1D) on a small set of numerical features.

Features used

grid: starting grid position

laps: number of laps completed

year: season year

round: round number within the season

age: driver's age at the race (computed from dob and year)

pole position: binary indicator derived from qualifying result (1 if qualified 1st)

Target

winner: binary label where 1 indicates the driver finished first (positionOrder == 1)

Project layout

code.ipynb or train.py: main notebook / script containing the data loading, preprocessing, model training, and evaluation code.

requirements.txt: (recommended) pip dependencies such as pandas, numpy, tensorflow, scikit-learn, matplotlib, seaborn, kagglehub (or instructions to download dataset).

README.md: this file.

data/: optional directory to store downloaded CSV files if not using Kagglehub.

Dependencies

Python 3.8+ recommended

pandas

numpy

scikit-learn

matplotlib

seaborn

tensorflow (tested with 2.x)

kagglehub (optional, for programmatic dataset download) or use Kaggle CLI / download manually

Installation

Create and activate a virtual environment (optional but recommended)

python -m venv venv

source venv/bin/activate (Linux/macOS) or venv\Scripts\activate (Windows)

Install dependencies

pip install -r requirements.txt
or (minimal)

pip install pandas numpy scikit-learn matplotlib seaborn tensorflow kagglehub

Usage

Download dataset:

Option A (kagglehub): the script uses kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020") to fetch files automatically.

Option B (manual): download the dataset from Kaggle and place CSV files under /kaggle/input/formula-1-world-championship-1950-2020/ or update the paths in the script.

Run the notebook/script:

In Colab or Kaggle kernel: open the notebook and run cells.

Locally: python train.py (adjust paths and remove Kaggle-specific helpers)

What the script does (high level)

Imports CSVs: drivers, results, races, constructors, qualifying.

Cleans data: drops nulls for constructors and qualifying; computes driver age.

Merges tables on raceId and driverId and matches constructors.

Creates binary labels: winner, pole position, podium.

Selects numeric features, scales them with StandardScaler.

Reshapes features to 3D to feed Conv1D: (samples, timesteps/features, 1).

Builds a Sequential Conv1D model with BatchNormalization, MaxPooling, Dropout, Dense layers.

Trains with early stopping on validation loss.

Plots training/validation loss curve and evaluates on test set.

Model architecture

Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_features,1))

BatchNormalization

MaxPooling1D(pool_size=2)

Dropout(0.3)

Conv1D(filters=128, kernel_size=2, activation='relu')

BatchNormalization

Dropout(0.3)

Flatten

Dense(128, activation='relu')

Dropout(0.4)

Dense(64, activation='relu')

Dense(1, activation='sigmoid')

Loss: binary_crossentropy

Optimizer: Adam

Metrics: accuracy

EarlyStopping: monitor='val_loss', patience=5, restore_best_weights=True

Evaluation

Prints model accuracy on test set.

Produces a loss vs epoch plot for training/validation.

You can compute other metrics (precision, recall, F1, confusion matrix) using sklearn.metrics.

Known issues & notes

Using Conv1D on a small number of independent features (6) is atypical. Conv1D assumes local sequential relationships across the time/feature axis, which may not hold here. A Dense MLP or tree-based model (RandomForest, XGBoost) is likely more appropriate.

Class imbalance: winners are extremely rare relative to non-winners. That class imbalance can make accuracy misleading. Consider stratified sampling, class weights, oversampling (SMOTE) or calibrating thresholds.

Missing values: the code drops rows with missing constructor/qualifying which may bias the dataset. Consider more careful imputation or retaining rows where possible.

Age computation: uses the driver's year minus the birth year substring. This assumes dob exists and is formatted "YYYY-MM-DD". Verify correctness.

Feature engineering: add more informative features (constructor strength, driver season standings, circuit characteristics, weather, qualifying gap, pit stops).

Target leakage risk: ensure features available at race start only (e.g., grid, qualifying) are used; avoid using in-race results or future info.

Improvements & experiments to try

Replace Conv1D with a Dense network or tree-based models (RandomForest, XGBoost) and compare performance.

Handle class imbalance with class_weight or resampling techniques.

Add categorical embeddings (constructor, circuit, driver) and use them as features.

Use time-series or sequence models to incorporate a driver's recent form across races.

Use cross-validation and stratified splits to get robust estimates.

Evaluate precision, recall, F1, ROC-AUC and confusion matrix; report class-specific metrics.

Example commands to compute additional metrics (add to script)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = (model.predict(x_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

Credits and dataset

Dataset: "Formula 1 World Championship 1950–2020" by Rohan Rao on Kaggle. Link: https://www.kaggle.com/rohanrao/formula-1-world-championship-1950-2020
📄 Optical Character Recognition (OCR) using Tesseract and OpenCV
📌 Overview

This project demonstrates Optical Character Recognition (OCR) using Tesseract OCR, OpenCV, and Google Colab. The program allows users to upload an image containing printed text, preprocesses the image to improve readability, and extracts the text using the Tesseract OCR engine.

Image preprocessing techniques such as grayscale conversion, noise removal, and adaptive thresholding are applied to enhance OCR accuracy.

🚀 Features
Upload images directly in Google Colab.
Preprocess images for improved text recognition.
Remove image noise using median filtering.
Apply adaptive thresholding for better text visibility.
Extract printed text using Tesseract OCR.
Display the recognized text in the output console.
🛠️ Technologies Used
Python
OpenCV
Pytesseract
NumPy
Pillow (PIL)
Google Colab
⚙️ Workflow
Image Upload
      │
      ▼
Read Image using OpenCV
      │
      ▼
Convert to Grayscale
      │
      ▼
Noise Removal (Median Blur)
      │
      ▼
Adaptive Thresholding
      │
      ▼
Tesseract OCR
      │
      ▼
Extracted Text Output
📂 Input
Image containing printed English text
Supported formats: JPG, JPEG, PNG
📤 Output

The program extracts and prints the recognized text from the uploaded image.

Example:

Hello World
Welcome to Optical Character Recognition.
🔍 Image Preprocessing Steps

The project improves OCR performance through:

Conversion to grayscale
Median blur for noise reduction
Adaptive Gaussian thresholding
Binary image generation for clearer text recognition
📚 Learning Objectives

This project demonstrates:

Image preprocessing with OpenCV
Noise reduction techniques
Adaptive thresholding
Optical Character Recognition (OCR)
Working with Tesseract OCR
Image upload handling in Google Colab
Extracting text from images using Python
▶️ How to Run
1. Install Dependencies
pip install pytesseract opencv-python pillow numpy
2. Install Tesseract OCR

For Google Colab, run:

!apt install tesseract-ocr

For Windows, download and install Tesseract OCR from the official installer, then add it to your system PATH or specify its installation path in your code.

3. Run the Notebook

Execute all cells and upload an image when prompted.

📈 Applications
Document digitization
Receipt and invoice scanning
Automated data entry
License plate recognition
Book and article digitization
Business card text extraction
Form processing
Archive digitization
🔮 Future Improvements
Support handwritten text recognition.
Add support for multiple languages.
Detect and correct image rotation automatically.
Draw bounding boxes around detected text.
Export extracted text to PDF or Word documents.
Build a web application using Flask or Streamlit.
Add real-time OCR using a webcam.
