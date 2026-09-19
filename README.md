# Spam Detection

A simple machine learning project for classifying text messages as **Spam** or **Ham**.

The project follows the complete workflow from understanding the dataset and cleaning the messages to feature engineering, model training, evaluation, and saving the final model pipeline.

## Technologies Used

- Python 3.11+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

```text
Dataset
   ↓
Inspection
   ↓
Preprocessing
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Saved Pipeline
```

## Project Structure

```text
Syntecxhub_Spam_Detection/
│
├── data/
│   ├── spam.csv
│   ├── spam_clean.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── spam_pipeline.joblib
│
├── 1_inspection.ipynb
├── 2_preprocessing.ipynb
├── 3_eda.ipynb
├── 4_feature_engineering.ipynb
├── 5_model.ipynb
├── requirements.txt
└── .gitignore
```

## Notebooks

### 1. Dataset Inspection

**`1_inspection.ipynb`**

This notebook is used to understand the dataset before making any changes.

It covers:

- Loading the dataset
- Checking the shape of the data
- Checking columns and data types
- Checking missing values
- Checking duplicate messages
- Checking the spam and ham distribution
- Looking at sample spam and ham messages

The original dataset columns are cleaned up and renamed for easier use in the following notebooks.

---

### 2. Preprocessing

**`2_preprocessing.ipynb`**

This notebook prepares the text messages for machine learning.

The main steps include:

- Converting text to lowercase
- Removing unnecessary characters
- Removing extra spaces
- Keeping useful numeric information
- Removing empty messages
- Removing duplicate messages
- Converting `ham` and `spam` labels into numerical values

The cleaned data is saved as:

```text
data/spam_clean.csv
```

The preprocessing is kept simple so that the effect of each step is easy to understand.

---

### 3. Exploratory Data Analysis

**`3_eda.ipynb`**

This notebook is used to understand patterns in the cleaned dataset.

The analysis includes:

- Spam vs ham distribution
- Message length
- Word count
- Number of digits in messages
- Capital letters
- Common words in spam messages
- Common words in ham messages

The dataset contains more ham messages than spam messages, so the model is evaluated using more than just accuracy.

---

### 4. Feature Engineering

**`4_feature_engineering.ipynb`**

Machine learning models cannot directly work with raw text, so the messages are converted into numerical features.

Two common text vectorization methods are explored:

- CountVectorizer
- TF-IDF

The project also compares different model and feature combinations using:

- Multinomial Naive Bayes
- Logistic Regression

The dataset is split into training and testing data before fitting the vectorizer to avoid data leakage.

The final feature and model combination is selected based on the evaluation results from the experiments.

---

### 5. Model Training and Evaluation

**`5_model.ipynb`**

This notebook contains the final model training and evaluation.

The models used are:

- Multinomial Naive Bayes
- Logistic Regression

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

The final trained pipeline is saved as:

```text
models/spam_pipeline.joblib
```

The saved pipeline can be reused to make predictions on new messages.

## Model Results

The final model is evaluated on the test data using accuracy, precision, recall, and F1-score.

The final results are available in:

```text
5_model.ipynb
```

The project focuses on using the actual evaluation results from the dataset rather than assuming that one model will always perform better.

## Using the Saved Model

The saved pipeline can be loaded with Joblib:

```python
import joblib

model = joblib.load("models/spam_pipeline.joblib")

message = "Congratulations! You won a free prize. Click now!"

prediction = model.predict([message])[0]

print("Spam" if prediction == 1 else "Ham")
```

Another example:

```python
message = "Hey, are we still meeting at 5?"

prediction = model.predict([message])[0]

print("Spam" if prediction == 1 else "Ham")
```

The saved pipeline contains the text vectorizer and trained model, so the same preprocessing used during training can be applied when making predictions.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nimishsadhu/Syntecxhub_Spam_Detection.git
```

### 2. Move into the project folder

```bash
cd Syntecxhub_Spam_Detection
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the notebooks in order

```text
1_inspection.ipynb
        ↓
2_preprocessing.ipynb
        ↓
3_eda.ipynb
        ↓
4_feature_engineering.ipynb
        ↓
5_model.ipynb
```

## Approach

The project uses traditional NLP and machine learning techniques.

The overall process is:

```text
Raw Message
     ↓
Text Cleaning
     ↓
CountVectorizer / TF-IDF
     ↓
Numerical Features
     ↓
Machine Learning Model
     ↓
Spam / Ham
```

The project does not use deep learning or transformer models. The main focus is on understanding the basic text classification workflow and building a reusable machine learning pipeline.

## Requirements

The project was tested with Python 3.11+.

The main dependencies are:

```text
pandas==3.0.2
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
joblib==1.5.3
jupyter
```

## Future Improvements

Some possible improvements for this project are:

- Try additional text features
- Tune model hyperparameters
- Test the model on a larger dataset
- Build a simple web interface for predictions
- Analyze false positives and false negatives on new messages

## Author

**Nimish Sadhu**

B.Tech Computer Science and Engineering (AI/ML)

GitHub: https://github.com/nimishsadhu
