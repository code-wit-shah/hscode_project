# hscode_project
Project :HS Code classification Project with flask api

Import Libraries: Load necessary Python libraries for data handling, natural language processing (NLP), machine learning, and model persistence.

Load Data: Read a CSV file into a pandas DataFrame containing product descriptions and commodity codes.

Preprocess Data:

Encode the target labels (commodity codes) using LabelEncoder.
Vectorize product descriptions with TfidfVectorizer, filtering out rare classes that occur only once.
Split Data: Divide the data into training and test sets.

Train Model: Train an SGDClassifier using the training data.

Evaluate Model:

Calculate accuracy, classification report, and confusion matrix on the test set.
Compute precision, recall, and F1-score.
Cross-Validation: Perform cross-validation to assess the model's robustness.

Save and Load Models:

Save the vectorizer, label encoder, and model to disk using joblib.
Load these saved objects for making predictions on new data.
Predict New Data:

Transform new product descriptions using the loaded vectorizer.
Predict and decode commodity codes for these new products
API:
Flask Application for Product Classification

Purpose: Classifies product descriptions into categories using a pre-trained machine learning model.

Dependencies: Flask, joblib, numpy.

Model Components:

vectorizer: Transforms text data into numerical format.
label_encoder: Converts model outputs into human-readable labels.
model: Machine learning model for classification.
Endpoint: /predict (POST request)

Input: JSON with a list of product descriptions.
Output: JSON with top 3 predicted categories and their probabilities (in percentage) for each product.
Error Handling: Returns a 400 status code for missing products or a 500 status code for internal errors.

Run: The app listens on port 5000.
