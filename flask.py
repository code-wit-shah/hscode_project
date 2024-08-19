from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained models and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')
model = joblib.load('sgd.joblib')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'products' not in data:
        return jsonify({'error': 'No products field provided'}), 400
    
    products = data['products']
    
    if not isinstance(products, list):
        return jsonify({'error': 'Products should be a list of strings'}), 400
    
    # Preprocess the products
    preprocessed_products = [preprocess_text(product) for product in products]
    
    # Vectorize the products
    product_vectors = vectorizer.transform(preprocessed_products)
    
    # Predict the probabilities
    probabilities = model.predict_proba(product_vectors)
    
    # Get top 3 HS codes and their probabilities for each product
    top_n = 3
    predictions = []
    for prob in probabilities:
        top_indices = prob.argsort()[-top_n:][::-1]
        top_codes = label_encoder.inverse_transform(top_indices)
        top_probs = prob[top_indices]
        predictions.append([
            {'hs_code': code, 'probability': prob} for code, prob in zip(top_codes, top_probs)
        ])
    
    # Return the predictions
    return jsonify({'predictions': predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
