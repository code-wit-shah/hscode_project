from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model, vectorizer, and label encoder
vectorizer = joblib.load('vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')
model = joblib.load('sgd.joblib')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_products = data.get('products')
    if not new_products:
        return jsonify({'error': 'Missing products'}), 400

    try:
        # Transform the new products
        new_sample_vector = vectorizer.transform(new_products)
        
        # Predict probabilities
        probabilities = model.predict_proba(new_sample_vector)
        
        # Get predicted class indices and labels
        predicted_classes = model.predict(new_sample_vector)
        predicted_class_labels = label_encoder.inverse_transform(predicted_classes)
        
        # Prepare response
        response = {}
        for i, product in enumerate(new_products):
            # Get probabilities for the current product
            product_probabilities = probabilities[i] * 100  # Convert to percentage
            class_indices = np.argsort(product_probabilities)[::-1][:3]  # Get indices of top 3 probabilities
            
            top_classes = label_encoder.inverse_transform(class_indices)
            top_probabilities = product_probabilities[class_indices]
            
            # Construct the result for this product
            product_result = {
                'top_predictions': [
                    {'hscode': cls, 'probability': f'{int(prob)}%'}for cls, prob in zip(top_classes, top_probabilities)
                ]
            }
            response[product] = product_result
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
