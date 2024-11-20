from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained model and vectorizer
kmeans = joblib.load('ticket_categorization_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    ticket_text = request.json.get('ticket_text')

    # Preprocess the ticket text
    text_features = vectorizer.transform([ticket_text])

    # Predict the ticket level (cluster)
    ticket_level = kmeans.predict(text_features)

    # Map cluster labels to L1, L2, L3
    level_map = {0: 'L1', 1: 'L2', 2: 'L3'}
    ticket_level = level_map.get(ticket_level[0], 'Unknown')

    return jsonify({'ticket_level': ticket_level})

if __name__ == '__main__':
    app.run(debug=True)
