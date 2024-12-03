from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import zipfile
import os

app = Flask(__name__)

# Step 1: Extract and load the dataset
zip_path = "spam_mail_filter.zip"
extract_to = "extracted_data"
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

data_path = os.path.join(extract_to, "spam_or_not_spam.csv")
data = pd.read_csv(data_path)
data = data.dropna(subset=["email", "label"])
data['label'] = data['label'].astype(int)

# Step 2: Train the model
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['email'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# Step 3: Define an API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the message from the request
        content = request.json
        message = content.get('message', '')

        # Predict using the trained model
        vec_message = vectorizer.transform([message])
        prediction = model.predict(vec_message)[0]
        result = "Spam" if prediction == 1 else "Ham"

        return jsonify({'message': message, 'prediction': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
