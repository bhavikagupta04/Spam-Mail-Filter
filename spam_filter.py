import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import zipfile
import os

# Step 1: Extract the ZIP file
zip_path = "spam_mail_filter.zip"
extract_to = "extracted_data"
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Dataset extracted.")

# Step 2: Load the dataset
data_path = os.path.join(extract_to, "spam_or_not_spam.csv")
data = pd.read_csv(data_path)

# Step 3: Handle missing values
data = data.dropna(subset=["email", "label"])  # Remove rows with missing values

# Step 4: Preprocess labels and messages
data['label'] = data['label'].astype(int)  # Ensure labels are integers (0 = Ham, 1 = Spam)
messages = data['email']
labels = data['label']

# Step 5: Vectorize the messages
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(messages)

# Step 6: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 7: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 9: Create a function to predict new messages
def predict_message(message):
    vec_message = vectorizer.transform([message])
    prediction = model.predict(vec_message)[0]
    return "Spam" if prediction == 1 else "Ham"

# Step 10: Test the function
test_message = input("Enter a message to classify: ")
print(f"The message is classified as: {predict_message(test_message)}")
