from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Load vocabulary
with open("tfidfvectorizer.pkl", "rb") as file:
    vocab = pickle.load(file)  # This should be the vocabulary dictionary

# Create TfidfVectorizer with the pre-loaded vocabulary
vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=vocab)
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    user_input = ''
    
    if request.method == 'POST':
        user_input = request.form['text']
        # Transform input text using the loaded vectorizer
        transformed_input = vectorizer.transform([user_input])
        # Predict using the loaded model
        prediction = model.predict(transformed_input)[0]

    return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
