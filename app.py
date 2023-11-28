from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

# Load the pre-trained model from the saved Keras model file
model = load_model('C:\\Users\\shikh\\OneDrive\\Desktop\\work\\transformer_model_for_classifcation\\transformer_sentiment_model.h5')

# Tokenizer for text processing
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # Adjust parameters as needed

# Function to clean and preprocess text for prediction
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join(word for word in cleaned_text.split() if word not in stop_words)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']

    # Preprocess input
    cleaned_input = preprocess_text(user_input)

    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([cleaned_input])
    padded_sequence = pad_sequences(sequence, maxlen=100, truncating='post', padding='post')

    # Make sentiment prediction
    prediction_prob = model.predict(padded_sequence)[0][0]

    # Determine sentiment based on prediction probability
    if 0.4 <= prediction_prob <= 0.55:
        sentiment = "Neutral"
    elif prediction_prob > 0.55:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return render_template('index.html', user_input=user_input, prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)