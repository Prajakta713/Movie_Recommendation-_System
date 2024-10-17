from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2
import pandas as pd

app = Flask(__name__)

# Load your emotion detection model
model = load_model('D:/emotion_movie_recommender/model - Copy.h5')  

# Load your movie dataset
movies_df = pd.read_csv('D:/emotion_movie_recommender/imdb_top_1000_with_emotions.csv')

# Print the columns to check if 'Genre' exists
print("Columns in the movies DataFrame:", movies_df.columns)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 48x48 pixels
    image = cv2.resize(image, (48, 48))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Expand dimensions to match model input shape (48, 48, 1)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (1)
    # Normalize the image
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension (1, 48, 48, 1)

# Function to detect emotion from the image
def detect_emotion(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    emotion = np.argmax(predictions[0])
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return emotion_labels[emotion]

# Function to recommend movies based on detected emotion
def recommend_movies(emotion):
    if emotion == 'happy':
        return movies_df[
            (movies_df['Comedy'] == 1) | 
            (movies_df['Family'] == 1) | 
            (movies_df['Musical'] == 1) | 
            (movies_df['Romantic Comedy'] == 1)
        ].sample(5)
    elif emotion == 'sad':
        return movies_df[
            (movies_df['Drama'] == 1) | 
            (movies_df['Romance'] == 1) | 
            (movies_df['Biography'] == 1)
        ].sample(5)
    elif emotion == 'fear':
        return movies_df[
            (movies_df['Horror'] == 1) | 
            (movies_df['Psychological Thriller'] == 1) | 
            (movies_df['Mystery'] == 1)
        ].sample(5)
    elif emotion == 'anger':
        return movies_df[
            (movies_df['Crime'] == 1) | 
            (movies_df['Revenge'] == 1) | 
            (movies_df['War'] == 1) | 
            (movies_df['Dystopian Sci-Fi'] == 1)
        ].sample(5)
    elif emotion == 'surprise':
        return movies_df[
            (movies_df['Sci-Fi'] == 1) | 
            (movies_df['Fantasy'] == 1) | 
            (movies_df['Supernatural'] == 1) | 
            (movies_df['Mystery'] == 1)
        ].sample(5)
    elif emotion == 'neutral':
        return movies_df.sample(5)
    elif emotion == 'disgust':
        return movies_df[movies_df['Documentary'] == 1].sample(5)
    else:  # Fallback for any unrecognized emotion
        return movies_df.sample(5)


@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_movies = None
    detected_emotion = None

    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect emotion from the image
        detected_emotion = detect_emotion(img)
        print(f"Detected Emotion: {detected_emotion}")  # Debugging

        # Recommend movies based on detected emotion
        recommended_movies = recommend_movies(detected_emotion)
        
        # Check if recommended_movies is None
        if recommended_movies is None or recommended_movies.empty:
            recommended_movies = pd.DataFrame(columns=movies_df.columns)  # Empty DataFrame to avoid errors

    # Pass detected_emotion and recommended_movies to the template
    return render_template('index.html', detected_emotion=detected_emotion, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
