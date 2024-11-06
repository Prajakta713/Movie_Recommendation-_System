# Emotion-based Movie Recommendation System 🎬😊  
This project leverages facial emotion detection to recommend movies that match the user’s current mood. By using a combination of facial expression recognition and a curated movie dataset, it provides a personalized movie recommendation experience based on detected emotions.  
  
## Table of Contents  
1. [About the Project](#AbouttheProject)  
2. [Features](#Features)  
3. [Datasets](#Datasets)  
4. [Requirements](#Requirements)  
5. [Installation](#Installation)  
6. [Usage](#Usage)  
7. [Model Training](#ModelTraining)  
8. [Results](#Results)  
  
## About the Project  
The Emotion-based Movie Recommendation System is built to offer movie suggestions tailored to users' emotional states. By capturing facial expressions and classifying emotions, the system connects these emotional states with movie genres and titles that align with or counteract the detected emotions.  
  
## Features  
- **Emotion Detection**: Uses the FER-2013 dataset to classify emotions such as happy, sad, angry, and surprised.  
- **Movie Recommendations**: Suggests movies based on emotion, using the IMDB Top 1000 Movies Dataset.  
- **Interactive Interface**: Allows users to upload a photo and receive instant recommendations.  
  
## Datasets  
- **FER-2013 Dataset** - Used to train the emotion detection model. ([Link to dataset](https://www.kaggle.com/msambare/fer2013))  
- **IMDB Top 1000 Movies Dataset** - Contains data on top-rated movies, including genre and metadata. ([Link to dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata))  
  
Note: The IMDB dataset includes columns such as 'Series_Title', 'Released_Year', 'Genre', 'IMDB_Rating', and 'Director' for filtering and recommending movies.  
  
## Requirements  
- Python 3.9+  
- Flask  
- TensorFlow/Keras  
- OpenCV  
- Pandas, NumPy  
- Scikit-Learn  
  
## Installation  
Clone the Repository
```bash  
git clone https://github.com/yourusername/Emotion-Movie-Recommendation.git  
cd Emotion-Movie-Recommendation
```  
Install Required Libraries
```bash  
pip install -r requirements.txt
```  
Download and Prepare Datasets  
  
* FER-2013 dataset for emotion detection.  
* IMDB Top 1000 Movies dataset in CSV format.  
Set up Flask Backend  
  
Configure the Flask app to run the emotion detection and recommendation model.  
  
## Usage  
Run the Application
```bash  
python app.py
```  
Access the Web Interface  
  
Open your browser and navigate to http://127.0.0.1:5000  
Upload an image of your face to receive movie recommendations based on the detected emotion.  
  
## Model Training  
The emotion detection model is trained using the FER-2013 dataset. Key steps include:  
  
* Data preprocessing (image resizing, grayscale conversion)  
* Model architecture (e.g., Convolutional Neural Network)  
* Training and validation to achieve optimal accuracy for emotion detection.  
To train the model, follow these steps:
```python  
python train_emotion_model.py
```  
## Results  
The model currently achieves an accuracy of approximately 53% on the FER-2013 dataset. This can be further improved by using more advanced neural network architectures or larger datasets. A brief discussion of the model's performance, challenges faced, and potential areas for improvement could be added here.
