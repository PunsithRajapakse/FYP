from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary of spherical foods with their density and calorie-per-gram values
food_data = {
    'apple': {'density': 0.8, 'calories_per_g': 0.52, 'avg_radius_cm': 4},
    'mango': {'density': 1.0, 'calories_per_g': 0.60, 'avg_radius_cm': 5},
    'orange': {'density': 0.94, 'calories_per_g': 0.47, 'avg_radius_cm': 4},
    'peach': {'density': 0.88, 'calories_per_g': 0.39, 'avg_radius_cm': 3.5},
    'plum': {'density': 0.96, 'calories_per_g': 0.46, 'avg_radius_cm': 3},
    'grapefruit': {'density': 0.95, 'calories_per_g': 0.42, 'avg_radius_cm': 6},
    'lemon': {'density': 0.92, 'calories_per_g': 0.29, 'avg_radius_cm': 4},
    'lime': {'density': 0.91, 'calories_per_g': 0.30, 'avg_radius_cm': 3},
    'pomegranate': {'density': 1.1, 'calories_per_g': 0.83, 'avg_radius_cm': 4.5},
    'melon': {'density': 0.95, 'calories_per_g': 0.34, 'avg_radius_cm': 12}
}

# Load a pre-trained image classification model
model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
with open(labels_path, "r") as f:
    class_labels = [line.strip().lower() for line in f.readlines()]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return image, edges

def estimate_radius(edges, food_type):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=200)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        detected_radius = circles[0][0][2] / 10
        avg_radius = food_data[food_type]['avg_radius_cm']
        return (detected_radius + avg_radius) / 2
    return food_data[food_type]['avg_radius_cm']

def classify_food(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class_index]
    
    for food in food_data.keys():
        if food in predicted_class_name:
            return food
    return None

def calculate_calories(food_type, radius_cm):
    if food_type not in food_data:
        return None, None
    
    volume = (4/3) * np.pi * (radius_cm ** 3)
    density = food_data[food_type]['density']
    calories_per_g = food_data[food_type]['calories_per_g']
    
    weight = volume * density
    weight = min(weight, 1000)
    calories = weight * calories_per_g
    return weight, calories


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        food_type = classify_food(filepath)
        if not food_type:
            return render_template('index.html', error='Food type could not be identified.')
        
        image, edges = preprocess_image(filepath)
        radius_cm = estimate_radius(edges, food_type)
        weight, calories = calculate_calories(food_type, radius_cm)
        
        return render_template('index.html',
                               food_type=food_type.capitalize(),
                               radius=f"{radius_cm:.2f} cm",
                               weight=f"{weight:.2f} g",
                               calories=f"{calories:.2f} kcal",
                               image_url=filepath)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
