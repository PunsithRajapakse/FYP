from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_absolute_error, accuracy_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['TEST_IMAGE_FOLDER'] = 'static/test_images/'

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEST_IMAGE_FOLDER'], exist_ok=True)

# Dictionary of spherical foods with density and calorie-per-gram values
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

# Load pre-trained MobileNetV2 model
model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
with open(labels_path, "r") as f:
    class_labels = [line.strip().lower() for line in f.readlines()]

def preprocess_image(image_path):
    """Preprocess image for edge detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return image, edges

def estimate_radius(edges, food_type):
    """Estimate radius of spherical food items."""
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=200)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        detected_radius = circles[0][0][2] / 10
        avg_radius = food_data[food_type]['avg_radius_cm']
        return (detected_radius + avg_radius) / 2
    return food_data[food_type]['avg_radius_cm']

def classify_food(image_path):
    """Classify the uploaded food image using MobileNetV2."""
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
    """Calculate estimated weight and calories of food."""
    if food_type not in food_data:
        return None, None

    volume = (4/3) * np.pi * (radius_cm ** 3)
    density = food_data[food_type]['density']
    calories_per_g = food_data[food_type]['calories_per_g']

    weight = min(volume * density, 1000)  # Limit max weight to 1000g
    calories = weight * calories_per_g
    return weight, calories

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and calorie estimation."""
    if request.method == 'POST':
        file = request.files['file']
        if not file or file.filename == '':
            return render_template('index.html', error='No file uploaded.')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        food_type = classify_food(filepath)
        if not food_type:
            return render_template('index.html', error='Could not identify the food type.')

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

@app.route('/check_accuracy')
def check_accuracy():
    """Calculate classification and calorie estimation accuracy using real test images."""
    
    test_images = os.listdir(app.config['TEST_IMAGE_FOLDER'])

    actual_foods = []
    predicted_foods = []
    actual_calories = []
    predicted_calories = []

    for image_file in test_images:
        image_path = os.path.join(app.config['TEST_IMAGE_FOLDER'], image_file)
        actual_food = image_file.split("_")[0]  # Assuming test image names like "apple_1.jpg"
        actual_calories.append(food_data[actual_food]["calories_per_g"] * 100)  # Assume 100g food

        # Get model prediction
        predicted_food = classify_food(image_path)
        if predicted_food:
            predicted_foods.append(predicted_food)
            weight, predicted_cal = calculate_calories(predicted_food, food_data[predicted_food]['avg_radius_cm'])
            predicted_calories.append(predicted_cal)
        else:
            predicted_foods.append("unknown")
            predicted_calories.append(0)

        actual_foods.append(actual_food)

    # Calculate classification accuracy and calorie estimation error
    classification_accuracy = accuracy_score(actual_foods, predicted_foods) * 100
    calorie_error = mean_absolute_error(actual_calories, predicted_calories)

    return jsonify({
        "classification_accuracy": f"{classification_accuracy:.2f}%",
        "calorie_estimation_error": f"{calorie_error:.2f} kcal"
    })

if __name__ == '__main__':
    app.run(debug=True)
