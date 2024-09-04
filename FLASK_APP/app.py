from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model
model = keras.models.load_model(r'C:\OLD D DRIVE\PLANT_DISEASE_NEW\Plant_dataset\keras-model\plant_disease.keras')

# Ensure the uploads directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and display
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        # Save the file to the static/uploads folder
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Generate the URL for displaying the image
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        
        return render_template('index.html', image_url=image_url)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.form['image_path']

    # Construct the full file path for the image
    filepath = os.path.join(app.root_path, image_url.lstrip('/'))

    # Load and preprocess the image
    def process_image(img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape
        img_array = img_array / 255.0  # Normalize the image
        return img_array

    img = process_image(filepath)

    # Make prediction
    prediction = model.predict(img)
    result = np.argmax(prediction, axis=1)[0]
    
    # Decode the predicted label
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['Healthy', 'Powdery', 'Rust'])  # Replace with your actual classes
    predicted_label = label_encoder.inverse_transform([result])[0]

    return render_template('result.html', result=predicted_label)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
