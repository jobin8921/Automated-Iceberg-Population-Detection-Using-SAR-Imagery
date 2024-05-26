from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('iceberg_detection_model.h5')

# Define allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image
def preprocess_image(image_path):
    img = image.load_img(image_path, color_mode='grayscale', target_size=(75, 75))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        if os.path.exists(filepath):
            # Preprocess the uploaded image
            img_array = preprocess_image(filepath)

            # Predict using the preprocessed image
            prediction = model.predict(img_array)[0][0]
            result = 'Ship' if prediction > 0.2 else 'Iceberg'

            return render_template('result.html', filename=filename, result=result)
        else:
            return render_template('error.html', message='File not found!')
    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
