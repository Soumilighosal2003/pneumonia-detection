from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('models/model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_pneumonia', methods=['POST'])
def detect_pneumonia():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        pneumonia_detected = prediction[0][0] <= 0.5
        return render_template('result.html', prediction="Pneumonia Detected" if pneumonia_detected else "No Pneumonia Detected", pneumonia_detected=pneumonia_detected, filename=filename)
    else:
        return 'Invalid file format. Please upload an image with .png, .jpg, or .jpeg extension.'

if __name__ == '__main__':
    app.run(debug=True)
