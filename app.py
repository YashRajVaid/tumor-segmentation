"""
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
from cv import region_splitting_merging,merge,split,preprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    threshold_mean = float(request.form['threshold_mean'])
    threshold_std = float(request.form['threshold_mean'])
    min_size = int(request.form['min_size'])

    # Read the image file
    img = Image.open(file)
    img = img.convert('L')
    img = np.array(img)
    #preprocess_img = preprocess(img)
    segmented_img = region_splitting_merging(img, threshold_std=0.02, threshold_mean=0.7)
    # Call region split-merge from the image_processing module
    #segmented_img = region_split_merge(img, threshold=threshold, min_size=min_size)

    # Convert the resulting segmented image back to a PIL image
    segmented_pil = Image.fromarray(segmented_img.astype(np.uint8))

    # Convert to base64 for frontend display
    buffered = BytesIO()
    segmented_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'segmented_img': img_str})

if __name__ == '__main__':
    app.run(debug=True)

"""
# app.py
from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from cv import segment_image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(image_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    threshold = int(request.form.get('threshold', 10))
    min_size = int(request.form.get('min_size', 16))
    merge_threshold = int(request.form.get('merge_threshold', 15))

    image = cv2.imread(filepath)
    segmented = segment_image(image, threshold, min_size, merge_threshold)

    segmented_path = os.path.join(RESULT_FOLDER, 'segmented_' + filename)
    cv2.imwrite(segmented_path, segmented)

    return render_template('index.html', original=filepath, segmented=segmented_path)


if __name__ == '__main__':
    app.run(debug=True)

"""
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from cv import region_split_merge  # Import the function you implemented

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check valid file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle image upload and segmentation
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve parameters from sliders
        threshold = int(request.form['threshold'])
        min_size = int(request.form['min_size'])
        threshold_mean = int(request.form['threshold_mean'])
        threshold_std = int(request.form['threshold_std'])

        # Retrieve uploaded image
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read the image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Perform region splitting and merging with parameters
            segmented_image = region_split_merge(image, threshold, min_size, threshold_mean, threshold_std)

            # Save the segmented image for displaying in the frontend
            segmented_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_' + filename)
            cv2.imwrite(segmented_filepath, segmented_image)

            # Return the original and segmented images to the frontend
            return render_template('index.html', original_image=filepath, segmented_image=segmented_filepath)

    return render_template('index.html', original_image=None, segmented_image=None)

if __name__ == '__main__':
    app.run(debug=True)
"""