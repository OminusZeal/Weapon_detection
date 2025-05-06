from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import os
from detection import detect_weapons_in_image
import json
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 16 MB limit
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
        
        img = request.files['image']

        # Check if the file is allowed
        if img.filename == '' or not allowed_file(img.filename):
            return "Invalid file type. Only jpg, jpeg, and png are allowed.", 400
        
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        image_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(image_path)

        # Run detection and get the result path
        result_path = detect_weapons_in_image(image_path)
        result_filename = os.path.basename(result_path)

        # Redirect to result page with filename for display
        return redirect(url_for('result', filename=result_filename))
    
    return render_template("index.html")

@app.route('/result/<filename>')
def result(filename):
    # Pass the filename to the result page
    return render_template("result.html", filename=filename)

@app.route('/live-camera')
def live_camera():
    return render_template("live_camera.html")  # Create this template to show webcam feed

@app.route('/outputs/<filename>')
def send_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/capture', methods=['POST'])
def capture():
    try:
        data_url = request.form.get('image')
        if not data_url:
            return {"error": "No image data received"}, 400

        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        image_path = os.path.join(UPLOAD_FOLDER, 'captured.png')
        with open(image_path, 'wb') as f:
            f.write(image_data)

        if not os.path.exists(image_path):
            return {"error": "Failed to save image"}, 500

        result_path = detect_weapons_in_image(image_path)

        if result_path is None or not os.path.exists(result_path):
            return {"error": "Detection failed"}, 500

        result_filename = os.path.basename(result_path)
        return {"result_image": result_filename}

    except Exception as e:
        print("[ERROR in /capture]", str(e))
        return {"error": str(e)}, 500


@app.route('/detection_result/<filename>')
def detection_result(filename):
    model = request.args.get('model', 'SVM')  # Default to SVM if not provided

    # Load evaluation metrics from JSON
    try:
        with open('evaluation_metrics.json', 'r') as f:
            results = json.load(f)
    except Exception as e:
        return f"Error loading evaluation metrics: {str(e)}", 500

    if model not in results:
        return f"Model '{model}' not found in evaluation metrics.", 400

    model_results = results[model]
    accuracy = model_results["accuracy"] * 100
    confusion_matrix = model_results["confusion_matrix"]
    avg_iou = model_results["avg_iou"]

    return render_template('detection_result.html',
                           filename=filename,
                           accuracy=accuracy,
                           confusion_matrix=confusion_matrix,
                           avg_iou=avg_iou,
                           model=model)

if __name__ == "__main__":
    app.run(debug=True)
