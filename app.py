from flask import Flask, request, jsonify
import os
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the path for saving uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path for models
MODEL_FOLDER = 'deep-text-recognition-benchmark'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Define the models
models = {
    'None-ResNet-None-CTC.pth': 'https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
    'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/uc?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
    'None-VGG-None-CTC.pth': 'https://drive.google.com/uc?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
    'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/uc?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
    'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/uc?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
    'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/uc?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
}

# Download models if they don't already exist
for k, v in models.items():
    model_path = os.path.join(MODEL_FOLDER, k)
    if not os.path.exists(model_path):
        os.system(f'gdown -O {model_path} "{v}"')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Assuming the language code is part of the form data
        language_code = request.form.get('language', 'eng')  # Default to English if not provided

        # Here you can add conditions to select the model based on language_code
        # For simplicity, we use one model for demonstration
        model_path = os.path.join(MODEL_FOLDER, "TPS-ResNet-BiLSTM-Attn.pth")

        # Run the model
        result = os.popen(f'CUDA_VISIBLE_DEVICES=0 python3 {MODEL_FOLDER}/demo.py \
                            --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
                            --image_folder {UPLOAD_FOLDER}/ --saved_model {model_path}').read()

        # Here, you should parse the output of the model to extract the recognized text
        # For simplicity, we just return the raw result
        os.remove(file_path)
        return jsonify({'recognized_text': extract_predicted_labels(result)})

# Function to extract predicted labels from the recognized text
def extract_predicted_labels(recognized_text):
    # Split the text by lines
    lines = recognized_text.split('\n')
    # Initialize an empty list to hold the predicted labels
    predicted_labels = []
    # Regex pattern to match lines with predicted labels
    pattern = re.compile(r'\t([^\t]+)\t')
    # Start processing lines after the header part
    for line in lines:
        # Check if the line contains predicted label information
        match = pattern.search(line)
        if match:
            # Add the extracted label to the list
            predicted_labels.append(match.group(1))
    return predicted_labels[1].strip()


if __name__ == '__main__':
    app.run(debug=True)
