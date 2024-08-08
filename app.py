from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import torch
from torchvision.models import resnet18
from torchvision import transforms
import gdown

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Google Drive file ID and model weights file path
file_id = '1IgTz9feg-lU6Uq_BJ-vsvVQnkdg98BRC'
model_weights_path = 'model_weights.pth'

# Download the model weights from Google Drive if not already downloaded
gdown.download(f'https://drive.google.com/uc?id={file_id}', model_weights_path, quiet=False)

# Number of classes in your classification problem
num_classes = 5

# Load the ResNet-18 model
model = resnet18(pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

# Define class names for the classes you are classifying
class_names = [
    "Early_Blight",
    "Late_Blight",
    "Septoria_Leaf",
    "Yellow_Leaf",
    "Healthy"
]

# Define the transformation for preprocessing images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        if 'image' not in data:
            raise ValueError('Image data is missing in the request.')

        image_data = data['image']
        if not image_data.startswith('data:image/'):
            raise ValueError('Invalid image data format.')

        # Extract the base64 part of the image data
        image_data = image_data.split(',')[1]  # Remove the data URL scheme

        # Decode base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.convert('RGB')  # Ensure image is in RGB mode

        # Preprocess the image
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return jsonify({'result': predicted_class})

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        print(f"Error: {error_message}")
        return jsonify({'error': str(e)}), 500


