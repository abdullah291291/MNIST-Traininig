import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from model import CNN_Model

app = Flask(__name__)

# Load the trained model
model = CNN_Model()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the transformation for incoming images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict_digit():
    try:
        # Get the uploaded image
        image_file = request.files['image']
        if image_file:
            # Open and preprocess the image
            image = Image.open(image_file).convert("L")
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension

            # Make the prediction
            with torch.no_grad():
                prediction = model(image)

            # Get the predicted digit (class)
            _, predicted_class = prediction.max(1)
            result = {
                'digit': predicted_class.item(),
                'confidence': prediction[0][predicted_class].item()
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No image file uploaded'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
