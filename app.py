"""Flask web application for playing card classification.

This app provides a simple web interface to upload card images and get predictions
from the trained PyTorch model.
"""

import os
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from train_model import CardClassifier

# Initialize Flask app.
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size.
app.config['UPLOAD_FOLDER'] = 'uploads'

# Allowed file extensions for upload.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Global variables for model and classes.
model = None
class_names = None
device = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension.
    
    Args:
        filename: Name of the uploaded file.
        
    Returns:
        True if file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model and class names.
    
    Returns:
        Tuple of (model, class_names, device).
    """
    global model, class_names, device
    # Determine device to use.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Load class names from directory structure.
    train_dir = os.path.join('data', 'train')
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    # Initialize and load the model.
    model = CardClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load('best_card_model.pth', map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode.
    print(f'Model loaded successfully with {len(class_names)} classes.')
    return model, class_names, device

def predict_image(image_path):
    """Make a prediction on an uploaded image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary with prediction results.
    """
    # Define image transformation matching training.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    # Load and preprocess the image.
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension.
    # Move to device and make prediction.
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        # Get top 5 predictions.
        top_probs, top_indices = torch.topk(probabilities, k=5)
    # Format results.
    results = {
        'top_prediction': class_names[top_indices[0][0].item()],
        'confidence': f"{top_probs[0][0].item() * 100:.2f}",
        'top_5': [
            {
                'class': class_names[idx.item()],
                'confidence': f"{prob.item() * 100:.2f}"
            }
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
    }
    return results

@app.route('/')
def index():
    """Render the main page with upload form.
    
    Returns:
        Rendered HTML template.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction results.
    
    Returns:
        JSON response with prediction results or error message.
    """
    # Check if file was uploaded.
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    # Check if file was selected.
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    # Check if file type is allowed.
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG.'}), 400
    try:
        # Save the uploaded file.
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Make prediction.
        results = predict_image(filepath)
        # Clean up uploaded file.
        os.remove(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Create upload directory if it doesn't exist.
    os.makedirs('uploads', exist_ok=True)
    # Load the model at startup.
    print('Loading model...')
    load_model()
    print('Model ready!')
    # Run the Flask app.
    print('\nStarting Flask server...')
    print('Open your browser and go to: http://127.0.0.1:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
