# Playing Card Classification with PyTorch

Deep learning model for classifying playing cards using PyTorch with GPU optimization and a web interface.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run Web Interface
```bash
python app.py
```

Then open browser to: **http://127.0.0.1:5000**

## Features

- 🚀 **GPU-Optimized Training** - Automatic CUDA detection and GPU acceleration.
- 🎯 **Transfer Learning** - Pre-trained ResNet18 for high accuracy.
- 🌐 **Web Interface** - Beautiful Flask app for easy predictions.
- 📊 **Data Augmentation** - Rotation, flipping, color jitter.
- 💾 **Auto-Save Best Model** - Saves model with highest validation accuracy.
- 📈 **Training Visualization** - Loss and accuracy plots.

## Dataset Structure

```
data/
├── train/    # 7,624 images across 53 classes
├── valid/    # 265 images for validation
└── test/     # 265 images for testing
```

## Files

- `train_model.py` - Training pipeline and model definition.
- `app.py` - Flask web application for predictions.
- `templates/index.html` - Web interface (single page).
- `requirements.txt` - Python dependencies.
- `best_card_model.pth` - Trained model (generated after training).
- `training_history.png` - Training metrics plot (generated).

## Training Configuration

Edit parameters in `train_model.py` main function:
- `BATCH_SIZE` - Images per batch (default: 32).
- `NUM_EPOCHS` - Training epochs (default: 20).
- `LEARNING_RATE` - Optimizer learning rate (default: 0.001).
- `NUM_WORKERS` - Data loading workers (default: 4).

## Model Architecture

- **Base**: ResNet18 pre-trained on ImageNet.
- **Custom Head**: Linear(512) → ReLU → Dropout(0.3) → Linear(53).
- **Trainable Params**: ~5 million.
- **Expected Accuracy**: 90-95% on test set.

## Web Interface Features

- Drag-and-drop image upload.
- Real-time predictions with confidence scores.
- Top 5 alternative predictions.
- Responsive, modern design.
- GPU/CPU automatic selection.

## Making Predictions (Python)

```python
from train_model import CardClassifier, predict_single_image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CardClassifier(num_classes=53)
model.load_state_dict(torch.load('best_card_model.pth'))
model = model.to(device)

# Get class names
import os
class_names = sorted(os.listdir('data/train'))

# Predict
predicted_class, confidence = predict_single_image(
    model, 'image.jpg', class_names, device
)
print(f'{predicted_class}: {confidence:.2f}%')
```

## GPU Optimizations

- Automatic CUDA device detection.
- Pin memory for faster CPU→GPU transfers.
- Non-blocking data transfers.
- Efficient batch processing.

## Expected Performance

**GPU (RTX 2080 SUPER)**:
- Training Time: ~5-10 minutes (20 epochs).
- Test Accuracy: 90-95%.

**CPU**:
- Training Time: ~1-2 hours (20 epochs).
- Test Accuracy: Same as GPU.

## Troubleshooting

**GPU Out of Memory**:
```python
# Reduce BATCH_SIZE in train_model.py
BATCH_SIZE = 16  # or 8
```

**Model Not Found** (when running web app):
```bash
# Train the model first
python train_model.py
```

**Port Already in Use**:
```python
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for GPU training)
