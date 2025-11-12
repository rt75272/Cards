"""Playing Card Classification Model using PyTorch.

This script trains a convolutional neural network to classify playing cards
using the dataset stored in the data folder.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class CardClassifier(nn.Module):
    """Convolutional Neural Network for playing card classification.
    
    Uses a pre-trained ResNet18 model with transfer learning for better
    accuracy and faster convergence.
    """
    
    def __init__(self, num_classes=53):
        """Initialize the model architecture.
        
        Args:
            num_classes: Number of card classes to predict (default: 53).
        """
        super(CardClassifier, self).__init__()
        # Load pre-trained ResNet18 model.
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze early layers to retain pre-trained features.
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        # Replace the final fully connected layer for our number of classes.
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Output tensor with class predictions.
        """
        return self.model(x)

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """Create PyTorch data loaders for training, validation, and testing.
    
    Args:
        data_dir: Path to the data directory containing train/valid/test folders.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        Dictionary containing train, valid, and test data loaders.
    """
    # Define data transformations for training data with augmentation.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    # Define data transformations for validation and test data without augmentation.
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    # Load datasets from directory structure.
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'valid'),
        transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=eval_transform
    )
    # Create data loaders with GPU-optimized settings.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speeds up CPU to GPU transfer.
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
        'classes': train_dataset.classes
    }

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Device to run training on (CPU or GPU).
        
    Returns:
        Average training loss and accuracy for the epoch.
    """
    model.train()  # Set model to training mode.
    running_loss = 0.0
    correct = 0
    total = 0
    # Iterate through batches with progress bar.
    for inputs, labels in tqdm(train_loader, desc='Training'):
        # Move data to GPU for faster computation.
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # Zero the parameter gradients.
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization.
        loss.backward()
        optimizer.step()
        # Track statistics.
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, valid_loader, criterion, device):
    """Validate the model on the validation set.
    
    Args:
        model: The neural network model.
        valid_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on (CPU or GPU).
        
    Returns:
        Average validation loss and accuracy.
    """
    model.eval()  # Set model to evaluation mode.
    running_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient computation for validation.
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc='Validating'):
            # Move data to GPU.
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Forward pass only.
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Track statistics.
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def train_model(model, data_loaders, num_epochs=30, learning_rate=0.001, device='cuda'):
    """Complete training loop for the model.
    
    Args:
        model: The neural network model to train.
        data_loaders: Dictionary containing train and valid data loaders.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        device: Device to train on ('cuda' or 'cpu').
        
    Returns:
        Trained model and training history.
    """
    # Move model to GPU for faster training.
    model = model.to(device)
    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler for adaptive learning.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    # Track training history.
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    best_valid_acc = 0.0
    # Training loop.
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)
        # Train for one epoch.
        train_loss, train_acc = train_one_epoch(
            model, data_loaders['train'], criterion, optimizer, device
        )
        # Validate the model.
        valid_loss, valid_acc = validate(
            model, data_loaders['valid'], criterion, device
        )
        # Update learning rate based on validation loss.
        scheduler.step(valid_loss)
        # Record history.
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        # Print epoch results.
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%')
        # Save best model.
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'best_card_model.pth')
            print(f'✓ Model saved with validation accuracy: {valid_acc:.2f}%')
    return model, history

def test_model(model, test_loader, device='cuda'):
    """Evaluate the model on the test set.
    
    Args:
        model: The trained neural network model.
        test_loader: DataLoader for test data.
        device: Device to run testing on ('cuda' or 'cpu').
        
    Returns:
        Test accuracy percentage.
    """
    model.eval()  # Set model to evaluation mode.
    correct = 0
    total = 0
    # Disable gradient computation for testing.
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            # Move data to GPU.
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Forward pass.
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Track statistics.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    return test_acc

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save the training history graphs.
    
    Args:
        history: Dictionary containing training and validation metrics.
        save_path: Path to save the plot image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Plot loss.
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['valid_loss'], label='Valid Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    # Plot accuracy.
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['valid_acc'], label='Valid Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nTraining history plot saved to {save_path}')

def predict_single_image(model, image_path, classes, device='cuda'):
    """Make a prediction on a single image.
    
    Args:
        model: The trained neural network model.
        image_path: Path to the image file.
        classes: List of class names.
        device: Device to run prediction on ('cuda' or 'cpu').
        
    Returns:
        Predicted class name and confidence score.
    """
    from PIL import Image
    
    # Load and transform the image.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension.
    # Move to GPU and make prediction.
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100
    return predicted_class, confidence_score

def main():
    """Main function to orchestrate the training pipeline."""
    # Set random seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    # Check if GPU is available and set device accordingly.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    # Configuration parameters.
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4
    # Load data.
    print('\nLoading datasets...')
    data_loaders = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    num_classes = len(data_loaders['classes'])
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {data_loaders["classes"][:5]}... (showing first 5)')
    # Initialize model.
    print('\nInitializing model...')
    model = CardClassifier(num_classes=num_classes)
    # Count trainable parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params:,}')
    # Train the model.
    print('\nStarting training...')
    model, history = train_model(
        model, data_loaders, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        device=device
    )
    # Plot training history.
    plot_training_history(history)
    # Load best model and evaluate on test set.
    print('\nLoading best model for testing...')
    model.load_state_dict(torch.load('best_card_model.pth'))
    test_accuracy = test_model(model, data_loaders['test'], device)
    # Example prediction on a single image.
    print('\n' + '='*60)
    print('Example Prediction:')
    print('='*60)
    # Find a test image to demonstrate prediction.
    test_image_path = 'data/test/ace of clubs/001.jpg'
    if os.path.exists(test_image_path):
        predicted_class, confidence = predict_single_image(
            model, test_image_path, data_loaders['classes'], device
        )
        print(f'Image: {test_image_path}')
        print(f'Predicted: {predicted_class}')
        print(f'Confidence: {confidence:.2f}%')
    print('\nTraining complete!')

# The big red activation button.
if __name__ == '__main__':
    main()
