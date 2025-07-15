import torch
import torchvision.transforms as transforms
from PIL import Image
from .model import FashionCNN
from .data_loading import get_classes, get_data_loaders
from .training import evaluate_model
from .evaluation import visualize_predictions
import torch.nn as nn
import matplotlib.pyplot as plt

def load_model(model_path, device='cpu'):
    """Загрузка сохраненной модели"""
    model = FashionCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image, device='cpu'):
    """Предсказание для одного изображения"""
    if len(image.shape) == 2:
        image = image.float()
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.float()
        image = image.unsqueeze(0)
    else:
        raise ValueError("Изображение должно быть 2D (H, W) или 3D (1, H, W)")
    
    image = image.to(device)
    image = (image - 0.5) / 0.5
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    classes = get_classes()
    return {
        'class': classes[predicted.item()],
        'class_idx': predicted.item(),
        'probabilities': {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
    }
    
def test_model(model_path, device='cpu', image_path=None):
    """Тестирование загруженной модели"""
    model = load_model(model_path, device)
    
    if image_path:
        # Тестирование на конкретном изображении
        result = predict_external_image(model, image_path, device)
        print(f"\nPrediction for {image_path}:")
        print(f"Class: {result['class']}")
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.2%}")
        
        # Визуализация изображения и предсказания
        visualize_single_prediction(image_path, result, model, device)
    else:
        # Стандартное тестирование на тестовом наборе
        _, _, test_loader = get_data_loaders()
        criterion = nn.CrossEntropyLoss()
        
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        
        visualize_predictions(model, test_loader, device)

def predict_external_image(model, image_path, device='cpu'):
    """Предсказание для произвольного изображения"""
    image = Image.open(image_path).convert('L')  
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    classes = get_classes()
    return {
        'class': classes[predicted.item()],
        'class_idx': predicted.item(),
        'probabilities': {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
    }

def visualize_single_prediction(image_path, prediction, model, device='cpu'):
    """Визуализация одного изображения с предсказанием"""
    
    original_img = Image.open(image_path).convert('L')
    
    
    plt.figure(figsize=(8, 4))
    
    # Отображаем оригинальное изображение
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    
    processed_img = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    processed_img = transform(processed_img).squeeze().numpy()
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"Predicted: {prediction['class']}\n"
              f"Confidence: {prediction['probabilities'][prediction['class']]:.1%}")
    plt.axis('off')
    
    # Добавляем информацию о вероятностях всех классов
    plt.figtext(0.5, 0.05, 
                "\n".join([f"{cls}: {prob:.2%}" for cls, prob in prediction['probabilities'].items()]),
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    plt.suptitle("Image Classification Result", fontsize=14)
    plt.tight_layout()
    plt.show()