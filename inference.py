import torch
from model import FashionCNN
from data_loading import get_classes, get_data_loaders
from training import evaluate_model
from evaluation import visualize_predictions
import torch.nn as nn

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
    
def test_model(model_path, device='cpu'):
    """Тестирование загруженной модели"""
    model = load_model(model_path, device)
    _, _, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    visualize_predictions(model, test_loader, device)