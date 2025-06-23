import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import time
import json
import os
from .data_loading import get_classes

def plot_training_history(history):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, loader, device='cpu', num_images=15):
    """Визуализация предсказаний модели"""
    classes = get_classes()
    
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(15, 7))
    for idx in np.arange(min(num_images, len(images))):
        ax = fig.add_subplot(3, 5, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Pred: {classes[preds[idx]]}\nTrue: {classes[labels[idx]]}", 
                     color=("green" if preds[idx]==labels[idx] else "red"),
                     fontsize=9)
    plt.tight_layout()
    plt.show()

def save_history(history, filename='models/training_history.json'):
    """Сохранение истории обучения в файл"""
    with open(filename, 'w') as f:
        json.dump(history, f)

def load_history(filename='models/training_history.json'):
    """Загрузка истории обучения из файла"""
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def show_random_predictions(model, dataset, num_images=20, device='cpu'):
    """Вывод случайных изображений с предсказаниями"""
    classes = get_classes()
    
    # Выбираем случайные индексы
    indices = random.sample(range(len(dataset)), num_images)
    
    # Создаем grid для отображения
    plt.figure(figsize=(15, 10))
    rows = int(num_images / 5) + (1 if num_images % 5 else 0)
    
    model.eval()
    for i, idx in enumerate(indices):
        image, true_label = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
        
        # Отображаем изображение
        plt.subplot(rows, 5, i+1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"True: {classes[true_label]}\nPred: {classes[predicted.item()]}", 
                 fontsize=8,
                 color='green' if predicted.item() == true_label else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()