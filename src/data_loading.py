import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np
import random

class DualTransformDataset(Dataset):
    """Датасет, который возвращает как оригинальные, так и инвертированные изображения"""
    def __init__(self, dataset, selected_classes, transform=None, invert_prob=0.5):
        self.data = []
        self.targets = []
        self.transform = transform
        self.invert_prob = invert_prob
        self.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
        
        for img, label in dataset:
            if label in selected_classes:
                self.data.append(img)
                self.targets.append(self.class_to_idx[label])
    
    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            
        # С вероятностью invert_prob инвертируем изображение
        if random.random() < self.invert_prob:
            img = Image.fromarray(255 - np.array(img))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.data)

def get_data_loaders(batch_size=64, val_split=0.2):
    """Загрузка данных с аугментацией для 5 классов с инвертированными вариантами"""
    # Базовые преобразования
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Загружаем полные датасеты
    full_train_data = datasets.FashionMNIST(root="data", train=True, download=True)
    full_test_data = datasets.FashionMNIST(root="data", train=False, download=True)
    
    # Выбранные классы: Dress(3), Pullover(2), Trouser(1), Sandal(5), Bag(8)
    selected_classes = [3, 2, 1, 5, 8]
    
    # Создаем датасеты с поддержкой инверсии
    train_data = DualTransformDataset(
        full_train_data, 
        selected_classes, 
        transform=transform_train,
        invert_prob=0.5  # 50% chance to invert
    )
    
    test_data = DualTransformDataset(
        full_test_data,
        selected_classes,
        transform=transform_test,
        invert_prob=0.0  # Не инвертируем тестовые данные
    )
    
    # Разделяем train на train и validation
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    # Для валидации используем тестовые трансформации без инверсии
    val_data = DualTransformDataset(
        [(img, label) for img, label in val_data], 
        selected_classes, 
        transform=transform_test,
        invert_prob=0.0
    )
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def get_classes():
    """Возвращает список выбранных классов FashionMNIST"""
    full_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return [full_classes[3], full_classes[2], full_classes[1], full_classes[5], full_classes[8]]

def print_class_distribution():
    """Печатает распределение классов в train и test наборах"""
    full_train_data = datasets.FashionMNIST(root="data", train=True, download=True)
    full_test_data = datasets.FashionMNIST(root="data", train=False, download=True)
    
    selected_classes = [3, 2, 1, 5, 8]
    class_names = get_classes()
    
    train_counts = {name: 0 for name in class_names}
    test_counts = {name: 0 for name in class_names}
    
    for _, label in full_train_data:
        if label in selected_classes:
            cls_name = class_names[selected_classes.index(label)]
            train_counts[cls_name] += 1
    
    for _, label in full_test_data:
        if label in selected_classes:
            cls_name = class_names[selected_classes.index(label)]
            test_counts[cls_name] += 1
    
    print("\nClass distribution in training set:")
    for cls, count in train_counts.items():
        print(f"{cls}: {count} images")
    
    print("\nClass distribution in test set:")
    for cls, count in test_counts.items():
        print(f"{cls}: {count} images")
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    print(f"\nTotal train images: {total_train}")
    print(f"Total test images: {total_test}")