import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image

class RemappedDataset(Dataset):
    """Датасет с перенумерованными метками классов"""
    def __init__(self, dataset, selected_classes, transform=None):
        self.data = []
        self.targets = []
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
        
        for img, label in dataset:
            if label in selected_classes:
                self.data.append(img)
                self.targets.append(self.class_to_idx[label])
    
    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        
        # Конвертируем в PIL Image если это тензор
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.data)

def get_data_loaders(batch_size=64, val_split=0.2):
    """Загрузка данных с аугментацией для 4 классов: Dress, Pullover, Trouser, Sandal"""
    # Определяем преобразования
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
    
    # Выбранные классы: Dress(3), Pullover(2), Trouser(1), Sandal(5)
    selected_classes = [3, 2, 1, 5]
    
    # Создаем датасеты с перенумерованными метками
    train_data = RemappedDataset(full_train_data, selected_classes, transform_train)
    test_data = RemappedDataset(full_test_data, selected_classes, transform_test)
    
    # Разделяем train на train и validation
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    # Для валидации используем тестовые трансформации
    val_data = RemappedDataset(
        [(img, label) for img, label in val_data], 
        selected_classes, 
        transform_test
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
    # Возвращаем только нужные классы в порядке: Dress, Pullover, Trouser, Sandal
    return [full_classes[3], full_classes[2], full_classes[1], full_classes[5]]

def print_class_distribution():
    """Печатает распределение классов в train и test наборах"""
    # Загружаем полные датасеты без преобразований
    full_train_data = datasets.FashionMNIST(root="data", train=True, download=True)
    full_test_data = datasets.FashionMNIST(root="data", train=False, download=True)
    
    # Выбранные классы: Dress(3), Pullover(2), Trouser(1), Sandal(5)
    selected_classes = [3, 2, 1, 5]
    class_names = get_classes()
    
    # Считаем распределение
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
    
    # Печатаем результаты
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