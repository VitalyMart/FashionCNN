import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=64, val_split=0.2):
    """Загрузка данных с аугментацией"""
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

    full_train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train)
    full_test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)
    
    selected_classes = [0, 1, 2, 3, 4]
    
    train_idx = [i for i, (_, label) in enumerate(full_train_data) if label in selected_classes]
    train_data = torch.utils.data.Subset(full_train_data, train_idx)
    
    test_idx = [i for i, (_, label) in enumerate(full_test_data) if label in selected_classes]
    test_data = torch.utils.data.Subset(full_test_data, test_idx)
    
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    # Для валидации используем тестовые трансформации (без аугментации)
    val_data.dataset.transform = transform_test
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
    
def get_classes():
    """Возвращает список выбранных классов FashionMNIST"""
    full_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return full_classes[:5]  # Возвращаем только первые 5 классов

def get_class_distribution(dataset):
    """Возвращает распределение классов в датасете"""
    class_counts = {cls: 0 for cls in get_classes()}
    for _, label in dataset:
        class_counts[get_classes()[label]] += 1
    return class_counts

def print_class_distribution():
    """Печатает распределение классов в train и test наборах"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Загружаем полные датасеты
    full_train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    full_test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    
    # Фильтруем только первые 5 классов
    selected_classes = [0, 1, 2, 3, 4]
    train_idx = [i for i, (_, label) in enumerate(full_train_data) if label in selected_classes]
    test_idx = [i for i, (_, label) in enumerate(full_test_data) if label in selected_classes]
    
    train_data = torch.utils.data.Subset(full_train_data, train_idx)
    test_data = torch.utils.data.Subset(full_test_data, test_idx)
    
    # Получаем распределение классов
    train_dist = get_class_distribution(train_data)
    test_dist = get_class_distribution(test_data)
    
    # Печатаем результаты
    print("\nClass distribution in training set:")
    for cls, count in train_dist.items():
        print(f"{cls}: {count} images")
    
    print("\nClass distribution in test set:")
    for cls, count in test_dist.items():
        print(f"{cls}: {count} images")
    
    total_train = sum(train_dist.values())
    total_test = sum(test_dist.values())
    print(f"\nTotal train images: {total_train}")
    print(f"Total test images: {total_test}")