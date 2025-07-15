import argparse

import torch

from src.data_loading import get_data_loaders, print_class_distribution
from src.model import FashionCNN
from src.training import train_model
from src.evaluation import plot_training_history, save_history, load_history
from src.inference import test_model

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='FashionMNIST CNN Model Trainer')
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--image', type=str, help='Path to image file for prediction')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--check_imgs', action='store_true', 
                       help='Show number of images in each class')
    parser.add_argument('--model_path', type=str, default='models/fashion_mnist_model.pth', 
                       help='Path to save/load the model')
    
    return parser.parse_args()

def setup_device():
    """Определение устройства для вычислений (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def handle_check_images():
    """Обработка запроса проверки распределения классов"""
    print_class_distribution()

def handle_training(args, device):
    """Обработка процесса обучения модели"""
    print("Training the model...")
    train_loader, val_loader, _ = get_data_loaders()
    model = FashionCNN().to(device)
    
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        device=device
    )
    
    torch.save(model.state_dict(), args.model_path)
    save_history(history)
    print(f"Model saved to {args.model_path}")

def handle_testing(args, device):
    """Обработка процесса тестирования модели"""
    print("Testing the model...")
    test_model(args.model_path, device, args.image)

def handle_plotting():
    """Обработка запроса визуализации истории обучения"""
    print("Plotting training history...")
    history = load_history()
    
    if history:
        plot_training_history(history)
    else:
        print("No training history found. Train the model first with --train")

def main():
    args = parse_arguments()
    device = setup_device()

    if args.check_imgs:
        handle_check_images()
        return

    if args.train:
        handle_training(args, device)

    if args.test:
        handle_testing(args, device)

    if args.plot:
        handle_plotting()

    if not any([args.train, args.test, args.plot, args.check_imgs]):
        print("No action specified. Use --help to see available options.")

if __name__ == "__main__":
    main()