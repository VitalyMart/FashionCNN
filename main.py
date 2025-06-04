import argparse
import torch
from data_loading import get_data_loaders
from model import FashionCNN
from training import train_model
from evaluation import plot_training_history, save_history, load_history
from inference import test_model

def main():
    parser = argparse.ArgumentParser(description='FashionMNIST CNN Model Trainer')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='fashion_mnist_model.pth', 
                        help='Path to save/load the model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.train:
        print("Training the model...")
        train_loader, val_loader, _ = get_data_loaders()
        model = FashionCNN().to(device)
        history = train_model(model, train_loader, val_loader, epochs=args.epochs, device=device)
        torch.save(model.state_dict(), args.model_path)
        save_history(history)
        print(f"Model saved to {args.model_path}")

    if args.test:
        print("Testing the model...")
        test_model(args.model_path, device)

    if args.plot:
        print("Plotting training history...")
        history = load_history()
        if history:
            plot_training_history(history)
        else:
            print("No training history found. Train the model first with --train")

    if not any([args.train, args.test, args.plot]):
        parser.print_help()

if __name__ == "__main__":
    main()