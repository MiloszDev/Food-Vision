"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import multiprocessing
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Set up argument parsing
parser = argparse.ArgumentParser(description='Trains a PyTorch image classification model')

# Add arguments with type specifications
parser.add_argument('--train_dir', type=str, required=True, help="Directory path to training data")
parser.add_argument('--test_dir', type=str, required=True, help="Directory path to test data")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train the model")
parser.add_argument('--hidden_units', type=int, default=128, help="Number of hidden units in the model")

args = parser.parse_args()

# Use the parsed arguments
train_dir = args.train_dir
test_dir = args.test_dir
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_EPOCHS = args.num_epochs
HIDDEN_UNITS = args.hidden_units

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )
    print(f'Dataloaders were setup sucessfully!')
    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Start training with help from engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="tinyvgg_model.pth")