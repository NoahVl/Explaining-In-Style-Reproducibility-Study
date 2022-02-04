import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Dataset utils
from data.Kaggle_FFHQ_Resized_256px import ffhq_utils

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, checkpoint_name):
    """
    Saves the model parameters to a checkpoint file.

    Args:
        model: nn.Module object representing the model architecture.
        checkpoint_name: Name of the checkpoint file.
    """
    # Check if the saved_model directory exists, if not create it
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    torch.save(model.state_dict(), os.path.join("saved_models", checkpoint_name))


def load_model(model, checkpoint_name):
    """
    Loads the model parameters from a checkpoint file.

    Args:
        model: nn.Module object representing the model architecture.
        checkpoint_name: Name of the checkpoint file.
    Returns:
        model: nn.Module object representing the model architecture.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(os.path.join("saved_models", checkpoint_name), map_location=device))
    return model


def train_model(model, lr, batch_size, epochs, checkpoint_name, device, train_dataset, val_dataset):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
    Returns:
        model: Model that has performed best on the validation set.

    """
    assert epochs > 0, "To train the model the amount of epochs has to be higher than 1."

    # Make dataloaders from the datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              generator=torch.Generator().manual_seed(42), pin_memory=True, num_workers=6)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size,
                              generator=torch.Generator().manual_seed(42), pin_memory=True, num_workers=6)

    # Initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define loss
    loss = nn.CrossEntropyLoss()

    # Placeholder for saving the best model.
    best_valid_accuracy = 0

    # Use tensorboard to visualize the training process.
    writer = SummaryWriter(log_dir='./tboard_logs')

    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    for epoch in range(epochs):
        start_epoch_time = time.time()
        train_losses = []

        # Training loop
        for batch_num, batch in enumerate(train_loader):
            # Send data to device
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            # Send images through the model
            predictions = model(images)

            # Calculate loss
            batch_loss = loss(predictions, targets)
            train_losses.append(batch_loss.item())

            # Log the loss
            writer.add_scalar('Loss/train', batch_loss.item(), epoch * len(train_loader) + batch_num)

            if batch_num % 1 == 0 or batch_num == len(train_loader) - 1:
                print('\r',
                      f"Epoch: {epoch}: batch {batch_num + 1}/{len(train_loader)}"
                      f", running loss: {np.average(train_losses)}", end='')

            # Reset gradients
            optimizer.zero_grad()

            # Perform backward pass and optimization
            batch_loss.backward()
            optimizer.step()

        # Validation and train accuracy
        model.eval()
        with torch.no_grad():
            train_epoch_accuracy = evaluate_model(model, train_loader, device)
            valid_epoch_accuracy = evaluate_model(model, valid_loader, device)
            print(f", train accuracy: {train_epoch_accuracy}, validation accuracy: {valid_epoch_accuracy}, epoch took: "
                  f"{(time.time() - start_epoch_time) / 60:.2f} minutes")

            # Save model if it is the best model on the validation set.
            if valid_epoch_accuracy > best_valid_accuracy:
                save_model(model, checkpoint_name)
                best_valid_accuracy = valid_epoch_accuracy

            # Log the accuracy
            writer.add_scalar('Accuracy/train', train_epoch_accuracy, epoch)
            writer.add_scalar('Accuracy/validation', valid_epoch_accuracy, epoch)
        model.train()

    # Load best model and return it.
    model = load_model(model, checkpoint_name).to(device)

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.
    """
    correct_predictions = 0
    number_examples = 0
    for images, targets in data_loader:
        # Send data to device
        images, targets = images.to(device), targets.to(device)
        predictions = model(images)

        # Calculate number of correct predictions
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions += sum(predicted_labels == targets)
        number_examples += len(targets)

    accuracy = correct_predictions / number_examples

    return accuracy


def test_model(model, batch_size, device, seed, test_dataset):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
        test_dataset: The test dataset to use.
    Returns:
        test_results: Dictionary containing an overview of the accuracy.
    """
    set_seed(seed)

    # Set model to evaluation mode
    model.eval()

    test_results = {}

    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 generator=torch.Generator().manual_seed(42), pin_memory=True,
                                 num_workers=6)
        accuracy = evaluate_model(model, test_loader, device)
        test_results['accuracy'] = accuracy.item()

        # Test accuracy
        print(f"Test accuracy: {accuracy.item():.4f}")

    # Set model back to train mode
    model.train()

    return test_results


def load_mobilenet(device, amount_frozen_layers=15, freeze_all_layer=False):
    """
    Returns a MobileNet model.
    :param device: Device to put the model on.
    :param amount_frozen_layers: Amount of layers to freeze.
    :param freeze_all_layer: If true, all layers are frozen.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)

    # Freeze all layers
    if freeze_all_layer:
        for param in model.parameters():
            param.requires_grad = False

    # Make the last layer have only 2 outputs instead of 1000.
    model.classifier[1] = nn.Linear(1280, 2).to(device)

    # If you want to only freeze a few layers, you can do this:
    for layer in range(amount_frozen_layers):
        for param in model.features[layer].parameters():
            param.requires_grad = False

    return model


def main(args: argparse.Namespace):
    """
    Main function for training a classifier.
    :param args: Arguments from the command line.
    """
    # Define device and seed
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)

    if args.dataset == "FFHQ-Aging":
        train_dataset, valid_dataset, test_dataset = ffhq_utils.get_train_valid_test_dataset(
            "data/Kaggle_FFHQ_Resized_256px", "gender")
    else:
        raise NotImplementedError

    model = load_mobilenet(device, args.amount_frozen_layers, args.freeze_all_layers)

    # Check if model was already trained, if it was import it, if not train it
    if not os.path.exists(os.path.join("saved_models", args.checkpoint_name)):
        train_model(model, args.lr, args.batch_size, args.epochs, args.checkpoint_name, device, train_dataset,
                    valid_dataset)
    else:
        model = load_model(model, args.checkpoint_name).to(device)
        if args.continue_training:
            train_model(model, args.lr, args.batch_size, args.epochs, args.checkpoint_name, device, train_dataset,
                        valid_dataset)

    # Then test the model with all the defined corruption features
    # Return the results
    test_results = test_model(model, args.batch_size, device, args.seed, test_dataset)

    return test_results


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Start argparse
    parser = argparse.ArgumentParser(description="Train a classifier")

    # Model
    parser.add_argument("--freeze_all_layers", dest="freeze_all_layers", action="store_true")
    parser.add_argument("--amount_frozen_layers", dest="amount_frozen_layers", type=int, default=15)

    # Dataset
    parser.add_argument("--dataset", type=str, default="FFHQ-Aging", help="Dataset to train on")

    # Labels
    parser.add_argument("--labels", type=str, default="gender", help="Labels to train on")

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=200, type=int,  # Was 38 when training all the weights
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=50, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--checkpoint_name', default="FFHQ-Gender.pth", type=str, help="Name of the model checkpoint")
    parser.add_argument('--continue_training', dest="continue_training", action="store_true")

    # Parse and pass to main
    parse_args = parser.parse_args()
    results = main(parse_args)

    # Write results to a json file with the same name as the checkpoint
    # First check if a folder called classifier_results exists, if not create it and save the csv file there
    if not os.path.exists("classifier_results"):
        os.mkdir("classifier_results")
    with open(os.path.join("classifier_results", parse_args.checkpoint_name.split(".")[0] + ".json"), "w") as f:
        json.dump(results, f)
