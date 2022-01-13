import argparse


def main(args: argparse.Namespace):
    """
    Main function for training a classifier.
    :param args: Arguments from the command line.
    """



if __name__ == "__main__":
    # Start argparse
    parser = argparse.ArgumentParser(description="Train a classifier")

    # Dataset
    parser.add_argument("--dataset", type=str, default="FFHQ-Aging", help="Dataset to train on")

    # Labels
    parser.add_argument("--labels", type=str, default="gender", help="Labels to train on")

    # Parse and pass to main
    args = parser.parse_args()
    main(args)
