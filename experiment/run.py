import argparse
import yaml
#from utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with configurable settings")
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer')
    return parser.parse_args()

def main(args):
   # setup_logging()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Override parameters with command-line arguments
    num_epochs = args.epochs if args.epochs is not None else config['num_epochs']
    learning_rate = args.lr if args.lr is not None else config['learning_rate']

    # Proceed with setup and training using overridden values
    print(f"Running training for {num_epochs} epochs with a learning rate of {learning_rate}")

if __name__ == "__main__":
    args = parse_args()
    main(args)