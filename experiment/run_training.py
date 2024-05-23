import subprocess
import time
import sys
import argparse
import os

from threading import Thread, Event
import time
import signal

def parse_wrapper_args():
    parser = argparse.ArgumentParser(description='Wrapper for running training script with retries.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default="experiment/dataset_analysis.yaml", help='Path to the YAML configuration file.')
    parser.add_argument('--data_directory', type=str, default="data/sales_forecasting", help='Path to data directory')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset to be used for experiment')
    parser.add_argument('--heartbeat_file', type=str, default="/tmp/heartbeat", help='Path to the heartbeat file')
    parser.add_argument('--heartbeat_interval', type=int, default=300, help='Interval in seconds for sending heartbeat signals')
    return parser.parse_args()

def send_heartbeat(heartbeat_file, interval, stop_event):
    while not stop_event.is_set():
        with open(heartbeat_file, 'w') as f:
            f.write('alive')
        stop_event.wait(interval)


def find_all_datasets(data_directory):
    csv_files = []
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    if not csv_files:
        print("No CSV files found in the directory.")
    return csv_files

def run_training(args, data_directory, dataset):
    stop_event = Event()
    heartbeat_thread = Thread(target=send_heartbeat, args=(args.heartbeat_file, args.heartbeat_interval, stop_event))
    heartbeat_thread.deamon = True
    heartbeat_thread.start()

    def signal_handler(sig, frame):
        stop_event.set()
        print("KeyboardInterrupt caught, stopping training...")
        with open(args.heartbeat_file, 'w') as f:
            f.write('stopped')

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Set the PYTHONPATH to ensure the script can find the modules
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'

        # Remove .csv extension for correct input 
        dataset_name = os.path.splitext(dataset)[0]

        # Build the command with the parsed arguments
        command = [
            "python", "-m", "experiment.setup",
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--num_epochs", str(args.num_epochs),
            "--config_file", args.config_file,
            "--data_directory", data_directory,
            "--dataset", dataset_name
        ]
        subprocess.run(command, check=True, env=env)
        
        # Send finished signal
        with open(args.heartbeat_file, 'w') as f:
            f.write('finished')
    except subprocess.CalledProcessError as e:
        print(f"Training crashed: {e}")
        return False
    except Exception as e:
        print(f"Training encountered an unrecoverable error: {e}")
        sys.exit(1)  # Exit with a specific error code
    finally:
        # Stop the heartbeat thread
        stop_event.set()
        heartbeat_thread.join()
    return True

def get_best_hyperparameters(args, data_directory, dataset):
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'

        # Remove .csv extension for correct input 
        dataset_name = os.path.splitext(dataset)[0]

        # Build the command with the parsed arguments
        command = [
            "python", "-m", "experiment.setup",
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--num_epochs", str(args.num_epochs),
            "--config_file", args.config_file,
            "--data_directory", data_directory,
            "--dataset", dataset_name
        ]
        subprocess.run(command, check=True, env=env)
    except:
        print(f"Could not find hyperparameters for {dataset_name}")
    



if __name__ == "__main__":
    args = parse_wrapper_args()

    datasets = []
    if args.dataset:
        datasets = [os.path.join(args.data_directory, args.dataset)]
    else:
        datasets = find_all_datasets(args.data_directory)

    for dataset_path in datasets:
        data_directory, dataset = os.path.split(dataset_path)
        get_best_hyperparameters(args, data_directory, dataset)


  #      print(f"\033[1;32mStarting training for dataset: {dataset} in directory: {data_directory}\033[0m")
   #     while True:
    #        if not run_training(args, data_directory, dataset):
     #           print("\033[1;31mRestarting training after crash...\033[0m")
      #          time.sleep(5)  # Wait a bit before restarting
       #     else:
        #        print(f"\033[1;32mTraining completed successfully for dataset: {dataset}\033[0m")
         #       break
    

