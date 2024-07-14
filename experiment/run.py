import argparse
import numpy as np
import pandas as pd

from experiment.setup import init_comps, tune_model_on_dataset, train_model_on_dataset, load_existing_model
from utils.plot_time_series import plot_multivariate_time_series_predictions

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configure model parameters.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--config_file', type=str, default="experiment/dataset_analysis.yaml", help='Path to the YAML configuration file.')
    parser.add_argument('--data_directory', type=str, default="data/sales_forecasting/sales_forecasting_8h", help='Path to data directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for experiment')
    parser.add_argument('--models', nargs='+', type=str, default=['Baseline'], help='List of Models to be used')
    parser.add_argument('--mode', choices=['tune', 'train', 'load'], default='train', help='Operation mode: tune, train, or load a saved model.')

    parser.add_argument('--normalize', type=bool, default=False, help='Normalize the data')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use in training')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    time_series = {}
    
    for model_name in args.models:
        model, data_loader = init_comps(model_name, args)
        
        mode_function_map = {
            'tune': tune_model_on_dataset,
            'train': train_model_on_dataset,
            'load': load_existing_model
        }

        # Get the function to execute based on the mode
        execute = mode_function_map[args.mode]
        execute(args, model_name, model, data_loader)
        
        data = data_loader.get_data()
        
        
        to_predict, index = data_loader.get_prediction_set()
        preds, actuals = model.predict(to_predict)
        
        preds, actuals = [np.asarray(arr.reshape(-1, arr.shape[-1]).tolist()) for arr in (preds, actuals)]


        variate_names = data.columns.tolist()[:preds.shape[-1]]
        
        preds, actuals = [pd.DataFrame(data, index=index, columns=variate_names) for data in [preds, actuals]]
        
        time_series['Actual'] = actuals
        time_series[model_name] = preds
        
    print(time_series)
    plot_multivariate_time_series_predictions(time_series)
    
    
if __name__ == "__main__":
    main()