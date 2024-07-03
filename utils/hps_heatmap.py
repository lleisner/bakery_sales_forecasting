import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hyperparameter_heatmap(tuner, save_dir='experiment/plots', num_trials=100):
    """
    Extracts the hyperparameter tuning results from the Keras Tuner object and creates a heat map 
    if exactly two hyperparameters were tuned. The heat map is saved to the specified directory.

    Parameters:
    - tuner: The Keras Tuner object after running the hyperparameter search.
    - save_dir: Directory to save the plots.
    - num_trials: Number of top trials to consider for creating the heat maps.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get the results from the tuner
    results = tuner.oracle.get_best_trials(num_trials=num_trials)
    
    # Create a DataFrame to store the results
    data = []
    for trial in results:
        trial_data = trial.hyperparameters.values
        trial_data['score'] = trial.score
        data.append(trial_data)
    
    df = pd.DataFrame(data)
    
    # Get the list of hyperparameters
    hyperparameters = list(df.columns)
    hyperparameters.remove('score')
    
    # Check the number of hyperparameters
    num_hyperparameters = len(hyperparameters)
    print(f'Number of hyperparameters tuned: {num_hyperparameters}')
    
    if num_hyperparameters == 2:
        param1, param2 = hyperparameters
        
        # Pivot the DataFrame to get a matrix format suitable for heat maps
        heatmap_data = df.pivot(param1, param2, 'score')
        
        # Plot the heat map
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='viridis')
        plt.title(f'Hyperparameter Tuning Heatmap ({param1} vs {param2})')
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.savefig(f'{save_dir}/heatmap_{param1}_vs_{param2}.png')
        plt.close()
        
        print(f'Heatmap created and saved in {save_dir}')
    else:
        print('No heat map created because too many parameters were used. Please tune on exactly two hyperparameters.')

