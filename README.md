# Bakery Sales Forecasting

This repository contains the code and data used for demand forecasting in a bakery, leveraging advanced machine learning models like iTransformer and TiDE. The project aims to improve the accuracy of sales predictions to optimize production processes and reduce food waste.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project applies machine learning models to forecast daily sales for a bakery located on a North Sea island. The focus is on using models like iTransformer and TiDE to predict sales based on various factors such as weather conditions, holidays, and tourist activity.

## Directory Structure

- **data/**: Contains the datasets used for training and testing the models. Includes raw data and processed data files.
  
- **data_fetcher/**: Scripts for downloading and updating the datasets from relevant sources for future deployment. This includes functionality to pull in new sales data, weather forecasts, holiday schedules, and tourist info.
  
- **data_provider/**: Responsible for data preprocessing and preparing the data for model training.
  
- **experiment/**: Contains the scripts for setting up and running the experiments. This includes model training, hyperparameter tuning, and evaluation of model performance.
  
- **models/**: Implements the machine learning models used in the project. This includes the iTransformer and TiDE models, along with their respective architectures.
  
- **utils/**: Utility functions used throughout the project. This includes helper functions for data processing and visualizations.

## Installation

To run this project, you need to have Python 3.10 installed. It is recommended to use a virtual environment to manage dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/lleisner/bakery_sales_forecasting.git
   cd bakery_sales_forecasting

## Usage

To run the project, use the `experiment.run` script. This script allows you to configure various parameters for the machine learning models through command-line arguments.

### Basic Usage

Run the project with default settings:

    #```bash
    python -m experiment.run --dataset <dataset_name> --models iTransformer TiDE

### Command-Line Arguments
You can customize the run by specifying the following arguments:

--batch_size: Batch size for training the model. (default: 32)
--learning_rate: Learning rate for the optimizer. (default: 0.00005)
--num_epochs: Number of epochs for training. (default: 50)
--config_file: Path to the YAML configuration file. (default: experiment/dataset_analysis.yaml)
--data_directory: Path to the data directory. (default: data/sales_forecasting/sales_forecasting_8h)
--dataset: Dataset to be used for the experiment. (required)
--models: List of models to be used. (default: ['Baseline'])
--mode: Operation mode: tune, train, or load. (default: train)
--normalize: Whether to normalize the data. (default: False)
--loss: Loss function to use in training: mse or amse. (default: mse)

### Example Commands
Train models with custom batch size and learning rate:

python -m experiment.run --batch_size 64 --learning_rate 0.0001 --dataset <dataset_name> --models iTransformer TiDE

Tune the models using a specific configuration file:

python -m experiment.run --mode tune --config_file path/to/your_config.yaml --dataset <dataset_name>

Load a pre-trained model and evaluate:

python -m experiment.run --mode load --dataset <dataset_name> --models iTransformer
