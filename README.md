# Application of Machine Learning Models for Sales Forecasting in a Local Bakery

This repository contains the codebase the bachelor thesis.
## Overview

The project is dedicated to solving a timeseries prediction problem, namely the prediction of future sales for specific items at a bakery. It focuses on providing a working model that can make accurate predictions for the production planning of the bakery.

### Key Features

- **Feature 1:** Provide a working model
- **Feature 2:** Find a (close to) optimal model architecture for the problem (iTransformer??)
- **Feature 3:** (Maybe) Create a deployable solution that can actively be used in the bakery

## Project Structure

The repository is organized as follows:

- `data/`: Stores relevant datasets
- `data_fetcher/`: Updates existing datasets with real time information
- `data_provider/`: Transforms the raw data into usable datasets
- `models/`: Stores the different model architectures
- `experiment/`: Methods to train, tune or load existing models
- `utils/`: Contains utility functions used for plotting or other helper functions

## Datasets

The dataset ranges from 01.01.2019 to 31.12.2023 in hourly intervals. 
Each timestep cosists of the following features: 

- weather: (temperature, precipitation, cloud_cover, wind_speed, wind_direction)    (4)
- tourism: (time of arrival/departue of a ferry, tourist count)                     (3)
- holidays: (holidays of all Bundesländer)                                          (1)
- sales: (sales for a selected range / each item)                                   (2-64)
- operational_feature: (is_open)                                                    (1)


## Models

The model driving this project is the iTransformer architecture, initially introduced in the paper accessible at this link: https://arxiv.org/pdf/2310.06625.pdf. This architecture, rooted in the encoder segment of a transformer, introduces a pivotal concept:

Rather than adhering to the conventional method of modeling global dependencies across temporal tokens in time series data (where a token represents multiple variables at a specific timestamp), the iTransformer architecture restructures this approach. It embeds the time series data into variate tokens, where a token signifies multiple timestamps of a single variable.

This innovative restructuring, as per the paper's insights, yields performance enhancements by:

- Maximizing the utilization of increased lookback windows more effectively, surpassing the limitations of the base transformer's attention mechanism, especially when handling growing inputs.
- Generating more comprehensive attention maps that encapsulate multivariate correlations, thereby enhancing generalization capabilities across diverse variables.
- Optimizing the use of the feed-forward network to acquire nonlinear representations for each variate token.

The reported results showcase a notable performance boost of over 30% compared to the base transformer architecture. Furthermore, the iTransformer architecture outperforms other competitive (transformer-based) architectures by a significant margin.

## Getting Started

To run the project locally or replicate the environment, follow these steps:

1. **Clone the repository:**
    ```
    git clone <repository_url>
    cd clean_bachelor
    ```

2. **Setup Environment:**
    ```
    conda create -f conda_config.yml
    conda activate lleisner
    ```

3. **Run the Project:**‚
    ```
    python -m experiment.run 
    ```



## License

...

## Acknowledgements

...
