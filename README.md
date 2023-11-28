# Bachelorarbeit Project

This repository contains the codebase for my Bachelorarbeit (Bachelor's thesis). It represents a clean and updated version of the project, ensuring the absence of deprecated or redundant code.

## Overview

The project is dedicated to solving a timeseries prediction problem, namely the prediction of future sales for specific items at a bakery. It focuses on providing a working model that can make accurate predictions for the production planning of the bakery.

### Key Features

- **Feature 1:** Provide a working model
- **Feature 2:** Find a (close to) optimal model architecture for the problem (iTransformer??)
- **Feature 3:** (Maybe) Create a deployable solution that can actively be used in the bakery

## Project Structure

The repository is organized as follows:


- `data/`: Stores relevant data or datasets
- `data_provider/`: Transforms the raw data into usable datasets
- `model_ops/`: Methods for training, testing and validation (work in progress)
- `models/`: Stores the different model architectures
- `utils/`: Some utility functions

## Dataset

The dataset ranges from 01.01.2017 to 31.08.2023 in hourly intervals. Considering relevant times for sales in the bakery and closing periods, this results in roughly 30.000 timesteps. Each timestep cosists of the following features: 

- weather: (temperature, precipitation, wind_speed, wind_direction) (4)
- tourism: (time of arrivals, departues of each ferry)              (6)
- holidays: (holidays of all Bundesländer)                          (16)
- sales: (sales for a selected range / each item)                   (50 / 200+)

## Models

The foundational model driving this project is the iTransformer architecture, initially introduced in the paper accessible at this link: https://arxiv.org/pdf/2310.06625.pdf. This architecture, rooted in the encoder segment of a transformer, introduces a pivotal concept:

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

3. **Run the Project:**
    ```
    python main.py
    ```

4. **Have a look at the Data:**
    ```
    # full dataset:
    python -m data_provider.data_merger

    # partial datasets:
    python -m data_provider.sub_providers.fahrten_provider
    python -m data_provider.sub_providers.ferien_provider
    python -m data_provider.sub_providers.weather_provider
    python -m data_provider.sub_providers.sales_provider
    ```



## License

...

## Acknowledgements

...
