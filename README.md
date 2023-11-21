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
- `models/`: Stores the different model architectures
- `utils/`: Some utility functions

## Dataset

The dataset ranges from 01.01.2017 to 31.08.2023 in hourly intervals. Considering relevant times for sales in the bakery and closing periods, this results in roughly 20.000 timesteps. Each timestep cosists of the following features: 

- weather: (temperature, precipitation, wind_speed, wind_direction) (4)
- tourism: (time of arrivals, departues of each ferry)              (6)
- holidays: (holidays of all Bundesländer)                          (16)
- sales: (sales for a selected range / each item)                   (50 / 200+)


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



## License

...

## Acknowledgements

...
