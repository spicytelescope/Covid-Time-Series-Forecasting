# 05_Covid_Project_IA

> Author : Jordan Beziaud

Given data in a country, predict for one day the number of dead / covid cases

## Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Installation

This project uses python3.8, but any version > 3.6 will work.
To install all the dependencies, run the following command at the root of the project :

    python3 setup.py

In case it doesn't work, run `python3 -m pip install -r requirements.txt` instead.

## Usage & workflow

There is **2 main tools** to use in order to fully explore this project :
- 2 web applications created with `streamlit` that are meant for demonstrations purposes, such as using models to make predictions, or explore datasets
    To run them, **open 2 different shell instances and run** :
    ```bash
    # shell 1
    cd src/models
    streamlit run predict_model_app.py

    # shell 2
    cd visualization
    streamlit run exploration_app.py
    ```
    and discover them ! They are kept simple, with a limit set of features.
    **Warning : the prediction web app only work with ARIMA model for now. Look at the notebook number 4 for LSTM predictions**
- 2 scripts in `/src/models` to train and automatically save models in the `{ROOT_FOLDER]/models` directory (**warning : there is another dir called 'models' located in the `/src`, this one contains the train scripts not the saved models**)
    To use these scripts and create either ARIMA or LSTM NN trained models, run the following commands :
    ```bash
    cd src/models
    python3 train_arima_model.py INPUT_FILEPATH COUNTRY UNIT INPUT_WINDOW_MIN INPUT_WINDOW_MAX OUTPUT_WINDOW [MAX_M_CROSS_VAL]

    # or, if you wanna train LSTM Neural Network
    python3 train_lstm_model.py INPUT_FILEPATH COUNTRY UNIT INPUT_WINDOW_MIN INPUT_WINDOW_MAX [MAX_EPOCH_CV] [MAX_BATCH_SIZE_CV]
    ```

    If the command-line interface seems too complicated, just specify the `--help` option to get a complete explanation of each arguments. (example : `python3 ./train_arima_model.py --help`)
    **Warning : Training arima model with a big value for `max_m_cross_val` (see options) implies that a huge of `m` will be tested to train ARIMA model which will take an enormous amount of time. If training ARIMA model, proceed with caution and use value prior to 50 to get reasonable training time**

In addition to that, a script to create processed and ready-to-use dataset from raw data is available in `src/data`. Example usage:

```bash
cd src/data
python3 make_dataset.py ../../data/raw/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv ../../data/processed/processed_global_death_country.csv
```
