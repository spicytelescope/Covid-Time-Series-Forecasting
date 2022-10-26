# -*- coding: utf-8 -*-

# script used to train LSTM NN for time series forecasting

from typing import List, Tuple

import click
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from src.utils.CustomLogger import logger


def create_lstm_model() -> Sequential:
    """return a Sequentiel keras model based on LSTM architecture

    Returns:
        Sequential: keras sequential LSTM based model
    """
    look_back = 1

    model: Sequential = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def create_dataset(dataset, look_back=1) -> Tuple[np.ndarray, np.ndarray]:
    """convert the normal dataset in sub time series regarding the look_back parameters (look_back = 1 -> X=t and Y=t+1) in order to form the features and label subsets

    Args:
        dataset (_type_): input dataset
        look_back (int, optional): shifting parameter. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: return the features and label from dataset
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# 0 - CLI to select the country (or province) and then the input temporal window (e.g. train size) and output temporal window (e.g. test size)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("country", type=str)
@click.argument("unit", type=str)
@click.argument("input_window_min", type=int)
@click.argument("input_window_max", type=int)
@click.argument("max_epoch_cv", type=int, default=201)
@click.argument("max_batch_size_cv", type=int, default=31)
def main(
    input_filepath: str,
    country: str,
    unit: str,
    input_window_min: int,
    input_window_max: int,
    max_epoch_cv: int,
    max_batch_size_cv: int,
) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Args:
        input_filepath (str): dataset to train the model on
        country (str): country that will be selected inside the dataset (located in {input_filepath}) in order to create a train test from it
        unit (str): can be either "deaths", "recovered" or "cases"
        input_window_min (int): beggining index (means beggining date in the timeline) of the train set
        input_window_max (int): ending index (means ending date in the timeline) of the train set
        max_epoch_cv (int): max value for the epoch parameter that will be put in the param_grid for the Grid Search
        max_batch_size_cv (int): max value for the batch_size parameter that will be put in the param_grid for the Grid Search
    """

    INPUT_WINDOW: List[int] = [input_window_min, input_window_max]

    # 1 - importing the data, pre-processing it and cleaning it
    logger.debug("loading dataset")
    dataset: pd.DataFrame = pd.read_csv(
        input_filepath, index_col=0, parse_dates=True
    ).fillna(0)
    logger.info("dataset successfuly loaded")

    logger.debug("preprocessing dataset by selecting country")
    try:
        train: np.ndarray = np.array(
            dataset[country][INPUT_WINDOW[0] : INPUT_WINDOW[1]]
        )
    except:
        logger.error("country not found or incorrect input window")
        exit(1)
    logger.info("train set successfuly created")

    logger.debug("scaling train set")
    train = train.reshape(len(train), 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    logger.info("successfully scaled between 0 and 1 the train set")

    # 2 - features & label creation from time series (look_back use)
    # example : X=t and Y=t+1
    logger.debug("creating features and labels subsets")
    look_back = 1
    train_X, train_Y = create_dataset(train, look_back)

    # reshape input to be [samples, time steps, features] as requested by LSTM NN
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    logger.info("successfully created labels/features")

    # 3 - Fine-tuning the model
    logger.debug(
        f"Start fine-tune model phase with {max_epoch_cv=} and {max_batch_size_cv=}"
    )
    model = KerasRegressor(build_fn=create_lstm_model)
    param_grid = {
        "epochs": range(20, max_epoch_cv, 10),
        "batch_size": range(10, max_batch_size_cv, 5),
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=0)
    grid_result = grid.fit(train_X, train_Y)

    logger.info(
        f"GridSearchCV successfully ended with best score : {round(grid_result.best_score_, 2)} using {grid_result.best_params_}"
    )

    # 4 - fitting the best model
    best_model = create_lstm_model()
    best_model.fit(train_X, train_Y, **grid_result.best_params_)

    # 5 - export the model
    logger.debug("Saving model ...")

    model_export_filepath: str = (
        "../../models/LSTM" + f"_{unit}_{country}_{INPUT_WINDOW[0]}_{INPUT_WINDOW[1]}.h5"
    )
    best_model.save(model_export_filepath)
    logger.info(f"final model saved to {model_export_filepath}")


if __name__ == "__main__":
    main()
