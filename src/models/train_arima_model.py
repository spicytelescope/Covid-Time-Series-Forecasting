# -*- coding: utf-8 -*-
from typing import List

import click
import numpy as np
import pmdarima as pm
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import pickle

import pandas as pd
from src.utils.CustomLogger import logger

# 0 - CLI to select the country (or province) and then the input temporal window (e.g. train size) and output temporal window (e.g. test size)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("country", type=str)
@click.argument("unit", type=str)
@click.argument("input_window_min", type=int)
@click.argument("input_window_max", type=int)
@click.argument("output_window", type=int)
@click.argument("max_m_cross_val", type=int, default=1)
def main(
    input_filepath: str,
    country: str,
    unit: str,
    input_window_min: int,
    input_window_max: int,
    output_window: int,
    max_m_cross_val: int,
) -> None:

    """train an ARIMA model from a part ({input_window}) of a {country} in a dataset ({input_filepath}) following a {unit), using fine-tune ({max_m})

    Args:
        input_filepath (str): dataset to train the model on

        country (str): country that will be selected inside the dataset (located in {input_filepath}) in order to create a train test from it

        unit (str): can be either "deaths", "recovered" or "cases"

        input_window_min (int): beggining index (means beggining date in the timeline) of the train set

        input_window_max (int): ending index (means ending date in the timeline) of the train set

        output_window (int): number of day to predict, used in fine-tune for arima model to create the test set

        max_m_cross_val (int): max value for the m parameter (look on online documentation for more details about the utility of m in ARIMA models)
    """

    INPUT_WINDOW: List[int] = [input_window_min, input_window_max]
    OUTPUT_WINDOW: int = output_window

    # 1 - importing the data, pre-processing it and cleaning it
    logger.debug("loading dataset")
    dataset: pd.DataFrame = pd.read_csv(
        input_filepath, index_col=0, parse_dates=True
    ).fillna(0)
    logger.info("dataset successfuly loaded")

    logger.debug("preprocessing dataset by selecting country")
    try:
        train: pd.Series = dataset[country][INPUT_WINDOW[0] : INPUT_WINDOW[1]]
        test: pd.Series = dataset[country][
            INPUT_WINDOW[1] : INPUT_WINDOW[1] + OUTPUT_WINDOW
        ].rename("ground truth")
    except:
        logger.error("country not found or incorrect input window")
        exit(1)
    logger.info("train / test set successfuly created")

    # 2 - model tuning to find best parameters for the "auto_arima" model
    MAX_M: int = max_m_cross_val
    BEST_M: int = 1  # defaults to one if case the tune phase is skipped

    if MAX_M > 1:
        logger.debug(f"Starting model tuning by testing m from {1} to {MAX_M}")

        seasonnality_costs = []
        with tqdm(total=MAX_M) as pbar:
            for m in range(1, MAX_M + 1):
                pbar.set_description(f"training with {m=}")
                model = pm.auto_arima(train, m=m)

                # predicting
                raw_preds = model.predict(OUTPUT_WINDOW)
                preds = pd.DataFrame(
                    raw_preds, index=test.index[:OUTPUT_WINDOW], columns=["predictions"]
                )

                # biding to input
                seasonnality_costs.append(np.sqrt(mean_squared_error(test, preds)))
                pbar.update()

        BEST_M = np.argmax(seasonnality_costs) + 1  # type: ignore
        logger.info(f"Model tuning finished [{MAX_M=}] ! Best value for m : {BEST_M}")
    else:
        logger.info(
            f"Model tuning skipped due to MAX_M set to 1, processing to training phase"
        )

    # 3 - train with the best parameters
    logger.debug(f"Starting fitting phase for the ARIMA model with {BEST_M=}")
    model = pm.auto_arima(train, m=BEST_M)
    logger.info(f"ARIMA model ({BEST_M=}) successfuly trained !")
    # 4 - export the model
    logger.debug("Saving model ...")

    model_export_filepath: str = (
        "../../models/ARIMA"
        + f"_{unit}_{country}_{INPUT_WINDOW[0]}_{INPUT_WINDOW[1]}_m={BEST_M}"
    )
    with open(model_export_filepath + ".pkl", "wb") as export_model_file:
        pickle.dump(model, export_model_file)
    logger.info(f"final model saved to {model_export_filepath}")


if __name__ == "__main__":

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
