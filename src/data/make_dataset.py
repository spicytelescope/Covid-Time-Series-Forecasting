# -*- coding: utf-8 -*-
import click
from dotenv import find_dotenv, load_dotenv

from typing import List
from tqdm import tqdm
import pandas as pd
import sys


from src.utils.CustomLogger import logger


def convert_dataset_to_time_series(input_filepath: str) -> pd.DataFrame:
    """convert a csv file to a dataframe of time series (normal + 7-days rolling avg) organized by country

    Args:
        input_filepath (str): the input path of the csv file

    Returns:
        pd.DataFrame: dataframe of time series (normal + 7-days rolling avg) organized by country
    """

    dataset: pd.DataFrame = pd.read_csv(input_filepath)
    tmp_df: List[pd.DataFrame] = []

    with tqdm(
        total=len(dataset),
        desc="Converting dataset entry to time series format (normal + 7-days rolling avg)",
    ) as pbar:

        for i in range(len(dataset)):

            time_serie_name: str = (
                dataset.iloc[i]["Country/Region"]
                if pd.isna(dataset.iloc[i]["Province/State"])
                else dataset.iloc[i]["Province/State"]
            )
            pbar.set_description(
                f"Converting dataset entry to time series format (normal + 7-days rolling avg), processing {time_serie_name}"
            )

            tmp_df.append(
                convert_df_to_time_series(dataset.iloc[i].to_frame().T, time_serie_name)
            )

            pbar.update(1)

    process_dataset: pd.DataFrame = pd.concat(tmp_df, axis=1)
    return process_dataset


def convert_df_to_time_series(
    dataset: pd.DataFrame, time_series_name: str, differential=True
) -> pd.DataFrame:
    """Convert a dataframe of time records coming from one country (and one province attached to it) to a plotable time series
    Args:
        dataset (pd.DataFrame): input dataset
        time_series_name (str): name of the time series
        differential (bool, optional): differential data to get by day. Defaults to True.

    Returns:
        _type_: dataframe ready to be plotted
    """

    dataset_data_time_serie = dataset.drop(
        ["Province/State", "Country/Region", "Lat", "Long"], axis=1
    ).T
    dataset_data_time_serie.columns = [time_series_name]
    if differential:
        dataset_data_time_serie = dataset_data_time_serie.diff(axis=0)

        # rolling mean (7-days average)
        dataset_data_time_serie[
            f"{time_series_name}_7-days_average"
        ] = dataset_data_time_serie.rolling(window=7).mean()

    dataset_data_time_serie.index = pd.to_datetime(dataset_data_time_serie.index)
    return dataset_data_time_serie


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Args:
        input_filepath (str): location of the raw dataset
        output_filepath (str): location to the processed dataset
    """

    logger.info("starting data loading / processing")
    processed_dataset: pd.DataFrame = convert_dataset_to_time_series(input_filepath)

    logger.info(f"Saving final data set from raw data to {output_filepath}")
    processed_dataset.to_csv(output_filepath)


if __name__ == "__main__":

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
