"""
Scripts to show the predictions of a model
"""
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error
from typing import Callable


def plot_predictions(
    input: pd.DataFrame,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    unit_name: str,
    loss_fnc: Callable = mean_squared_error,
) -> None:
    """plot the predictions of the model for time series forecasting

    Args:
        input (List[float]): input time series
        ground_truth (List[float]): real time series continuity
        predictions (List[float]): forecasting of the time series
        unit_name (str): unit_name of prediction, one of the following : ['deaths', 'cases', 'recovered']
        loss_fnc (Callable): the loss function used to compute the cost for the prediction
    """
    # Correctly renaming each df
    input = input.rename(
        columns={input.columns[0]: "train"}
    )
    ground_truth = ground_truth.rename(
        columns={ground_truth.columns[0]: "ground truth"}
    )
    predictions = predictions.rename(
        columns={predictions.columns[0]: "predictions"}
    )

    # Biding forecasting to input for vizualiation
    ground_truth = pd.concat(
        [
            pd.DataFrame(input.tail(1)).rename(
                columns={input.columns[0]: "ground truth"}
            ),
            ground_truth,
        ]
    )
    predictions = pd.concat(
        [
            pd.DataFrame(input.tail(1)).rename(
                columns={input.columns[0]: "predictions"}
            ),
            predictions,
        ]
    )
    st.header(
        f"Prediction of {unit_name} of {len(predictions)} days following time sequence : {input.index[0].strftime('%d/%m/%Y')} - {input.index[-1].strftime('%d/%m/%Y')}"  # type: ignore
    )
    col1, col2 = st.columns((5, 1))

    col1.line_chart(data=pd.concat([input, predictions, ground_truth], axis=1))

    # loss handling
    loss: float = loss_fnc(ground_truth, predictions)  # type: ignore
    col2.metric("Loss", loss)
