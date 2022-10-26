import pickle
import re
from datetime import datetime, timedelta
from typing import List

import h5py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from src.models.train_lstm_model import create_dataset
from src.visualization.predictions_viz import plot_predictions
from tensorflow import keras

# idea : save some models that trained on "rolling mean" | "normal" and on range "till 2021 mid semester" | "till 2022" -> all for "deaths" | "recovered" | "confirmed

def output_window_select(MAX_MODEL_OUTPUT_PRED: int) -> int:
    """component to select the output window for a given model

    Args:
        MAX_MODEL_OUTPUT_PRED (int): constraint for a loaded model

    Returns:
        int: output window size
    """
    st.header("Select the output temporal window for the prediction")
    return st.slider("Output window size", min_value=1, max_value=MAX_MODEL_OUTPUT_PRED)


def prediction_web_app_component() -> None:

    MAX_MODEL_OUTPUT_PRED: int = 50
    st.title("Model prediction")
    st.header("I) Loading base data")

    col1, col2 = st.columns(2)

    col1.subheader("1 - Select a dataset")
    dataset_file = col1.file_uploader(label="model", accept_multiple_files=False, key=1)
    unit = col1.selectbox("unit", ["deaths", "recovered", "cases"])
    train: pd.DataFrame = pd.DataFrame()
    test: pd.DataFrame = pd.DataFrame()

    if dataset_file is not None:
        try:
            dataset = pd.read_csv(dataset_file, index_col=0, parse_dates=True).fillna(0)  # type: ignore
        except:
            st.error("The file isn't a dataset")
        else:
            col2.subheader("2 - Select a country")
            country_list: pd.DataFrame = pd.unique(dataset.columns)
            country: str = col2.selectbox("Country", country_list)

            try:
                if country != "":
                    train: pd.DataFrame = dataset[country].to_frame()
            except:
                st.error("Incorrect selected country")
            else:

                st.header("II) Loading Model")
                model_pickle_file = st.file_uploader(
                    label="", accept_multiple_files=False, key=2
                )
                model = None
                LSTM_MODEL = False
                proceed = False
                try:
                    model = pickle.load(model_pickle_file)  # type: ignore
                    model.predict(1)  # test if the instance loaded is a model
                except:
                    # not an ARIMA model, testing if it's a tensorflow one
                    try:
                        # model = keras.models.load_model(model_pickle_file)
                        with h5py.File(model_pickle_file, 'r') as h5_file:
                            model = keras.models.load_model(h5_file)

                    except:
                        st.error("Incorrect file type, upload a model")
                    else:
                        st.success("LSTM model successfully loaded")
                        LSTM_MODEL = True
                        proceed = True
                else:
                    st.success("ARIMA model successfully loaded")
                    proceed = True
                
                if proceed:
                    title = model_pickle_file.name
                    # Fetching the train time range by parsing the title
                    input_window_raw = re.search(
                        f"{country}_(\d+)_(\d+)", title
                    ).groups()

                    INPUT_WINDOW: List[int] = [int(x) for x in input_window_raw]
                    train: pd.DataFrame = train.iloc[
                        INPUT_WINDOW[0] : INPUT_WINDOW[1], :
                    ]

                    st.subheader("Selected train time range")
                    # Creating read-only time range slider
                    sub_col1, sub_col2 = st.columns((5, 1))
                    BEGINNING_TIME: datetime = datetime(2020, 1, 22)
                    sub_col1.slider(
                        "input time range",
                        min_value=BEGINNING_TIME,
                        max_value=datetime(2022, 3, 28)
                        - timedelta(days=MAX_MODEL_OUTPUT_PRED),
                        format="DD/MM/YY",
                        value=(
                            train.iloc[INPUT_WINDOW[0]].name.to_pydatetime(),
                            train.iloc[INPUT_WINDOW[1] - 1].name.to_pydatetime(),
                        ),
                        disabled=True,
                    )
                    sub_col2.write(INPUT_WINDOW)

                    st.header("III) Making predictions")
                    OUTPUT_WINDOW: int = output_window_select(MAX_MODEL_OUTPUT_PRED)
                    test: pd.DataFrame = dataset[country][
                        INPUT_WINDOW[1] : INPUT_WINDOW[1] + OUTPUT_WINDOW
                    ].to_frame()

                    if st.button("Predict"):
                        # making predictions
                        if not LSTM_MODEL:
                            raw_preds = model.predict(OUTPUT_WINDOW)
                        else:
                            test_X, test_Y = create_dataset(np.array(test), 1)
                            raw_preds = model.predict(test_X)

                        if raw_preds is not None or raw_preds != []:
                            preds = pd.DataFrame(
                                raw_preds,
                                index=test.index[:OUTPUT_WINDOW-2],
                                columns=["predictions"],
                            )

                            # Take into account the LSTM nn that outputs scaled predictions, so we need to scale the train & test set
                            if LSTM_MODEL:
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                train_scaled = pd.DataFrame(scaler.fit_transform(train), index=train.index)

                                test_scaled = pd.DataFrame(scaler.fit_transform(test_Y.reshape(-1, 1)), index=test.index[:OUTPUT_WINDOW-2])

                                plot_predictions(train_scaled, test_scaled, preds, unit)
                            else:
                                plot_predictions(train, test, preds, unit)
                        else:
                            st.error("Impossible to make predictions")


if __name__ == "__main__":
    prediction_web_app_component()
