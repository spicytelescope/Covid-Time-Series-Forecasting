"""
Data Viz Web App to explore each country's 3 curbs : deaths, cases and recovered
"""

from typing import List
import streamlit as st
import pandas as pd
from dtaidistance import dtw
from src.utils.helper import get_project_root
import os

root_path = get_project_root()
DATASET_PATH: str = (
    "/data/raw/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/"
)

INPUT_DIR: str = os.path.join(str(root_path), *DATASET_PATH.split("/")[1:-1])

@st.cache
def load_datasets(input_dir: str) -> List[pd.DataFrame]:
    """load required dataset for the web app

    Args:
        input_dir (str): input directory of the datasets to load

    Returns:
        pd.DataFrame: return the 3 dataframe
    """
    global_death: pd.DataFrame = pd.read_csv(
        input_dir + "/time_series_covid19_deaths_global.csv",
    )
    global_cases: pd.DataFrame = pd.read_csv(
        input_dir + "/time_series_covid19_confirmed_global.csv"
    )
    global_recovered: pd.DataFrame = pd.read_csv(
        input_dir + "/time_series_covid19_recovered_global.csv",
    )

    return [global_death, global_cases, global_recovered]


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


######################### EXPLORATION WEB APP RENDERING #########################


def exploration_web_app_component() -> None:
    """
    render a web app for dataset exploration purposes
    """
    with st.spinner("Loading datasets..."):
        GLOBAL_DEATH, GLOBAL_CASES, GLOBAL_RECOVERED = load_datasets(INPUT_DIR)
        # country_list: pd.DataFrame= np.concatenate([["Worldwide"], pd.unique(GLOBAL_DEATH["Country/Region"])])
        country_list: pd.DataFrame = pd.unique(GLOBAL_DEATH["Country/Region"])
    st.success("*Data successfuly loaded !*")

    st.title(
        "Data exploration of the COVID situation (cases / deaths / recovered) by country"
    )

    ######## COUNTRY SELECTION ########

    st.write("> Select a country to explore")

    country: str = st.selectbox("Country", country_list)
    province_list: List[str] = GLOBAL_DEATH[GLOBAL_DEATH["Country/Region"] == country]  # type: ignore

    # Creating dataset taking into account the province or not
    if len(GLOBAL_DEATH[GLOBAL_DEATH["Country/Region"] == country]) > 1:
        st.info("The country selected has multiple provinces or states, select one : ")
        province: str = st.selectbox("Province/State", province_list)
        death_time_series: pd.DataFrame = convert_df_to_time_series(
            GLOBAL_DEATH[
                (GLOBAL_DEATH["Country/Region"] == country)
                & (GLOBAL_DEATH["Province/State"] == province)
            ],
            time_series_name="Deaths",
        )
        cases_time_series: pd.DataFrame = convert_df_to_time_series(
            GLOBAL_CASES[
                (GLOBAL_CASES["Country/Region"] == country)
                & (GLOBAL_CASES["Province/State"] == province)
            ],
            time_series_name="Cases",
        )
        recovered_time_series: pd.DataFrame = convert_df_to_time_series(
            GLOBAL_RECOVERED[
                (GLOBAL_RECOVERED["Country/Region"] == country)
                & (GLOBAL_RECOVERED["Province/State"] == province)
            ],
            time_series_name="Recovered",
        )

    else:
        death_time_series: pd.DataFrame = convert_df_to_time_series(
            GLOBAL_DEATH[GLOBAL_DEATH["Country/Region"] == country],
            time_series_name="Deaths",
        )
        cases_time_series: pd.DataFrame = convert_df_to_time_series(
            GLOBAL_CASES[GLOBAL_CASES["Country/Region"] == country],
            time_series_name="Cases",
        )
        recovered_time_series: pd.DataFrame = convert_df_to_time_series(
            GLOBAL_RECOVERED[GLOBAL_RECOVERED["Country/Region"] == country],
            time_series_name="Recovered",
        )

    ######## Vizualisation ########

    st.header("Deaths")
    st.line_chart(death_time_series)

    st.header("Cases")
    st.line_chart(cases_time_series)

    st.header("Recovered")
    st.line_chart(recovered_time_series)

    st.header("Similarity between units (using Dynamic Time Warping)")
    c1, c2, c3 = st.columns(3)
    c1.subheader("Cases and deaths")
    c1.metric(
        "DTW distance",
        round(
            dtw.distance(
                death_time_series.Deaths.fillna(0).to_list(),
                cases_time_series.Cases.fillna(0).to_list(),
                2,
            ),
        ),
    )
    c2.subheader("Cases and recovery")
    c2.metric(
        "DTW distance",
        round(
            dtw.distance(
                recovered_time_series.Recovered.fillna(0).to_list(),
                cases_time_series.Cases.fillna(0).to_list(),
                2,
            ),
        ),
    )
    c3.subheader("Recovery and deaths")
    c3.metric(
        "DTW distance",
        round(
            dtw.distance(
                death_time_series.Deaths.fillna(0).to_list(),
                recovered_time_series.Recovered.fillna(0).to_list(),
                2,
            ),
        ),
    )


if __name__ == "__main__":
    exploration_web_app_component()
