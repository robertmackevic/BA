from pathlib import Path

import pandas as pd


def load_data_from_csv(path_to_csv: Path) -> pd.DataFrame:
    data = pd.read_csv(path_to_csv, index_col=[0])
    data.index = pd.to_datetime(data.index, dayfirst=True)
    data = data.asfreq("H")
    data = data.interpolate()
    return data
