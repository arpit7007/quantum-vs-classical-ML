import pandas as pd

def load_ohlcv_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.reset_index(drop=True)
    return df
