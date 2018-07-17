import pandas as pd


def on_balance_volume(df, n):
    i = 0
    OBV = [0]
    close_values = df['Close'].values
    volume_values = df['Volume'].values

    while i < df.index[-1]:
        next_close = close_values[i + 1]
        close = close_values[i]
        next_volume = volume_values[i + 1]

        if next_close - close > 0:
            OBV.append(next_volume)
        if next_close - close == 0:
            OBV.append(0)
        if next_close - close < 0:
            OBV.append(-next_volume)
        i = i + 1

    series = pd.Series(OBV)
    return pd.Series(series.rolling(n).mean()).values


def ppsr(df):
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


def simple_moving_average(df, n):
    MA = df['Close'].rolling(window=n).mean()
    df = df.assign(**{'SMA_' + str(n): MA.values})
    return df
