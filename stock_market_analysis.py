import requests
import json
from datetime import date, datetime, timedelta
from pymongo import MongoClient
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# Fetch data for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
api_key = 'api_key'

def fetch_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=fill&apikey={api_key}'
    r = requests.get(url)
    return r.json()

# Fetch data for each symbol and store in a dictionary
data = {symbol: fetch_data(symbol, api_key) for symbol in symbols}

# print(data)


client = MongoClient('mongodb+srv://mugilmithran01:<password>@cluster0.cfxmvyc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['stock_data']

def store_data(symbol, data):
    collection = db[symbol]
    time_series = data['Time Series (Daily)']
    formatted_data = [
        {'date': datetime.strptime(date, '%Y-%m-%d'), **values}
        for date, values in time_series.items()
    ]
    collection.insert_many(formatted_data)

# Store data for each symbol
for symbol, data in data.items():
    store_data(symbol, data)
    
    
def load_data(symbol):
    collection = db[symbol]
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def process_data(df):
    df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '6. volume': 'volume',
    }, inplace=True)
    df['adj_close'] = pd.to_numeric(df['close'])
    df['50_MA'] = df['adj_close'].rolling(window=50).mean()
    df['200_MA'] = df['adj_close'].rolling(window=200).mean()
    df['Bollinger_Upper'] = df['50_MA'] + 2*df['adj_close'].rolling(window=50).std()
    df['Bollinger_Lower'] = df['50_MA'] - 2*df['adj_close'].rolling(window=50).std()
    df['RSI'] = compute_rsi(df['adj_close'])
    return df

def compute_rsi(df, window=14):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

def plot_rsi_with_candlestick(df, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=(f'Candlestick Chart with RSI for {symbol}', 'RSI'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='blue')
    ), row=2, col=1)

    fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                  line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                  line=dict(color='green', dash='dash'), row=2, col=1)

    # Update layout with dark theme and grid
    fig.update_layout(
        title=f'Candlestick Chart with RSI for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        yaxis2=dict(title='RSI', showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=700
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)
df = compute_rsi(df)

fig = plot_rsi_with_candlestick(df, symbol)
fig.show()

def compute_short_term_moving_average(df, window=50):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df[f'{window}_MA'] = df['adj_close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    return df

def plot_moving_averages_with_candlestick(df, symbol):
    fig = go.Figure()

#     # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # short_term Moving Average
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['50_MA'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='blue', width=1.5)
    ))

    # Update layout with dark theme and grid
    fig.update_layout(
        title=f'Candlestick Chart with 50-Day Moving Averages for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=700
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)
df = compute_short_term_moving_average(df)

fig = plot_moving_averages_with_candlestick(df, symbol)
fig.show()

def compute_bollinger_bands(df, window=20, num_std_dev=2):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    rolling_mean = df['adj_close'].rolling(window=window).mean()
    rolling_std = df['adj_close'].rolling(window=window).std()
    df['Bollinger_Mean'] = rolling_mean
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std_dev)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std_dev)
    df.dropna(inplace=True)
    return df

def plot_bollinger_bands_with_candlestick(df, symbol):
    fig = go.Figure()

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Upper'],
        mode='lines',
        name='Bollinger Upper',
        line=dict(color='blue', width=1.5, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Lower'],
        mode='lines',
        name='Bollinger Lower',
        line=dict(color='blue', width=1.5, dash='dash')
    ))

    # Bollinger Mean
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Mean'],
        mode='lines',
        name='Bollinger Mean',
        line=dict(color='orange', width=1.5)
    ))

    # Update layout with dark theme and grid
    fig.update_layout(
        title=f'Candlestick Chart with Bollinger Bands for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=700
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)
df = compute_bollinger_bands(df)

fig = plot_bollinger_bands_with_candlestick(df, symbol)
fig.show()


def compute_macd(df, short_window=12, long_window=26, signal_window=9):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df['EMA_12'] = df['adj_close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['adj_close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def plot_macd_with_candlestick(df, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=(f'Candlestick Chart for {symbol}', 'MACD'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # MACD Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=1.5)
    ), row=2, col=1)

    # Signal Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal_Line'],
        mode='lines',
        name='Signal Line',
        line=dict(color='red', width=1.5)
    ), row=2, col=1)

    # MACD Histogram
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD'] - df['Signal_Line'],
        name='MACD Histogram',
        marker_color='gray'
    ), row=2, col=1)

    # Update layout with dark theme and grid
    fig.update_layout(
        title=f'Candlestick Chart with MACD for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        yaxis2=dict(title='MACD', showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=700
    )

    return fig

# Example usage
symbol = 'AAPL'
df = load_data(symbol)
df = compute_macd(df)

fig = plot_macd_with_candlestick(df, symbol)
fig.show()


def add_volume_analysis(df, volume_window=20):
    df['volume'] = pd.to_numeric(df['5. volume'])
    df['Volume_MA'] = df['volume'].rolling(window=volume_window).mean()
    df['Volume_Color'] = ['green' if close >= open_ else 'red' for close, open_ in zip(df['4. close'], df['1. open'])]
    df.dropna(inplace=True)
    return df

def plot_enhanced_volume_with_candlestick(df, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=(f'Candlestick Chart for {symbol}', 'Volume Analysis'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # Volume Bar Chart with Colors
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color=df['Volume_Color']
    ), row=2, col=1)

    # Volume Moving Average
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_MA'],
        mode='lines',
        name='Volume MA',
        line=dict(color='orange', width=1.5)
    ), row=2, col=1)

    # Update layout with dark theme and grid
    fig.update_layout(
        title=f'Candlestick Chart with Volume Analysis for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        yaxis2=dict(title='Volume', showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=700
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)
df = add_volume_analysis(df)

fig = plot_enhanced_volume_with_candlestick(df, symbol)
fig.show()


def add_indicators(df, short_ma=50, macd_short=12, macd_long=26, macd_signal=9, bollinger_window=20, rsi_window=14, volume_window=20):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    
    # Moving Averages
    df['50_MA'] = df['adj_close'].rolling(window=short_ma).mean()
    
    # Bollinger Bands
    rolling_mean = df['adj_close'].rolling(window=bollinger_window).mean()
    rolling_std = df['adj_close'].rolling(window=bollinger_window).std()
    df['Bollinger_Mean'] = rolling_mean
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    # MACD
    df['EMA_12'] = df['adj_close'].ewm(span=macd_short, adjust=False).mean()
    df['EMA_26'] = df['adj_close'].ewm(span=macd_long, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()

    # RSI
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volume Analysis
    df['volume'] = pd.to_numeric(df['5. volume'])
    df['Volume_MA'] = df['volume'].rolling(window=volume_window).mean()
    df['Volume_Color'] = ['green' if close >= open_ else 'red' for close, open_ in zip(df['4. close'], df['1. open'])]
    df.dropna(inplace=True)
    return df

def plot_all_indicators(df, symbol):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=(
            f'Candlestick Chart for {symbol}', 
            'Volume Analysis',
            'MACD', 
            'RSI'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # Candlestick Chart with MA and Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['50_MA'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='blue', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Upper'],
        mode='lines',
        name='Bollinger Upper',
        line=dict(color='blue', width=1.5, dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Lower'],
        mode='lines',
        name='Bollinger Lower',
        line=dict(color='blue', width=1.5, dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Mean'],
        mode='lines',
        name='Bollinger Mean',
        line=dict(color='orange', width=1.5)
    ), row=1, col=1)

    # Volume Bar Chart with Moving Average
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['5. volume'],
        name='Volume',
        marker_color=df['Volume_Color']
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_MA'],
        mode='lines',
        name='Volume MA',
        line=dict(color='orange', width=1.5)
    ), row=2, col=1)

    # MACD and Signal Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=1.5)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal_Line'],
        mode='lines',
        name='Signal Line',
        line=dict(color='red', width=1.5)
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD'] - df['Signal_Line'],
        name='MACD Histogram',
        marker_color='gray'
    ), row=3, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=1.5)
    ), row=4, col=1)

    fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=4, col=1)
    fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=4, col=1)

    # Update layout with dark theme and grid
    fig.update_layout(
        title=f'Comprehensive Stock Analysis for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='simple_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        yaxis2=dict(title='Volume', showgrid=True, gridcolor='gray', gridwidth=0.5),
        yaxis3=dict(title='MACD', showgrid=True, gridcolor='gray', gridwidth=0.5),
        yaxis4=dict(title='RSI', showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=900
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)
df = add_indicators(df)

fig = plot_all_indicators(df, symbol)
fig.show()

def calculate_trendline(df):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df = df.dropna(subset=['adj_close'])  # Drop rows where adj_close is NaN
    x = np.arange(len(df)).reshape(-1, 1)  # Reshape for linear regression
    y = df['adj_close'].values
    
    model = LinearRegression().fit(x, y)
    trendline = model.predict(x)
    
    return trendline

def plot_candlestick_with_trendline(df, symbol):
    fig = go.Figure()

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Trendline
    trendline = calculate_trendline(df)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=trendline,
        mode='lines',
        name='Trendline',
        line=dict(color='black', width=2)
    ))

    # Update layout
    fig.update_layout(
        title=f'Candlestick Chart with Trendline for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=600
    )

    return fig

# Example usage
symbol = 'AAPL'
df = load_data(symbol)

fig = plot_candlestick_with_trendline(df, symbol)
fig.show()

# Support Resistance
def calculate_support_resistance(df, window=20):
    df['low'] = pd.to_numeric(df['3. low'])
    df['high'] = pd.to_numeric(df['2. high'])
    # Rolling window to calculate support and resistance
    df['Support'] = df['low'].rolling(window=window).min()
    df['Resistance'] = df['high'].rolling(window=window).max()
    df.dropna(inplace=True)
    support_level = df['Support'].iloc[-1]  # Most recent support level
    resistance_level = df['Resistance'].iloc[-1]  # Most recent resistance level
    return support_level, resistance_level

def plot_candlestick_with_support_resistance(df, symbol):
    support_level, resistance_level = calculate_support_resistance(df)
    fig = go.Figure()

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Support and Resistance Levels
    fig.add_trace(go.Scatter(
        x=[df.index.min(), df.index.max()],
        y=[support_level, support_level],
        mode='lines',
        name='Support Level',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[df.index.min(), df.index.max()],
        y=[resistance_level, resistance_level],
        mode='lines',
        name='Resistance Level',
        line=dict(color='red', width=2)
    ))

    # Update layout
    fig.update_layout(
        title=f'Candlestick Chart with Support and Resistance for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='simple_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=600
    )

    return fig

# Example usage
symbol = 'AAPL'
df = load_data(symbol)

fig = plot_candlestick_with_support_resistance(df, symbol)
fig.show()

# Trend analysis
def calculate_moving_average(df, short_window=100):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df['Short_MA'] = df['adj_close'].rolling(window=short_window).mean()
    # df['Long_MA'] = df['adj_close'].rolling(window=long_window).mean()
    return df

def calculate_trendline(df):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df = df.dropna(subset=['adj_close'])
    x = np.arange(len(df)).reshape(-1, 1)
    y = df['adj_close'].values
    
    model = LinearRegression().fit(x, y)
    trendline = model.predict(x)
    
    return trendline

def plot_candlestick_with_trend_analysis(df, symbol):
    trendline = calculate_trendline(df)
    df = calculate_moving_average(df)
    
    fig = go.Figure()

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Trendline
    fig.add_trace(go.Scatter(
        x=df.index,
        y=trendline,
        mode='lines',
        name='Trendline',
        line=dict(color='cyan', width=2, dash='dash')
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Short_MA'],
        mode='lines',
        name='Short-Term MA',
        line=dict(color='blue', width=1.5)
    ))

    # Update layout
    fig.update_layout(
        title=f'Candlestick Chart with Trend Analysis for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=600
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)

fig = plot_candlestick_with_trend_analysis(df, symbol)
fig.show()

# Suggestion
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df['MA20'] = df['adj_close'].rolling(window=window).mean()
    df['STD20'] = df['adj_close'].rolling(window=window).std()
    df['Bollinger_Upper'] = df['MA20'] + (df['STD20'] * num_std_dev)
    df['Bollinger_Lower'] = df['MA20'] - (df['STD20'] * num_std_dev)
    return df

def calculate_moving_average(df, window=50):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df['50_MA'] = df['adj_close'].rolling(window=window).mean()
    return df

def calculate_volume_average(df, window=20):
    df['volume'] = pd.to_numeric(df['5. volume'])
    df['Volume_Avg'] = df['volume'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def plot_candlestick_with_indi(df, symbol):
    df['adj_close'] = pd.to_numeric(df['4. close'])
    df['volume'] = pd.to_numeric(df['5. volume'])
    
    fig = go.Figure()

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Upper'],
        mode='lines',
        name='Bollinger Upper',
        line=dict(color='orange', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Bollinger_Lower'],
        mode='lines',
        name='Bollinger Lower',
        line=dict(color='orange', width=1.5)
    ))

    # 50-Day Moving Average
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['50_MA'],
        mode='lines',
        name='50-Day Moving Average',
        line=dict(color='blue', width=1.5)
    ))

    # Volume (scaled)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='gray',
        yaxis='y2'
    ))

    # Update layout
    fig.update_layout(
        title=f'Candlestick Chart with Indicators and Buy Signals for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=800
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)

fig = plot_candlestick_with_indi(df, symbol)
fig.show()


# Fibanocci retracement

def calculate_fibonacci_levels(df):
    df['low'] = pd.to_numeric(df['3. low'])
    df['high'] = pd.to_numeric(df['2. high'])
    high = df['high'].max()
    low = df['low'].min()
    
    difference = high - low
    
    levels = {
        '0.0%': high,
        '23.6%': high - difference * 0.236,
        '38.2%': high - difference * 0.382,
        '50.0%': high - difference * 0.500,
        '61.8%': high - difference * 0.618,
        '100.0%': low
    }
    
    return levels

def plot_candlestick_with_fibonacci(df, symbol):
    levels = calculate_fibonacci_levels(df)
    
    fig = go.Figure()

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Fibonacci Retracement Levels with different colors
    colors = {
        '0.0%': 'blue',
        '23.6%': 'purple',
        '38.2%': 'orange',
        '50.0%': 'red',
        '61.8%': 'green',
        '100.0%': 'gray'
    }

    # Fibonacci Retracement Levels
    for level_name, level_value in levels.items():
        fig.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()],
            y=[level_value, level_value],
            mode='lines',
            name=f'Fib {level_name}',
            line=dict(color=colors[level_name], width=1.5)
        ))

    # Update layout
    fig.update_layout(
        title=f'Candlestick Chart with Fibonacci Retracement Levels for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5
        ),
        # yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.5),
        height=800
    )

    return fig

# Execution
symbol = 'AAPL'
df = load_data(symbol)

fig = plot_candlestick_with_fibonacci(df, symbol)
fig.show()