
import streamlit as st
import yfinance as yf
from datetime import date
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Cryptocurrency Price Prediction App')

st.markdown("""*Developed By Abhinav Tiwari and Aditya Singh*""")

symbols = ('BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'DOGE-USD', 'BNB-USD')
selected_symbol = st.selectbox('Select cryptocurrency for prediction', symbols)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data 
def load_data(symbol):
    data = yf.download(symbol, start=START, end=TODAY)
    df = pd.DataFrame(data)
    df = df.reset_index()
    df = df[['Date', 'Open', 'Close']]
    df = df.rename(columns={"Date": "ds", "Open": "y_open", "Close": "y_close"})
    return df

data_load_state = st.text('Loading data...')
data = load_data(selected_symbol)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.set_index('ds').tail(10))

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y_open'], name="crypto_open"))
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y_close'], name="crypto_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Linear Regression.
df_train = data[['ds','y_close']]
df_train = df_train.rename(columns={"ds": "ds", "y_close": "y"})

# Extracting features from the date
df_train['year'] = df_train['ds'].dt.year
df_train['month'] = df_train['ds'].dt.month
df_train['day'] = df_train['ds'].dt.day
df_train['dayofweek'] = df_train['ds'].dt.dayofweek
df_train['weekofyear'] = df_train['ds'].dt.weekofyear

X = df_train.drop(columns=['ds', 'y'])
y = df_train['y']

model = LinearRegression()
model.fit(X, y)

# Creating future dates for prediction
future_dates = pd.date_range(start=TODAY, periods=period, freq='D')
future_df = pd.DataFrame({'ds': future_dates})
future_df['year'] = future_df['ds'].dt.year
future_df['month'] = future_df['ds'].dt.month
future_df['day'] = future_df['ds'].dt.day
future_df['dayofweek'] = future_df['ds'].dt.dayofweek
future_df['weekofyear'] = future_df['ds'].dt.weekofyear

forecast = model.predict(future_df.drop(columns=['ds']))

# Show and plot forecast
st.subheader('Forecast data')
st.write(future_df.tail())

forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
st.write(forecast_df)

# Plot forecast data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['ds'], y=data['y_close'], name="Actual"))
fig1.add_trace(go.Scatter(x=future_df['ds'], y=forecast, name="Predicted"))
fig1.layout.update(title_text='Cryptocurrency Forecast', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig1)

# Get tomorrow's date
tomorrow = (date.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

# Get the index of tomorrow's date in the future_df DataFrame
tomorrow_index = future_df[future_df['ds'] == tomorrow].index[0]

# Get the predicted price for tomorrow
tomorrow_predicted_price = forecast[tomorrow_index]

# Show the predicted price for tomorrow
st.subheader('Tomorrow Predicted Price')
st.write(f'The predicted price for ({selected_symbol}) for tomorrow ({tomorrow}) is -> **{tomorrow_predicted_price:.2f} USD.**')
