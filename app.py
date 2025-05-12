import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Title
st.header('📈 Stock Market Predictor')

# Input stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2025-01-31'

# Load stock data
data = yf.download(stock, start, end)
st.subheader('Raw Stock Data')
st.write(data)

# Train-test split
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_combined)

# Moving Averages
st.subheader('Price vs MA50')
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(ma_100, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100, 'r', label='MA100')
plt.plot(ma_200, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig3)

# Create sequences
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])
x, y = np.array(x), np.array(y)

# Load model (local file in repo)
model_path = "keras_model.h5"
model = tf.keras.models.load_model(model_path)

# Predict and inverse scale
y_predicted = model.predict(x)
y_predicted = scaler.inverse_transform(y_predicted)
y_true = scaler.inverse_transform(y.reshape(-1, 1))

# Final prediction plot
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y_true, 'r', label='Original Price')
plt.plot(y_predicted, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
