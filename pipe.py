"""
title: FX Forecaster
requirements: curl_cffi, yfinance, numpy, tensorflow, pandas, isoweek, scikit-learn, keras
"""

from pydantic import BaseModel, Field

import yfinance as yf
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import datetime as dt
import isoweek
import math
from sklearn.preprocessing import MinMaxScaler
import curl_cffi

curl_cffi.requests.Session(impersonate="chrome")

year = 2025
time_cols = ["week", "day", "hour", "minute"]


# ======= Helpers =======
# Fetch yfinance data
def get_data():
    offset = 0
    now = pd.Timestamp.today()

    # No stock market data on weekends
    if now.day_of_week == 5:
        offset = 1
    elif now.day_of_week == 6:
        offset = 2

    now = now - pd.Timedelta(days=offset)
    now = now.floor("10 min")
    prev = now - pd.Timedelta(minutes=50)
    data = yf.download("EURUSD=X", start=prev, end=now, interval="5m")
    return data


# Convert data to 10-minute intervals and derive columns
def clean_data(data):
    count = 0
    high = 0.00000
    low = 9.99999
    open_price = 0.00000
    close_price = 0.00000

    clean = pd.DataFrame(
        columns=[
            "week",
            "day",
            "hour",
            "minute",
            "open",
            "high",
            "low",
            "close",
            "momentum",
            "avg",
            "range",
            "ohlc",
        ]
    )

    for row in data.itertuples():
        if row[2] > high:
            high = row[2]

        if row[3] < low:
            low = row[3]

        if count == 0:
            minute = row[0].minute
            open_price = row[4]

        if count == 1:
            close_price = row[1]
            hour = row[0].hour
            day = row[0].weekday()
            week = row[0].week
            momentum = open_price = close_price
            avg = (low + high) / 2
            price_range = high - low
            ohlc = (open_price + high + low + close_price) / 4

            new_row = [
                week,
                day,
                hour,
                minute,
                open_price,
                high,
                low,
                close_price,
                momentum,
                avg,
                price_range,
                ohlc,
            ]
            clean.loc[len(clean)] = new_row

            count = 0
            high = 0
            low = 999
            continue

        count += 1
        # end for
    return clean


def to_normal_vector(value):
    angle = value * math.pi * 2
    x = math.sin(angle)
    y = math.cos(angle)
    x = (x + 1) / 2
    y = (y + 1) / 2
    return (x, y)


# Convert pandas dataframe to normalized time values
def normalize_time(df):
    time = pd.DataFrame(
        columns=[
            "week_x",
            "week_y",
            "day_x",
            "day_y",
            "hour_x",
            "hour_y",
            "minute_x",
            "minute_y",
        ]
    )

    for row in df.itertuples():
        week_ratio = row[1] / isoweek.Week.last_week_of_year(year).week
        week = to_normal_vector(week_ratio)
        day_ratio = row[2] / 7
        day = to_normal_vector(day_ratio)
        hour_ratio = row[3] / 24
        hour = to_normal_vector(hour_ratio)
        minute_ratio = row[4] / 6
        minute = to_normal_vector(minute_ratio)

        new_row = [
            week[0],
            week[1],
            day[0],
            day[1],
            hour[0],
            hour[1],
            minute[0],
            minute[1],
        ]
        time.loc[len(time)] = new_row

    return time


# CODE MODIFIED FROM https://www.geeksforgeeks.org/deep-learning/time-series-forecasting-using-tensorflow/
def create_sequences(data, seq_length=5):
    input_data, target = [], []
    for i in range(len(data) - seq_length):
        input_data.append(data.iloc[i : i + seq_length])
        target.append(data.iloc[i + seq_length])
    return np.array(input_data), np.array(target)


# CODE MODIFIED FROM https://www.geeksforgeeks.org/deep-learning/time-series-forecasting-using-tensorflow/


# Reverse normalization of prices
def un_normalize(data):
    data = data.reshape(1, -1)
    scaler.fit(data)
    return scaler.inverse_transform(data)


def float_to_string(num):
    return ("%.2f" % (num,)).rstrip("0").rstrip(".")


# Convert time into format appropriate for graph
def time_to_string(series):
    week = float_to_string(series["week"])
    day = float_to_string(series["day"])
    hour = float_to_string(series["hour"])
    minute = float_to_string(series["minute"])

    stamp = str(year) + " " + week + " " + day + " " + hour + " " + minute
    return dt.datetime.strptime(stamp, "%Y %U %w %H %M")


# ======= Helpers end =========


class Pipe:
    class Valves(BaseModel):
        MODEL_ID: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()

    def pipe(self, body: dict):
        data = get_data()
        clean = clean_data(data)
        time = normalize_time(clean)
        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "momentum",
            "avg",
            "range",
            "ohlc",
        ]
        prices = clean[price_columns]
        scaler = MinMaxScaler()
        scaler.set_output(transform="pandas")
        prices = scaler.fit_transform(prices.transpose())  # for one value
        prices = prices.transpose()
        prices.reset_index(drop=True, inplace=True)
        norm_data = pd.concat([time, prices], axis=1)
        norm_data.loc[len(norm_data)] = 0
        # step = 50 # predict the 60th minute from 50
        X, y = create_sequences(norm_data)
        model = keras.models.load_model("model.keras")
        prediction = model.predict(X)
        pred_prices = un_normalize(pred)
        last_time = clean[-1:]
        last_time = last_time.iloc[0]
        last_time = last_time[time_cols]
        pred_time = time_to_string(last_time)
        j = clean.to_numpy()
        ticker = []
        for row in j:
            time_data = pd.Series(row[:4], index=time_cols)
            t = time_to_string(time_data)
            date = t.strftime("%Y-%d-%m")
            open_price = i[4]
            high = i[5]
            low = i[6]
            close = i[7]
            momentum = i[8]
            avg = i[9]
            price_range = i[10]
            ohlc = i[11]

            ticker.append(
                {
                    "Date": date,
                    "Open": open_price.item(),
                    "High": high.item(),
                    "Low": low.item(),
                    "Close": close.item(),
                }
            )

        # PROVIDED BY @D3 AT OBSERVABLE HQ UNDER ISC LICENSE: https://observablehq.com/@d3/candlestick-chart/
        return """
        chart = {
        
          // Declare the chart dimensions and margins.
          const width = 928;
          const height = 600;
          const marginTop = 20;
          const marginRight = 30;
          const marginBottom = 30;
          const marginLeft = 40;
        
          // Declare the positional encodings.
          const x = d3.scaleBand()
              .domain(d3.utcDay
                  .range(ticker.at(0).Date, +ticker.at(-1).Date + 1)
                  .filter(d => d.getUTCDay() !== 0 && d.getUTCDay() !== 6))
              .range([marginLeft, width - marginRight])
              .padding(0.2);
        
          const y = d3.scaleLog()
              .domain([d3.min(ticker, d => d.Low), d3.max(ticker, d => d.High)])
              .rangeRound([height - marginBottom, marginTop]);
        
          // Create the SVG container.
          const svg = d3.create("svg")
              .attr("viewBox", [0, 0, width, height]);
        
          // Append the axes.
          svg.append("g")
              .attr("transform", `translate(0,${height - marginBottom})`)
              .call(d3.axisBottom(x)
                .tickValues(d3.utcMonday
                    .every(width > 720 ? 1 : 2)
                    .range(ticker.at(0).Date, ticker.at(-1).Date))
                .tickFormat(d3.utcFormat("%-m/%-d")))
              .call(g => g.select(".domain").remove());
        
          svg.append("g")
              .attr("transform", `translate(${marginLeft},0)`)
              .call(d3.axisLeft(y)
                .tickFormat(d3.format("$~f"))
                .tickValues(d3.scaleLinear().domain(y.domain()).ticks()))
              .call(g => g.selectAll(".tick line").clone()
                .attr("stroke-opacity", 0.2)
                .attr("x2", width - marginLeft - marginRight))
              .call(g => g.select(".domain").remove());
        
          // Create a group for each day of data, and append two lines to it.
          const g = svg.append("g")
              .attr("stroke-linecap", "round")
              .attr("stroke", "black")
            .selectAll("g")
            .data(ticker)
            .join("g")
              .attr("transform", d => `translate(${x(d.Date)},0)`);
        
          g.append("line")
              .attr("y1", d => y(d.Low))
              .attr("y2", d => y(d.High));
        
          g.append("line")
              .attr("y1", d => y(d.Open))
              .attr("y2", d => y(d.Close))
              .attr("stroke-width", x.bandwidth())
              .attr("stroke", d => d.Open > d.Close ? d3.schemeSet1[0]
                  : d.Close > d.Open ? d3.schemeSet1[2]
                  : d3.schemeSet1[8]);
        
          // Append a title (tooltip).
          const formatDate = d3.utcFormat("%B %-d, %Y");
          const formatValue = d3.format(".2f");
          const formatChange = ((f) => (y0, y1) => f((y1 - y0) / y0))(d3.format("+.2%"));
        
          g.append("title")
              .text(d => `${formatDate(d.Date)}
        Open: ${formatValue(d.Open)}
        Close: ${formatValue(d.Close)} (${formatChange(d.Open, d.Close)})
        Low: ${formatValue(d.Low)}
        High: ${formatValue(d.High)}`);
        
          return svg.node();
        }
        """
         # PROVIDED BY @D3 AT OBSERVABLE HQ UNDER ISC LICENSE: https://observablehq.com/@d3/candlestick-chart/
