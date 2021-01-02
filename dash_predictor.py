# -*- coding: utf-8 -*-

# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from fbprophet import Prophet
import numpy as np
from fbprophet.plot import plot_plotly

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#set desired region and column name with value for prediction
country = 'United States'
region = 'Washington'
prediction = 'ConfirmedCases'
prediction_length = 30

#URL for data from Oxford: updated daily
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'

#Load data
df = pd.read_csv(DATA_URL, \
                    parse_dates=['Date'], \
                    encoding="ISO-8859-1", \
                    usecols = ['Date','CountryName','RegionName', prediction], \
                    dtype={"RegionName": str, \
                           "CountryName":str}, \
                    error_bad_lines=False)

#extract desired regional data
df = df[df['CountryName'] == country][:-1]
if region:
    df = df[df['RegionName'] == region][:-1]
else:
    df = df[df['Jurisdiction'] == 'NAT_TOTAL']

#prepare dataframe for Prophet
df = df.drop(columns = ['CountryName','RegionName'])
df = df.rename(columns = {'Date':'ds','ConfirmedCases':'y'})
df = df.fillna(method = 'ffill')
df = df.fillna(0)

#create Prophet instance
m = Prophet(seasonality_mode = 'multiplicative', \
                yearly_seasonality = False, \
                daily_seasonality = False, \
                weekly_seasonality = True) \
#add holidays (only support United States)
if country == 'United States':
    m.add_country_holidays(country_name='US')

#fit model and create prediction
m.fit(df)
future = m.make_future_dataframe(periods=prediction_length)
forecast = m.predict(future)

fig = plot_plotly(m, forecast, changepoints=False, \
                xlabel="Date", ylabel=prediction, \
                uncertainty=True, \
                plot_cap=True)

fig.layout.title = {'text': f'True and Predicted {prediction} in {region}'}
fig.update_layout(showlegend=True)

app.layout = html.Div(children=[
    html.H1(children='COVID Predictor'),

    html.Div(children='''
        COVID-19 Cumulative Case Prediction
    '''),

    dcc.Graph(
        id='prediction_graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)