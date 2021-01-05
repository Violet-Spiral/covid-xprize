from fbprophet import Prophet
import pandas as pd
import numpy as np

def get_covid_data():
    
    #get the latest data from OxCGRT
    DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    full_df = pd.read_csv(DATA_URL,
                    parse_dates=['Date'],
                    encoding="ISO-8859-1",
                    dtype={"RegionName": str,
                           "RegionCode":str,
                           "CountryName":str,
                           "CountryCode":str},
                    error_bad_lines=False)

    #add new cases and new deaths columns

    return full_df

def mean_percent_error(y_test, y_hat):
    from math import e
    error = np.abs(y_test - y_hat)
    percent_error = error/(y_test + e)
    mean_percent_error = percent_error.sum() / len(y_test)
    return mean_percent_error

def find_best_regressors(df, train_df, test_df):
    todrop = [c for c in df.columns if c in ['ds','y','NewCases','NewDeaths', 'ConfirmedCases','ConfirmedDeaths']]
    regressors = df.columns.drop(todrop)

    keepers = []
    trials = pd.DataFrame(columns = ['regressors','MAPE'])
    improving = True
    while improving:
        best = None
        improving = False

        print(f'current keepers are {keepers}')
        for regressor in regressors:
            keepers.append(regressor)
            m = Prophet(seasonality_mode = 'multiplicative',
                            yearly_seasonality = False, 
                            daily_seasonality = False, 
                            weekly_seasonality = True)
            m.add_country_holidays(country_name='US')
            for keeper in keepers:
                m.add_regressor(keeper)
            m.fit(train_df)
            future = m.make_future_dataframe(periods=len(test_df))
            future = pd.merge(future,df[['ds'] + keepers].reset_index(drop=True),how = 'outer', on = 'ds')
            forecast = m.predict(future)
            prophet_mape = mean_percent_error(test_df['y'].values, forecast['yhat'][-len(test_df):].values)
            trials = trials.append({'regressors':f'{keepers}','MAPE':prophet_mape}, ignore_index=True)
            #MAPE has improved
            if prophet_mape == trials['MAPE'].min():
                improving = True
                best = regressor
            keepers.pop()
        if best:
            keepers.append(best)
        if improving:
            regressors = regressors.drop(best)
    return keepers

def train_prophet(train_df, test_df, additional_regressors=None, growth='linear'):
    m = Prophet(seasonality_mode = 'multiplicative',
                yearly_seasonality = False, 
                daily_seasonality = False, 
                weekly_seasonality = True,
               growth=growth)
    m.add_country_holidays(country_name='US')
    if additional_regressors is not None:
        for regressor in additional_regressors:
            m.add_regressor(regressor)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=len(test_df))
    if additional_regressors is not None:
        future = pd.merge(future, additional_regressors, how = 'inner', 
                          left_on = 'ds', right_index=True)
    forecast = m.predict(future)
    return m, forecast