import pandas as pd
from fbprophet import Prophet
import os
from matplotlib.pyplot import savefig
import argparse


def get_simple_covid_data():
    """
    Download latest confirmed cases and deaths from Oxford by both states and countries.
    Create new cases and new deaths columns as the running difference in the confirmed cases and deaths, 
    which are cumulative.
    return resulting dataframe
    """
    #download latest data from Oxford
    DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    df = pd.read_csv(DATA_URL,
                    parse_dates=['Date'],
                    encoding="ISO-8859-1",
                    dtype={"RegionName": str},
                    error_bad_lines=False,
                    usecols = ['Date','Jurisdiction','RegionName',
                               'CountryName','ConfirmedCases','ConfirmedDeaths'])
    
    #forward fill NaNs in confirmed cases and confirmed deaths columns
    #if January 1st is NaN, set to 0
    df.loc[(df.Date == '2020-01-01') & (df['ConfirmedCases'].isna()), 'ConfirmedCases'] = 0
    df.loc[(df.Date == '2020-01-01') & (df['ConfirmedDeaths'].isna()), 'ConfirmedDeaths'] = 0
    df[['ConfirmedCases','ConfirmedDeaths']] = df[['ConfirmedCases','ConfirmedDeaths']].fillna(method = 'ffill')

    #add new cases and new deaths columns
    for state in df[(df['Jurisdiction'] == 'STATE_TOTAL')]['RegionName'].unique():
        state_inds = (df['Jurisdiction'] == 'STATE_TOTAL') & (df['RegionName'] == state)
        df.loc[state_inds, 'NewCases'] = df.loc[state_inds, 'ConfirmedCases'].diff().fillna(0)
        df.loc[state_inds, 'NewDeaths'] = df.loc[state_inds, 'ConfirmedDeaths'].diff().fillna(0)

    for country in df[(df['Jurisdiction'] == 'NAT_TOTAL')]['CountryName'].unique():
        nat_inds = (df['Jurisdiction'] == 'NAT_TOTAL') & (df['CountryName'] == country)
        df.loc[nat_inds, 'NewCases'] = df.loc[nat_inds, 'ConfirmedCases'].diff().fillna(0)
        df.loc[nat_inds, 'NewDeaths'] = df.loc[nat_inds, 'ConfirmedDeaths'].diff().fillna(0)
        
    return df

def predict(country='United States', region = None, days_ahead=30, predict='cases', output_folder = None,
            rolling_mean = False):

    #retrieve latest covid data
    df = get_simple_covid_data()
    
    #subset df by country and state.  Defaults if no regional info is passed is all of United States
    df = df[df['CountryName'] == country]
    if region:
        df = df[(df['CountryName'] == country)
        & (df['Jurisdiction'] == 'STATE_TOTAL')
        & (df['RegionName'] == region)]
    else: 
        df = df[(df['Jurisdiction'] == 'NAT_TOTAL') & (df['CountryName'] == country)]
    
    if predict == 'deaths':
        df = df[['Date','NewDeaths']].rename(columns = {'Date':'ds','NewDeaths':'y'})
    else:
        df = df[['Date','NewCases']].rename(columns = {'Date':'ds','NewCases':'y'})

    #create forecast using Facebook Prophet
    m = Prophet(seasonality_mode = 'multiplicative',
                yearly_seasonality = False,
                daily_seasonality = False,
                weekly_seasonality = True)
    m.add_country_holidays(country_name='US')
    m.fit(df)
    future = m.make_future_dataframe(periods=days_ahead)
    forecast = m.predict(future)[['ds','yhat']].tail(days_ahead+7)
    if rolling_mean:
        forecast['yhat'] = forecast['yhat'].rolling(window=7).mean()
        forecast['yhat'].fillna(df.reset_index()['y'], inplace=True)

    #save forecast as JSON
    json_forecast = pd.DataFrame(columns = ['id','date','prediction'])
    json_forecast['date'] = forecast['ds'].dt.strftime('%m-%d-%Y')
    json_forecast['id'] = range(len(forecast))
    json_forecast['prediction'] = forecast['yhat'].round().astype(int)
    if output_folder:
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    else: 
        output_folder = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(output_folder,'prediction.json')
    json_forecast.to_json(output_file_path, orient='records')

    #create graph of forecasted casesc
    if rolling_mean:
        average = f'7-Day Rolling Average'
    else:
        average = 'Daily Cases'
    if region:
        title = f'{average}: {days_ahead} Day Prediction for {region}, {country}'
    else: 
        title = f'{average}: {days_ahead} Day Prediction for, {country}'
        
    fig = forecast.plot(x='ds', y='yhat', 
                        ylim = (0,forecast['yhat'].max()*1.1),
                        xlim = (forecast['ds'].min(),forecast['ds'].max()),
                        figsize = (10,5),
                        title = title,
                        xlabel = 'Date', ylabel = predict.title(), grid=True)
    output_image_path = os.path.join(output_folder,'prediction_graph.png')
    savefig(output_image_path, dpi=200)
#     return forecast[['ds','yhat']]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country",
                        dest="country",
                        type=str,
                        required=False,
                        help="Country to predict")
    parser.add_argument("-r", "--region",
                        dest="region",
                        type=str,
                        required=False,
                        help="Region to predict")
    parser.add_argument("-d", "--days_ahead",
                        dest="days_ahead",
                        type=str,
                        required=False,
                        help="How many days ahead to predict")
    parser.add_argument("-o", "--output_folder",
                        dest="output_folder",
                        type=str,
                        required=False,
                        help="The path to the folder where prediction JSON and Graph should be written")
    parser.add_argument("-m", "--rolling_mean",
                        dest="rolling_mean",
                        type=bool,
                        required=False,
                        help="True or False, whether to return the 7 day rolling mean")    
    args = parser.parse_args()
    if args.country == None:
        args.country = 'United States'
    if args.days_ahead == None:
        args.days_ahead = 30
    args.days_ahead = int(args.days_ahead)
    args.rolling_mean = bool(args.rolling_mean)
    print(f"Generating predictions {args.days_ahead} days ahead...")
    predict(country = args.country, region = args.region, \
            days_ahead = args.days_ahead, output_folder = args.output_folder, \
            rolling_mean = args.rolling_mean)
    print("Done!")