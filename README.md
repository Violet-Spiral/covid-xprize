# covid-xprize
Violet-Spiral Covid Xprize work.

We were inspired by the COVID-19 Xprize competition to create a model of daily cases to predict future infection rates.  We prepared the data and used both SARIMAX and Facebook Prophet models to make predictions.  We achieved 84% accuracy with Facebook Prophet and 99% accuracy with tuned SARIMAX model.  

![Covid Prediction Graph](https://github.com/Violet-Spiral/covid-xprize/blob/josh/models/SARIMA_prediction2021_01_05.png)

We split the data between case totals before December first as a training set and after December 1st to validate the models.  

We introduced government interventions to attempt to improve the accuracy.  The accuracy on the test range improved when proprosed intervention changes were provided, but we suspect that the model reversed the direction of the causation of the interventions.  While governments increase the stringency of the restrictions in response to increased cases, the model seemed to predict that increased stringency would increase infection rates.

We also tried introducing a lag on the exogenous variables.  Instead of feeding the model the current NPIs (no pharmeceutical interventions, things like school clostures, mask mandates, and travel restrictions) we gave it what they were two weeks before, hoping perhaps that the model would find the lagged relationships between the interventions and the cases.  This, too, led to the same problem of the model learning the reversed causitive relationship between exogenous and endogenous variables.  The only other explanation is that stricter mandates actually cause rising case rates, and this seems unlikely.  Much more evidence would need to be gathered to support such an outrageous finding!

Instead we turned our attentions back to the Seasonal Auto-Regressive Integrated Moving Average model (SARIMA).  While the statsmodels implementation of this model also accepts exogenous variables (SARIMAX), we decided to stick with modeling how previous cases and deaths can predict future ones.

The data has a clear 7 day season.  This likely corresponds to a weekly rhythm of either increased and decreased testing, such as most people going in for testing on weekends, or a weekly rhythm of reporting, such as labs batch reporting on certain days.  It's important to keep in mind that the case data represents when infections are reported, not when they actually occur.

With a 7 day seasonality explicitly designated in the model (The 'S' in SARIMAX), we also had to find the proper differencing to account for the upward trend in the data.  A first order differencing transforms cumulative case data into daily new cases.  Unfortunately, daily case rates are trending upward around the world as well.  A second order differencing was needed to eliminate the trend and continue to make our data more stationary.  Stationarity, or the lack of seasonality, trend, and heteroskadicity, is necessary for an ARMA model to make valid predictions.  

SARIMAX is a wonderful model type, because with the seasonality arguments and the differencing arguments the first two can be accounted for in the model without data preprocessing.  However, the standard deviations of the changes in the data have also been rising over this year.  The high and low swings have greater magnitude.  This creates heteroskadicity, or a change in standard deviation of the residuals over time.  We corrected this by log transforming the data before giving it to the model.  This also necessitated an inverse transform on the model's predictions to bring them back to the right scale, but this was not hard.  `np.log(X)` works to log transform a numpy array or a pandas dataframe or seriest, and `np.exp` inverts this transformation.

We validated our model by training it on data at the national level in the United States from March 2020 until November 2020 and tested using the month of December.  Most regions in the world did not have significant cases before March and at the time of the modeling that gave us 8 months of training data and one month of testing data.  It also represented an interesting test because case rates began to climb dramatically in late September and that trend increased through November and even more through December.  December's data had larger positive trend and higher standard deviations than the training data and seemed to be a different beast than the other months, and the Thanksgiving and Christmas holidays represented interesting inflection points.  

The model, with the proper lags, seemed to do well, however, predicting actual cases with only .009 mean absolute percentage error.  The proper lags were estimated by referring to auto-regressive plots and partial auto-regressive plots, but were ultimately chosen with an exhaustive search through several options.  

Now, what remains to be seen is whether this exhaustive search to minimize the error in the December validation set also overfit the model.  The next test will be to take the same model and test it against January data to see if it continues to perform at high accuracy.

## Update:

It's now 1/19/2021 and it seemed like a good time do another accuracy test.  I trained the same SARIMAX model with the same lags on COVID-19 cumulative case data at the national level in the United States, up until the new year (01-01-2021) and the accuracy is still almost 99% on the prediction for January.

![January SARIMAX prediction](models/SARIMA_prediction2021_1_19.png)

My takeaway here is that COVID cases in the United States is very predictable using only the cumulative case data itself, with appropriate AR and MA lags and the right lags for the weekly seasonality component.  This seems to suggest that government interventions, on the whole, have not really been game-changers in the progress of the disease, at least not since November.  

We can only hope that my model is defeated by the rollout of vaccines.  That would be a good failure of my model.
