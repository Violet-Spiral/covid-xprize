# covid-xprize
Violet-Spiral Covid Xprize work.

We were inspired by the COVID-19 Xprize competition to create a model of daily cases to predict future infection rates.  We prepared the data and used both SARIMAX and Facebook Prophet models to make predictions.  We achieved 84% accuracy with Facebook Prophet and 99% accuracy with tuned SARIMAX model.  

![Covid Prediction Graph](https://github.com/Violet-Spiral/covid-xprize/blob/josh/models/SARIMA_prediction2021_01_05.png)

We split the data between case totals before December first as a training set and after December 1st to validate the models.  

We introduced government interventions to attempt to improve the accuracy.  The accuracy on the test range improved when proprosed intervention changes were provided, but we suspect that the model reversed the direction of the causation of the interventions.  While governments increase the stringency of the restrictions in response to increased cases, the model seemed to predict that increased stringency would increase infection rates.
