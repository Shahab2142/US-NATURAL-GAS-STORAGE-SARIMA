#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:58:41 2024

@author: shahab-nasiri
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_log_error


#Import data from EIA - Weekly Natural Gas Storage Report
data = pd.read_csv('NaturalGasStorageUS.csv')
data.tail()

plt.figure(figsize=[12, 5]); # Set dimensions for figure
data.plot(x='Week ending', y='Total Lower 48', figsize = (12, 3.5), legend = True, color='g')
plt.title('Lower 48 Natural Gas Storage - BCF')
plt.ylabel('Storage Lower 48 - BCF')
plt.xlabel('Week')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Augmented Dickey-Fuller test
ad_fuller_result = adfuller(data['Total Lower 48'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

best_model = SARIMAX(data['Total Lower 48'], order=(2, 1, 1), seasonal_order=(2, 1, 1, 52)).fit(dis=-1)
print(best_model.summary())

#Diagnosing the model residuals
best_model.plot_diagnostics(figsize=(15,12));

#Forecasting 3 years steps ahead
forecast_values = best_model.get_forecast(steps = 156)

#Confidence intervals of the forecasted values
forecast_ci = forecast_values.conf_int()

#Plot the data
ax = data.plot(x='Week ending', y='Total Lower 48', figsize = (12, 5), legend = True, color='g')

#Plot the forecasted values 
forecast_values.predicted_mean.plot(ax=ax, label='Forecasts', figsize = (12, 5), grid=True)

#Plot the confidence intervals
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[: , 0],
                forecast_ci.iloc[: , 1], color='#D3D3D3', alpha = .5)
plt.title('US Natural Gas Storage - BCF', size = 16)
plt.ylabel('Total Lower 48 (Bcf)', size=12)
plt.xlabel('Date', size=12)
plt.legend(loc='upper center', prop={'size': 12})
#annotation
ax.text(570, 100, 'Forecasted Values Until 2023', fontsize=11,  color='blue')
plt.show()

#divide into train and validation set to calculate R-squared score and mean absolute percentage error 
train = data[:int(0.85*(len(data)))]
test = data[int(0.85*(len(data))):]
start=len(train)
end=len(train)+len(test)-1
predictions = best_model.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA Predictions')
evaluation_results = pd.DataFrame({'r2_score': r2_score(test['Total Lower 48'], predictions)}, index=[0])
evaluation_results['mean_absolute_error'] = mean_absolute_error(test['Total Lower 48'], predictions)
evaluation_results['mean_squared_error'] = mean_squared_error(test['Total Lower 48'], predictions)
evaluation_results['root_mean_squared_error'] = np.sqrt(mean_squared_error(test['Total Lower 48'], predictions))
evaluation_results['mean_absolute_percentage_error'] = np.mean(np.abs(predictions - test['Total Lower 48'])
                                                               /np.abs(test['Total Lower 48']))*100 
evaluation_results