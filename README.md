# NASA-temperature-Analysis
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/POWER_Point_Daily_20180101_20230331_025d6033N_085d1370E_LST.csv')
data.head()


data.describe()

data

data1 = data[["YEAR", "MO", "DY"]].copy()
data1.columns = ["year", "month", "day"]
data1['Date']= pd.to_datetime(data1)
data1['T2M']=data['T2M']
data1

df = data1.set_index(pd.DatetimeIndex(data1['Date']))
df

# convert the 'Date column to a datatime type
data1['Date'] = pd.to_datetime(data1['Date'])

# Set the 'Date column as the index
data1.set_index('Date', inplace=True)
data1

import matplotlib.pyplot as plt
# create a line plot
plt.plot(data.index,data['T2M'])
# add label and title
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('NASA Temperature Data')
# display the plot
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition =seasonal_decompose(data['T2M'] ,period = 360)
trend = decomposition.trend
seasonality=decomposition.seasonal
residuals= decomposition.resid
decomposition.resid

plt.figure(figsize=(10,8))
plt.subplot(411)
plt.plot(data.index,data['T2M'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(data.index,trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(data.index,seasonality, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(data.index,residuals, label='Residuals')
plt.legend(loc='best')


plt.tight_layout()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# plot ACF
plt.figure(figsize=(10,4))
plot_acf(data['T2M'], lags=50)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function(ACF)')
plt.show()

# plot PACF
plt.figure(figsize=(10,4))
plot_pacf(data['T2M'], lags=50)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function(PACF)')
plt.show()

from statsmodels.tsa.stattools import adfuller
result=adfuller(data['T2M'])
adf_statistic = result[0]
p_value=result[1]
print(f'ADF Statistic: {adf_statistic:.4f}')
print(f'p_value: {p_value:.4f}')
# define a function to interpret the test results
def interpret_adf_results(p_value):
  if p_value < 0.05: #compared to 5% error
    print('The time series is stationary')
  else:
    print('The time series is non-stationary')
# interpret the test results
interpret_adf_results(p_value)


!pip install pycaret

df=data1[['T2M']]

from pycaret.time_series import TSForecastingExperiment
fig_kwargs={'renderer': 'notebook'}
forecast_horizon= 10
fold=3
exp=TSForecastingExperiment()
exp.setup(data=df,fh=forecast_horizon,fold=fold, session_id=123,fig_kwargs=fig_kwargs)
exp.check_stats()

exp.models()

arima = exp.create_model('arima')
arima

tuned_arima= exp.tune_model(arima)
tuned_arima

best=exp.compare_models()

exp.get_metrics()

exp.predict_model(best)

from pycaret.time_series import*
exp_name=setup(data=df,fh=12)
top3=compare_models(n_select=3)
blender=blend_models(top3)


from pycaret.datasets import get_data
airline = get_data('airline')
from pycaret.time_series import *
exp_name = setup(data = df,  fh = 12)
plot_model(plot="diff", data_kwargs={"order_list": [1, 2], "acf": True, "pacf": True})
plot_model(plot="diff", data_kwargs={"lags_list": [[1], [1, 12]], "acf": True, "pacf": True})
arima = create_model('arima')
plot_model(plot = 'ts')
plot_model(plot = 'decomp', data_kwargs = {'type' : 'multiplicative'})
plot_model(plot = 'decomp', data_kwargs = {'seasonal_period': 24})
plot_model(estimator = arima, plot = 'forecast', data_kwargs = {'fh' : 24})
tuned_arima = tune_model(arima)
plot_model([arima, tuned_arima], data_kwargs={"labels": ["Baseline", "Tuned"]})
