"""
911 Emergency Calls Forecasting Model
June 14th 2019 Hourly Predictions
@author: Monika Scislo
"""

from math import sqrt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import h5py      
import numpy as np
import datetime
import pandas as pd
from sklearn.externals import joblib
import warnings
from statistics import mean
warnings.filterwarnings("ignore")
from keras.models import load_model

# read new data 
# =============================================================================
dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

June14 = pd.read_csv("14June.csv",
                header=0, names=['timeStamp', 'month', 'dayofmonth', 'dayofweek', 'hour', 'holiday',  'hightemp', 'lowtemp', 'avgtemp','RH','wind','press','total_precip','last_year', 'real' ],
                dtype={'timeStamp': str, 'month': float, 'dayofmonth': float, 'dayofweek': float,'hour': float,  'holiday': float, 'hightemp': float,
                       'lowtemp': float, 'avgtemp': float,'RH': float,'wind': float,'press': float,'total_precip': float,'last_year': float, 'real': float},
                parse_dates=['timeStamp'], date_parser=dateparse)
June14.index = pd.DatetimeIndex(June14.timeStamp)
timeStamp=June14['timeStamp']
# =============================================================================
#input data
# =============================================================================
June14=June14.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
dane=June14.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
scaler = joblib.load("scaler.save") 
scaled = scaler.transform(June14)
June14_Y=scaled[:,13]
June14_X=scaled[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
size=June14_X.shape[1]
# =============================================================================

#predictions
# =============================================================================
model = load_model('lstm_model2.h5')
June14_X = np.reshape(June14_X, (-1, 1, size))

new_predictions= model.predict(June14_X, verbose=0)

prediction_reshaped = np.zeros((len(new_predictions), size+1))
June14_Y_reshaped = np.zeros((len(June14_Y), size+1))

prediction_r = np.reshape(new_predictions, (len(new_predictions),))
June14_Y = np.reshape(June14_Y, (len(June14_Y),))

prediction_reshaped[:,size] = prediction_r
June14_Y_reshaped[:,size] = June14_Y

prediction_inversed = scaler.inverse_transform(prediction_reshaped)[:,size]
June14_Y_inversed = scaler.inverse_transform(June14_Y_reshaped)[:,size]
# =============================================================================

#calculating error rates
# =============================================================================
rmse = sqrt(mean_squared_error(June14_Y_inversed, prediction_inversed))
maae=mean_absolute_error(June14_Y_inversed, prediction_inversed)
r2=r2_score(June14_Y_inversed,prediction_inversed) 
mape_err=mean(np.abs((June14_Y_inversed - prediction_inversed) / June14_Y_inversed)) * 100

#print error rates and R^2
# =============================================================================
print('MAPE: %.3f' % mape_err)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % maae)
print('R^2: %.3f' % r2)
# =============================================================================

#plot results
# =============================================================================
x_axis=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18','19','20','21','22','23']
plt.plot(prediction_inversed,'b', label='prediction')
plt.plot(x_axis, June14_Y_inversed,'lightsalmon', label='actual data')
plt.title('Predictions in comparison with actual data')
plt.xlabel('hour')
plt.ylabel('emergency calls' )
plt.legend( loc='upper right')
plt.show()
# =============================================================================