"""
911 Emergency Calls Forecasting Model
Hourly Predictions
@author: Monika Scislo
"""

from keras.callbacks import EarlyStopping
from math import sqrt
from keras.metrics import mse, mae, mape, msle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import h5py
from keras.layers import Dense, Activation, LSTM  
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")
from statistics import mean
np.random.seed(7)


#preparing weather data 
# =============================================================================
dateparse = lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p')
dateparse2 = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
weather_pottstown = pd.read_csv('weather_pottstown.csv', header=0 , names=['timeStamp','High Temp.','Low Temp.','Avg Temp.','Temp Departure','HDD','CDD','GDD','Avg Dewpoint','Avg RH','Avg Wind Speed','Avg Wind Dir','Avg Press','Total Precip','# obs'],
                                    dtype={'timeStamp':str,'High Temp.':int,'Low Temp.':int,'Avg Temp.':int,'Temp Departure':int,'HDD':int,'CDD':int,'GDD':int,'Avg Dewpoint':int,'Avg RH':int,'Avg Wind Speed':int,'Avg Wind Dir':int,'Avg Press':float,'Total Precip':str,'# obs':int},parse_dates=['timeStamp'], date_parser=dateparse2)
weather_pottstown=weather_pottstown.sort_values(by=['timeStamp'])
weather_pottstown = weather_pottstown[(weather_pottstown.timeStamp >= "2017-01-01 00:00:00")]
weather_pottstown = weather_pottstown[(weather_pottstown.timeStamp <= "2018-11-17 00:00:00")]
weather_pottstown.index = pd.DatetimeIndex(weather_pottstown.timeStamp)
weather_pottstown=weather_pottstown.drop(columns=['timeStamp'])
weather_pottstown.head()

# =========================================================================================
dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#preparing historical data from one year ago
# =============================================================================
past_data_last_year = pd.read_csv("911.csv",
                                    header=0, names=['lat', 'lng', 'desc', 'zip', 'title', 'timeStamp', 'twp', 'addr', 'e'],
                                    dtype={'lat': str, 'lng': str, 'desc': str, 'zip': str, 'title': str, 'timeStamp': str, 'twp': str,
                       'addr': str, 'e': int},
                                    parse_dates=['timeStamp'], date_parser=dateparse)



past_data_last_year = past_data_last_year[(past_data_last_year.timeStamp >= "2016-01-01 00:00:00")]
#removing samples from 29th Feb
past_data_last_year=past_data_last_year.drop(past_data_last_year.index[32065:32407])
past_data_last_year['timeStamp'] = past_data_last_year['timeStamp'] + pd.DateOffset(years=1)
past_data_last_year.index = pd.DatetimeIndex(past_data_last_year.timeStamp)

past_data_last_year.head()
# title is the category
past_data_last_year["title"].value_counts()
#type: EMS, Fire, Traffic
past_data_last_year['type'] = past_data_last_year["title"].apply(lambda x: x.split(':')[0])
past_data_last_year["type"].value_counts()
ref=past_data_last_year
past_data_last_year = pd.pivot_table(past_data_last_year, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
# Resampling by hour
past_data_last_year = past_data_last_year.resample('60min', how=[np.sum]).reset_index()
past_data_last_year.head()
past_data_last_year=past_data_last_year.sum(axis = 1, skipna = True)
past_data_last_year.head()

#FIRE
fire_ly = ref[ref['type'] == 'Fire']
fire_pivot_ly=pd.pivot_table(fire_ly, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
fire_sum_ly = fire_pivot_ly.resample('60min', how=[np.sum]).reset_index()
fire_all_ly=fire_sum_ly.sum(axis = 1, skipna = True)
#TRAFFIC
traffic_ly = ref[ref['type'] == 'Traffic']
traffic_pivot_ly=pd.pivot_table(traffic_ly, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
traffic_sum_ly = traffic_pivot_ly.resample('60min', how=[np.sum]).reset_index()
traffic_all_ly=traffic_sum_ly.sum(axis = 1, skipna = True)
#EMS
ems_ly=ref[ref['type'] == 'EMS']
ems_pivot_ly=pd.pivot_table(ems_ly, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
ems_sum_ly = ems_pivot_ly.resample('60min', how=[np.sum]).reset_index()
ems_all_ly=ems_sum_ly.sum(axis = 1, skipna = True)

# =============================================================================

# prepraing data from 2017-2018 for total number of emergency calls
# =============================================================================

d = pd.read_csv("911.csv",
                header=0, names=['lat', 'lng', 'desc', 'zip', 'title', 'timeStamp', 'twp', 'addr', 'e'],
                dtype={'lat': str, 'lng': str, 'desc': str, 'zip': str, 'title': str, 'timeStamp': str, 'twp': str,
                       'addr': str, 'e': int},
                parse_dates=['timeStamp'], date_parser=dateparse)

d.index = pd.DatetimeIndex(d.timeStamp)
d = d[(d.timeStamp >= "2017-01-01 00:00:00")]
d.head()

d["title"].value_counts()

d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()

p = pd.pivot_table(d, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
#resampling every hour
pp = p.resample('60min', how=[np.sum]).reset_index()


#preparing data from 2017-2018 for categories EMS, Fire, Traffic
d = pd.read_csv("911.csv",
                header=0, names=['lat', 'lng', 'desc', 'zip', 'title', 'timeStamp', 'twp', 'addr', 'e'],
                dtype={'lat': str, 'lng': str, 'desc': str, 'zip': str, 'title': str, 'timeStamp': str, 'twp': str,
                       'addr': str, 'e': int},
                parse_dates=['timeStamp'], date_parser=dateparse)


d.index = pd.DatetimeIndex(d.timeStamp)
d = d[(d.timeStamp >= "2017-01-01 00:00:00")]
d.head()
d["title"].value_counts()
d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()

#FIRE
fire = d[d['type'] == 'Fire']
fire_pivot=pd.pivot_table(fire, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
fire_sum = fire_pivot.resample('60min', how=[np.sum]).reset_index()
fire.columns = fire.columns.get_level_values(0)
fire_all=fire_sum.sum(axis = 1, skipna = True)

#TRAFFIC
traffic = d[d['type'] == 'Traffic']
traffic_pivot=pd.pivot_table(traffic, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
traffic_sum = traffic_pivot.resample('60min', how=[np.sum]).reset_index()
traffic_all=traffic_sum.sum(axis = 1, skipna = True)
#EMS
ems=d[d['type'] == 'EMS']
ems_pivot=pd.pivot_table(ems, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
ems_sum = ems_pivot.resample('60min', how=[np.sum]).reset_index()
ems_all=ems_sum.sum(axis = 1, skipna = True)
p = pd.pivot_table(d, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
#resampling every hour
pp = p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp.index
ppp.head()
pp.index = pd.DatetimeIndex(pp.timeStamp)
pp.columns = pp.columns.get_level_values(0)
pp.head()

# =============================================================================

#creating matrix with input data
# =============================================================================

input_data=pd.DataFrame(columns=['year', 'month', 'dayofyear','dayofmonth','dayofweek', 'week','hour','holiday','High Temp.','Low Temp.','Avg Temp.','Temp Departure','HDD','CDD','GDD','Avg Dewpoint','Avg RH','Avg Wind Speed','Avg Wind Dir','Avg Press','Total Precip','Traffic', 'Fire', 'EMS', 'Traffic last year', 'Fire last year', 'EMS last year','All emerg calls ly', 'All emerg calls'])

input_data.head()
input_data['year'] = pd.DatetimeIndex(pp['timeStamp']).year
input_data.head()
input_data['hour'] = pd.DatetimeIndex(pp['timeStamp']).hour
input_data.head()
input_data['month'] = pd.DatetimeIndex(pp['timeStamp']).month
input_data.head()
input_data['dayofyear'] = pd.DatetimeIndex(pp['timeStamp']).dayofyear
input_data.head()
input_data['dayofmonth'] = pd.DatetimeIndex(pp['timeStamp']).day
input_data.head()
input_data['dayofweek'] = pd.DatetimeIndex(pp['timeStamp']).dayofweek
input_data.head()
input_data['week'] = pd.DatetimeIndex(pp['timeStamp']).week
input_data.head()
input_data['hour'] = pd.DatetimeIndex(pp['timeStamp']).hour
input_data.head()
input_data['Traffic']=traffic_all
input_data['Fire']=fire_all
input_data['EMS']=ems_all
input_data['Fire last year']=fire_all_ly
input_data['EMS last year']=ems_all_ly
input_data['Traffic last year']=traffic_all_ly
input_data['All emerg calls']=ppp
input_data['All emerg calls ly']=past_data_last_year
#manually adding state holidays
input_data['holiday']=0
holiday=input_data['holiday']
input_data.index=pp.index
holiday.index=input_data.index
holiday['2017-01-01']=1
holiday['2018-01-01']=1
holiday['2018-05-28']=1
holiday['2018-05-29']=1
holiday['2018-07-04']=1
holiday['2017-07-04']=1
holiday['2018-09-03']=1
holiday['2017-09-04']=1
holiday['2018-11-12']=1
holiday['2018-11-13']=1
holiday['2017-11-23']=1
holiday['2017-12-25']=1
input_data['holiday']=holiday
input_data['High Temp.']=weather_pottstown.iloc[:,0]
input_data['Low Temp.']=weather_pottstown.iloc[:,1]
input_data['Avg Temp.']=weather_pottstown.iloc[:,2]
input_data['Temp Departure']=weather_pottstown.iloc[:,3]
input_data['HDD']=weather_pottstown.iloc[:,4]
input_data['CDD']=weather_pottstown.iloc[:,5]
input_data['GDD']=weather_pottstown.iloc[:,6]
input_data['Avg Dewpoint']=weather_pottstown.iloc[:,7]
input_data['Avg RH']=weather_pottstown.iloc[:,8]
input_data['Avg Wind Speed']=weather_pottstown.iloc[:,9]
input_data['Avg Wind Dir']=weather_pottstown.iloc[:,10]
input_data['Avg Press']=weather_pottstown.iloc[:,11]
#getting rid of NaN values from precitipation data
for x in range (0,weather_pottstown.shape[0]):
    if weather_pottstown.iloc[x,12]=='Trace':
            weather_pottstown.iloc[x,12]=0.001
    if pd.isna(weather_pottstown.iloc[x,12]):
            weather_pottstown.iloc[x,12]=0
input_data['Total Precip']=weather_pottstown.iloc[:,12]
input_data['Total Precip'] = input_data['Total Precip'].astype(float)
#assigning daily weather observations to every sample
for y in range (8,21):
    for x in range (0,input_data.shape[0]):
        if pd.isna(input_data.iloc[x,y]):
            input_data.iloc[x,y]=input_data.iloc[x-1,y]

#EMS
# =============================================================================
#dane=input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,26,23]]
#X= input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,26,23]]
# =============================================================================

#FIRE
# =============================================================================
#dane=input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,25,22]]
#X= input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,25,22]]
# =============================================================================

#TRAFFIC
# =============================================================================
#dane=input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,24,21]]
#X= input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,24,21]]
# =============================================================================

#All emergency calls 
# =============================================================================
dane=input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,27,28]]
X= input_data.iloc[:,[1,3,4,6,7,8,9,10,16,17,19,20,27,28]]

#scaling data
# =============================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
#saving scaler
scaler_filename = "scaler_hourly"
scaled = scaler.fit_transform(X)
X_scaled=scaled
size=X_scaled.shape[1]
Y=scaled[:,size-1]
X=scaled[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
size=X.shape[1]
# =============================================================================
#building model
# =============================================================================
#train-test split and reshaping to 3D
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, shuffle=True)
X_train = np.reshape(X_train, (-1, 1, size))
X_test = np.reshape(X_test, (-1, 1, size))

# NEURAL NETWORK
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(64,activation='relu'))
#model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
#model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adagrad')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
#training
history = model.fit(X_train, y_train, epochs=10, batch_size=15, validation_split=0.15, verbose=2, callbacks=[early_stopping])
#predictions for test data
predictions= model.predict(X_test)
# =============================================================================

plt.figure(0)
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


#reshaping and inversing normalization
# =============================================================================
prediction_reshaped = np.zeros((len(predictions), size+1))
testY_reshaped = np.zeros((len(y_test), size+1))

prediction_r = np.reshape(predictions, (len(predictions),))
testY_r = np.reshape(y_test, (len(y_test),))

prediction_reshaped[:,size] = prediction_r
testY_reshaped[:,size] = testY_r

prediction_inversed = scaler.inverse_transform(prediction_reshaped)[:,size]
testY_inversed = scaler.inverse_transform(testY_reshaped)[:,size]



#calculating error rates
# =============================================================================
rmse = sqrt(mean_squared_error(testY_inversed, prediction_inversed))
maae=mean_absolute_error(testY_inversed, prediction_inversed)
r2=r2_score(testY_inversed,prediction_inversed) 
#removing 0 values to calculate mape
#prediction_inversed[(np.where(testY_inversed==0))]='Nan'
#testY_inversed[(np.where(testY_inversed==0))]='Nan'
#mape_err=mean(np.abs((testY_inversed - prediction_inversed) / testY_inversed)) * 100
# =============================================================================

#plot loss and predictions vs. real values
# =============================================================================
plt.figure(0)
plt.subplot(1,2,1)
plt.plot(history.history['loss'],'green')
plt.plot(history.history['val_loss'],'r')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.subplot(1,2,2)
plt.plot(prediction_inversed[1250:1320],'b', label='prediction')
plt.plot(testY_inversed[1250:1320],'lightsalmon', label='actual data')
plt.title('Predictions in comparison with actual data')
plt.xlabel('samples')
plt.ylabel('emergency calls')
plt.legend( loc='upper right')
plt.show()

# =============================================================================

#plot results for 4 different sample sets 
# =============================================================================
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(prediction_inversed[0:70], label='prediction')
plt.plot(testY_inversed[0:70], label='actual data')
plt.title('Predictions in comparison with actual data', fontdict={'fontsize': 12, 'fontweight': 'medium'})
plt.xlabel('samples 0:70', fontsize=10)
plt.ylabel('emergency calls', fontsize=10)
plt.subplot(2,2,2)
plt.plot(prediction_inversed[1250:1320], label='prediction')
plt.plot(testY_inversed[1250:1320], label='actual data')
plt.title('Predictions in comparison with actual data', fontdict={'fontsize': 12, 'fontweight': 'medium'})
plt.xlabel('samples 1250:1320', fontsize=10)
plt.ylabel('emergency calls', fontsize=10)
plt.subplot(2,2,3)
plt.plot(prediction_inversed[2550:2620], label='prediction')
plt.plot(testY_inversed[2550:2620], label='actual data')
plt.title('Predictions in comparison with actual data', fontdict={'fontsize': 11, 'fontweight': 'medium'})
plt.xlabel('samples 2550:2620',fontsize=10)
plt.ylabel('emergency calls', fontsize=10)
plt.subplot(2,2,4)
plt.plot(prediction_inversed[4000:4070], label='prediction')
plt.plot(testY_inversed[4000:4070], label='actual data')
plt.title('Predictions in comparison with actual data', fontdict={'fontsize': 11, 'fontweight': 'medium'})
plt.xlabel('samples 4000:4070', fontsize=10)
plt.ylabel('emergency calls', fontsize=10)
plt.legend(loc='upper left')
plt.show()

# =============================================================================


#get model details
# =============================================================================
model.output_shape 
model.summary()
model.get_config()
model.get_weights() 
#saving model
model.save('hourly_model.h5')
# =============================================================================

#print error rates and R^2
# =============================================================================
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % maae)
print('R^2: %.3f' % r2)
# =============================================================================