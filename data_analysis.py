"""
911 Emergency Calls Forecasting Model
Data Analysis 
@author: Monika Scislo
"""
import pandas as pd
import numpy as np
import datetime
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
import calendar

sns.set(style="white", color_codes=True)
dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')


#2016
##########

###############################################                   
###########     DAILY        ##################
###############################################
# Read data 
d=pd.read_csv("911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
d=d[(d.timeStamp <= "2016-12-31 00:00:00")]

d.head()
d["title"].value_counts()
d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()
g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)
pp.head()

fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 
ax.plot(pp['timeStamp'], ppp,'g',label='EMS')
ax.set_title("EMS")

plt.show()
###################################################################

g=d[d['type'] == 'Traffic' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)
pp.head()
ax.plot(pp['timeStamp'], ppp,'b',label='Traffic')
ax.set_title("Traffic")

plt.show()
###################################################

g=d[d['type'] == 'Fire' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)
ax.plot(pp['timeStamp'], ppp,'r',label='Fire')
ax.set_title("Daily 911 calls in 2016")

plt.show()
plt.legend(loc='upper right')

###############################################                   
###########     HOURLY        ##################
###############################################
# Read data 
d=pd.read_csv("911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-05-01 00:00:00")]
d=d[(d.timeStamp <= "2016-05-08 00:00:00")]

d.head()
d["title"].value_counts()
d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()
g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)
pp.head()

fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  

ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot(pp['timeStamp'], ppp,'g',label='EMS')

ax.set_title("EMS")

plt.show()
###################################################################

g=d[d['type'] == 'Traffic' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)
pp.head()
ax.plot(pp['timeStamp'], ppp,'b',label='Traffic')
ax.set_title("Traffic")
fig.autofmt_xdate()
plt.show()
###################################################

g=d[d['type'] == 'Fire' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)


ax.plot(pp['timeStamp'], ppp,'r',label='Fire')

ax.set_title("Hourly 911 calls in the first week of May 2016")

plt.show()
plt.legend(loc='upper right')

#2017
##########
###############################################                   
###########     DAILY        ##################
###############################################
# Read data 
d=pd.read_csv("911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2017-01-01 00:00:00")]
d=d[(d.timeStamp <= "2018-01-01 00:00:00")]

d.head()
d["title"].value_counts()
d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()
dd=d

g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
mean_ems_2017=mean(ppp)

pp.columns = pp.columns.get_level_values(0)

pp.head()

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot(pp['timeStamp'], ppp,'g',label='EMS')


ax.set_title("EMS")

plt.show()
###################################################################

g=d[d['type'] == 'Traffic' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
mean_traffic_2017=mean(ppp)
pp.columns = pp.columns.get_level_values(0)
ax.plot(pp['timeStamp'], ppp,'b',label='Traffic')
ax.set_title("Traffic")

plt.show()
###################################################

g=d[d['type'] == 'Fire' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
mean_fire_2017=mean(ppp)
pp.columns = pp.columns.get_level_values(0)
ax.plot(pp['timeStamp'], ppp,'r',label='Fire')

plt.xlabel('Day')
plt.ylabel('Emergency calls')
ax.set_title("Daily 911 calls in 2017")
fig.autofmt_xdate()
plt.show()
plt.legend(loc='upper right')

###############################################                   
###########     HOURLY        ##################
###############################################
# Read data 
d=pd.read_csv("911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2017-05-01 00:00:00")]
d=d[(d.timeStamp <= "2017-05-08 00:00:00")]

d.head()
d["title"].value_counts()

d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()

g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()

pp.columns = pp.columns.get_level_values(0)

pp.head()

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot(pp['timeStamp'], ppp,'g',label='EMS')


ax.set_title("EMS")

plt.show()
###################################################################

g=d[d['type'] == 'Traffic' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()

pp.columns = pp.columns.get_level_values(0)

pp.head()


ax.plot(pp['timeStamp'], ppp,'b',label='Traffic')



ax.set_title("Traffic")
fig.autofmt_xdate()
plt.show()
###################################################

g=d[d['type'] == 'Fire' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()

pp.columns = pp.columns.get_level_values(0)



ax.plot(pp['timeStamp'], ppp,'r',label='Fire')



ax.set_title("Hourly 911 calls in the first week of May 2017")

plt.show()
plt.legend(loc='upper right')

#######
##########2018
###############################################                   
###########     DAILY        ##################
###############################################
# Read data 
d=pd.read_csv("911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)

d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2018-01-01 00:00:00")]
d=d[(d.timeStamp <= "2018-11-16 00:00:00")]

d.head()
d["title"].value_counts()
d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()
g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
mean_ems_2018=mean(ppp)
pp.columns = pp.columns.get_level_values(0)

pp.head()

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot(pp['timeStamp'], ppp,'g',label='EMS')



ax.set_title("EMS")

plt.show()
###################################################################

g=d[d['type'] == 'Traffic' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
mean_traffic_2018=mean(ppp)
pp.columns = pp.columns.get_level_values(0)

pp.head()

ax.plot(pp['timeStamp'], ppp,'b',label='Traffic')


ax.set_title("Traffic")

plt.show()
###################################################

g=d[d['type'] == 'Fire' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('D', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
mean_fire_2018=mean(ppp)
pp.columns = pp.columns.get_level_values(0)



ax.plot(pp['timeStamp'], ppp,'r',label='Fire')
plt.xlabel('Day')
plt.ylabel('Emergency calls')
ax.set_title("Daily 911 calls in 2018")
fig.autofmt_xdate()
plt.show()
plt.legend(loc='upper right')

###############################################                   
###########     HOURLY        ##################
###############################################
# Read data 
d=pd.read_csv("911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2018-05-01 00:00:00")]
d=d[(d.timeStamp <= "2018-05-08 00:00:00")]

d.head()
d["title"].value_counts()

d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d["type"].value_counts()
g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)

pp.head()
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot(pp['timeStamp'], ppp,'g',label='EMS')


ax.set_title("EMS")

plt.show()
###################################################################

g=d[d['type'] == 'Traffic' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling every week 'W'.  This is very powerful
pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)

pp.head()

ax.plot(pp['timeStamp'], ppp,'b',label='Traffic')

ax.set_title("Traffic")
fig.autofmt_xdate()
plt.show()
###################################################

g=d[d['type'] == 'Fire' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

pp=p.resample('60min', how=[np.sum]).reset_index()
pp.head()
ppp=pp.sum(axis = 1, skipna = True)
ppp.index=pp['timeStamp']
ppp.head()
pp.columns = pp.columns.get_level_values(0)



ax.plot(pp['timeStamp'], ppp,'r',label='Fire')



ax.set_title("Hourly 911 calls in the first week of May 2018")

plt.show()
plt.legend(loc='upper right')

#heatmap hour
plt.figure(10)
g = dd
g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))
g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))
p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)
p.head()

cmap = sns.cubehelix_palette(light=5, as_cmap=True)
ax = sns.heatmap(p, cmap='Reds')
ax.set_title('Emergency calls')

#heatmap day
plt.figure(11)
g = dd
g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))
g['Day'] = g['timeStamp'].apply(lambda x: x.strftime('%d'))
p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Day'], aggfunc=np.sum)
p.head()
#sns.set_palette("PuBuGn_d")
cmap = sns.cubehelix_palette(light=5, as_cmap=True)
ax = sns.heatmap(p, cmap='BuGn')
ax.set_title('Emergency calls')

##################################
df = pd.read_csv('911.csv')
df.head()
df.index = pd.DatetimeIndex(df.timeStamp)
df=df[(df.timeStamp >= "2016-01-01 00:00:00")]
df=df[(df.timeStamp <= "2018-12-31 00:00:00")]
reason = np.unique(df['title'])
DATA = np.zeros((df.shape[0],6),dtype='O')
DATA[:,0] = df['lng'].values
DATA[:,1] = df['lat'].values
DATA[:,4] = df['title'].values
DATA[:,5] = df['twp'].values
for i in range(DATA.shape[0]):
    DATA[i,2] = df['timeStamp'].values[i][:10]
    DATA[i,3] = df['timeStamp'].values[i][10:]
    sp = DATA[i,3].split(':')
    DATA[i,3] = (int(sp[0])*3600 + int(sp[1])*60 + int(sp[2]))/3600
    
new_data = np.zeros(reason.size,dtype = 'O')
for i in range(reason.size):
    new_data[i] = DATA[np.where(DATA[:,4] == reason[i])]

week = np.array(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
for i in range(new_data.shape[0]):
    for j in range(new_data[i].shape[0]):
        w = np.array(new_data[i][j,2].split('-')).astype(int)
        new_data[i][j,0] = week[calendar.weekday(w[0],w[1],w[2])]

for i in range(DATA.shape[0]):
    DATA[i,2] = DATA[i,2][:-3]
for i in range(reason.size):
    new_data[i] = DATA[np.where(DATA[:,4] == reason[i])]

all_ = np.zeros(df["timeStamp"].values.size,dtype='O')

for i in range(all_.size):
    all_[i] = df['timeStamp'].values[i][:10] 
for i in range(all_.size):
    w = np.array(all_[i].split('-')).astype(int)
    all_[i] = week[calendar.weekday(w[0],w[1],w[2])]
plt.figure(figsize=(12,4))
plt.xlabel("Day of the week")
plt.title("Total number of emergency calls by day of the week")

sns.countplot(all_,order = week, palette="PuRd")
plt.ylabel("Emergency calls")
labels = "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
sizes = [np.sum(all_ == "Monday"),np.sum(all_ == "Tuesday"),np.sum(all_ == "Wednesday"),np.sum(all_ == "Thursday"),np.sum(all_ == "Friday"),\
         np.sum(all_ == "Saturday"),np.sum(all_ == "Sunday")]

all_ = np.zeros(df["timeStamp"].values.size,dtype='O')
for i in range(all_.size):
    h = np.array(df['timeStamp'].values[i][11:].split(":")).astype(int)
    all_[i] = (h[0] * 3600 + h[1] * 60 + h[2])/3600
all_ = all_.astype(int)
plt.figure(figsize=(12,4))
plt.title("Total number of emergency calls by hour")
sns.countplot(all_, palette="BuPu")
plt.xlabel("Hour")
plt.ylabel("Emergency calls")