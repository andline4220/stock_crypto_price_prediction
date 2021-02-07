import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
data=pd.read_csv('BTC-USD.csv')
data.head()

high_prices=data['High'].values
low_prices=data['Low'].values
mid_frices=(high_prices+low_prices)/2

seq_len=50
sequence_length=seq_len+1

result=[]
for index in range(len(mid_frices)-sequence_length):
    result.append(mid_frices[index:index + sequence_length])

normalized_data=[]
for window in result:
    normalized_window=[((float(p)/float(window[0]))-1)for p in window]
    normalized_data.append(normalized_window)

result=np.array(normalized_data)

row=int(round(result.shape[0]*0.9)) 
train=result[:row,:]
np.random.shuffle(train) 

x_train=train[:,:-1]
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
y_train=train[:,:-1]  

x_test=result[row:,:-1]
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
y_test=result[row:,-1]

x_train.shape,x_test.shape

fig=plt.figure(facecolor='white')
ax=fig.add_subplot(111)
ax.plot(y_test,label='True')
#ax.plot(pred,label='Prediction')
ax.legend()
plt.show()






