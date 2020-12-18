# import required packages
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import datetime

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
    data=pd.read_csv("/Users/eberechukwukathomas/Desktop/Thomas_20868189/data/q2_dataset.csv")
    data['Date']=pd.to_datetime(data['Date'],infer_datetime_format=True)
    data=data.drop(['Date'],axis=1)
    data=data.reindex(columns=[' Open',' Close/Last',' Volume', ' High', ' Low'])
    data=data.replace('\$', '', regex=True).astype(float)
    data=data.replace('\$', '', regex=True).astype(float)
    data.isna().sum()
    sc = MinMaxScaler(feature_range=(0,1))
    scaled_data= sc.fit_transform(data)
    X=[]
    y=[]

    for i in range(3, 1259):
        X.append(scaled_data[i-3:i,0])
        y.append(scaled_data[i,0])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)


	# 1. load your training data
    X_train, y_train=np.array(X_train), np.array(y_train)
    X_test,y_test=np.array(X_test), np.array(y_test)
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    X_train=pd.DataFrame(X_train)
    y_train=pd.DataFrame(y_train)
    X_test=pd.DataFrame(X_test)
    y_test=pd.DataFrame(y_test)
    frames=[X_train,y_train]
    frameess=[X_test, y_test]
    result=pd.concat(frames, keys=['X_train', 'y_train'])
    resultsss=pd.concat(frameess,keys=['X_test', 'y_test'])
    result.to_csv('train_data_RNN.csv')
    resultsss.to_csv('test_data_RNN.csv')
    data1=pd.read_csv('train_data_RNN.csv')
    X_train=data1[0:879:]
    X_train=X_train.drop(['Unnamed: 0','Unnamed: 1'], axis=1)
    y_train=data1[879:1759:]
    y_train=y_train.drop(['Unnamed: 0','1','2','Unnamed: 1'], axis=1)
    X_train, y_train=np.array(X_train), np.array(y_train)
    X_train=np.reshape(X_train, (X_train.shape[0],
    X_train.shape[1],1))
	# 2. Train your network
    model = Sequential()
    model.add(LSTM(units=512,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=50,batch_size=70)
    sc=sc.scale_
    print(sc)
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model
    model.save('20868189_RNN_model')
    
    
    