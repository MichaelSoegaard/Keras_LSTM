#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import joblib
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GRU, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
import pickle
import time
from datetime import datetime


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.config.experimental.list_physical_devices()

SYMBOL = 'GBPJPY'
BATCH_SIZE = 64
SEQ_LEN = 6
EPOCHS = 1
NAME = f"UNBAL_LSTM_REG_L64_L256_L192_L96_D240_D02_LR0_0001_{time.time()}"
LOG_DIR = f"tuner\\{int(time.time())}"
DO = 0.2
TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

tensorboard = TensorBoard(log_dir=f"logs\\{NAME}")

filepath=f"models\\{NAME}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
scaler_filename = f'scaler\\RNN_{SYMBOL}_{TIME}.save'


# In[3]:


data = pd.read_csv('data\\WORK_Value_sample20_GBPJPY_2015_2019_Smart_Diff_V3_outliers.csv')


data.drop(['Unnamed: 0'], inplace=True, axis=1)
data = data.iloc[:1128787,:]


#data.drop(['Close_diff', 'open_close_diff', 'high_low_diff'], inplace=True, axis=1)

split_len = int(len(data)*0.8)
train_data = data.iloc[:split_len,:]
test_data = data.iloc[split_len:,:]


scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(train_data.iloc[:,:-1])
X_test_scaled = scaler.transform(test_data.iloc[:,:-1])
train_scaled = np.concatenate([X_train_scaled, np.array(train_data.iloc[:,-1:])], axis=1)
test_scaled = np.concatenate([X_test_scaled, np.array(test_data.iloc[:,-1:])], axis=1)

joblib.dump(scaler, scaler_filename)



def make_stack(df):
    sequential_data = np.empty(shape=[df.shape[0]-SEQ_LEN+1,SEQ_LEN,df.shape[1]])  # this is a array that will CONTAIN the sequences

    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
    j = 0
    for row in df:  # iterate over the values
        prev_days.append(row)
        if len(prev_days) == SEQ_LEN:  # make sure we have 12 sequences!
            #test = np.expand_dims(np.stack(prev_days, axis=1).T, axis=0)
            #print(test.shape)
            #break
            sequential_data[j] = np.expand_dims(np.stack(prev_days, axis=1).T, axis=0)  # append those bad boys!
            j += 1

    np.random.shuffle(sequential_data)  # shuffle for good measure.
    X = sequential_data[:,:,:-1]
    y = sequential_data[:,11:12,-1]
    return X, y


# In[9]:


X_train, y_train = make_stack(train_scaled)
X_test, y_test = make_stack(test_scaled)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def create_model():
    
    model = Sequential()
    
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), return_sequences=True, stateful=False))
    model.add(Dropout(DO))
    
    model.add(LSTM(256, return_sequences=True, stateful=False))
    model.add(Dropout(DO))

    model.add(LSTM(192, return_sequences=True, stateful=False))
    model.add(Dropout(DO))
    
    model.add(LSTM(96, return_sequences=False, stateful=False))
    model.add(Dropout(DO))
           
    model.add(Dense(240, activation='relu'))
    
    model.add(BatchNormalization())
           
    model.add(Dense(1))
       
       #Model compile settings:
    opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
           
       # Compile model
    model.compile(loss='mse',
               optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()]
               ) #sparse_categorical_crossentropy

    return model


#model.fit(X_train, y_train, Epochs=10, validation_data = (X_test, y_test))

model = create_model()
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)#,
    #callbacks=[tensorboard, checkpoint]
)

model.save(f"models\\test_RNN_Regression_epoch{EPOCHS}_{NAME}_.h5")

model.predict(X_test)
# In[ ]:




