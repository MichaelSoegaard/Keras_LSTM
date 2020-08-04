#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GRU, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
import pickle
import time


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


BATCH_SIZE = 256
SEQ_LEN = 12
EPOCHS = 10
NAME = "LSTM_test2"
LOG_DIR = f"tuner\\{int(time.time())}"

tensorboard = TensorBoard(log_dir=f"logs\\{NAME}")


data = pd.read_csv('data\\WORK_Value_sample20_GBPJPY_2016_2020_5min_ATRmv_UNBALANCED_2905.csv')



data.drop(['Unnamed: 0'], inplace=True, axis=1)

split_len = int(len(data)*0.8)
print(split_len)
train_data = data.iloc[:split_len,:]
test_data = data.iloc[split_len:,:]


scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(train_data.iloc[:,:-1])
X_test_scaled = scaler.transform(test_data.iloc[:,:-1])
train_scaled = np.concatenate([X_train_scaled, np.array(train_data.iloc[:,-1:])], axis=1)
test_scaled = np.concatenate([X_test_scaled, np.array(test_data.iloc[:,-1:])], axis=1)



# In[8]:


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



def create_model(hp):
    
    model = Sequential()
    
    model.add(LSTM(hp.Int('input_units',
                                min_value=32,
                                max_value=192,
                                step=32), input_shape=(X_train.shape[1:]), return_sequences=True, stateful=False))
    model.add(Dropout(0.2))
       
           
   
    #for i in range(hp.Int('n_layers', 0, 3)):
    model.add(LSTM(hp.Int(f'GRU_1',
                                min_value=32,
                                max_value=256,
                                step=32), return_sequences=True, stateful=False))
    model.add(Dropout(0.2))

    model.add(LSTM(hp.Int(f'GRU_2',
                                min_value=32,
                                max_value=256,
                                step=32), return_sequences=True, stateful=False))
    model.add(Dropout(0.2))
        

    model.add(LSTM(hp.Int('GRU_Last',
                                min_value=32,
                                max_value=256,
                                step=32), return_sequences=False, stateful=False))
    model.add(BatchNormalization())
           
    model.add(Dense(hp.Int('Dense_units',
                                min_value=32,
                                max_value=256,
                                step=32), activation='relu'))

    model.add(Dropout(0.2)) #(hp.Int('Dropout', 2, 5, 1))/10)
           
    model.add(Dense(1, activation='linear'))
       
       #Model compile settings:
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
           
       # Compile model
    model.compile(loss='mse',
               optimizer=opt
               )

    return model


# In[14]:


tuner = RandomSearch(
    create_model,
    objective='val_loss',
    max_trials=20,  # how many variations on model?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=os.path.normpath('D:/'))

tuner.search_space_summary()

tuner.search(x=X_train,
             y=y_train,
             epochs=50,
             batch_size=256,
             callbacks=[tensorboard],
             validation_data=(X_test, y_test))

tuner.results_summary()


with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


# In[ ]:

'''
#model.fit(X_train, y_train, Epochs=10, validation_data = (X_test, y_test))
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard]
)


# In[ ]:

'''


