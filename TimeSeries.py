import pandas as pd 
import numpy as np
import math

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 



def series_to_supervised(ts_dataframe, steps_in=1, steps_out=1, dropnan=True):
    var_names=ts_dataframe.columns
    n_vars=ts_dataframe.shape[1]
    cols,names=list(),list()

    for index in range(steps_in,0,-1):
        cols.append(ts_dataframe.shift(index))
        names+=(f'{name}(t-{index})' for name in var_names)

    for index in range(0,steps_out):
        cols.append(ts_dataframe.shift(-index))
        if(index==0):
            names+=(f'{name}(t)' for name in var_names)
        if(index>0):
            names+=(f'{name}(t+{index})' for name in var_names)
    agg=pd.concat(cols,axis=1)
    agg.columns=names

    if(dropnan):
        agg.dropna(inplace=True)
    return agg

def create_differencing_reverse(ts_dataframe,steps_in=1, steps_out=1, dropnan=True):
    name=ts_dataframe.columns[0]
    cols,names=list(),list()
    var_names=ts_dataframe.columns

    for index in range(0,steps_out):
        cols.append(ts_dataframe.shift(-index)[steps_in:])
        if(index==0):
            names+=(f'{name}(t)' for name in var_names)
        if(index>0):
            names+=(f'{name}(t+{index})' for name in var_names)

    agg=pd.concat(cols,axis=1)
    agg.columns=names

    if(dropnan):
        agg.dropna(inplace=True)
    return agg[:-1]

def differencing(ts_dataframe,column_names,steps=1):

    for name in column_names:
        ts_dataframe[name+'_diff']=(ts_dataframe[name]-ts_dataframe[name].shift(steps))

def evaluate_model(model,X,y,scaler,difference_data):
    predict = model.predict(X)
    predict = scaler.inverse_transform(predict)
    Y = scaler.inverse_transform(y)

    predict=np.add(predict,difference_data.values)
    Y=np.add(Y,difference_data.values)

    return math.sqrt(mean_squared_error(Y, predict))

def plot_result(model,X,y,scaler,difference_data,ts_part=0):
    predict = model.predict(X)
    predict = scaler.inverse_transform(predict)
    Y = scaler.inverse_transform(y)

    predict=np.add(predict,difference_data.values)
    Y=np.add(Y,difference_data.values)

    plt.figure( figsize=(20,8))
    plt.plot(predict[ts_part:])
    plt.plot(Y[ts_part:])
    plt.legend(["Predicted vales", "True values"], loc='upper right')
    plt.show()