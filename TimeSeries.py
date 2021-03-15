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

def evaluate_multistep_model(model,X,y,scaler,difference_data,step_out,ts_part=0):
    predict = model.predict(X)
    predict = scaler.inverse_transform(predict)
    Y = scaler.inverse_transform(y)

    predict=np.add(predict,difference_data.values)
    Y=np.add(Y,difference_data.values)

    errors=list()

    for index in range(0,step_out):
        errors.append(math.sqrt(mean_squared_error(Y[:,index], predict[:,index])))

    return errors

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

def plot_multistep_result(model,X,y,scaler,difference_data,skip=100,step_out=5):
    predict = model.predict(X)
    predict = scaler.inverse_transform(predict)
    Y = scaler.inverse_transform(y)
    predict=np.add(predict,difference_data.values)
    Y=np.add(Y,difference_data.values)

    skip_space=skip*step_out

    plt.figure(figsize=(20,8))

    offset=0
    for index in range(skip_space,predict.shape[0],step_out):
        start=offset
        offset+=step_out
        end=offset
        if(end+skip_space<predict.shape[0]):
            x_axis=[x for x in range(start,end+step_out)]
            values=np.concatenate((Y[start+skip_space:end+skip_space,0],predict[index]))
   
        plt.plot(x_axis, values, color='red')
    

    plt.plot(Y[skip_space:,0],color='blue')
    plt.show()