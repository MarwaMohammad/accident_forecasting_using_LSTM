import pandas as pd
import datetime
import numpy as np
import gcsfs


#fastapi imports
from fastapi import FastAPI
from pydantic import BaseModel, validator, validate_arguments

# For loading model
import tensorflow as tf
import pickle

Time_Step = 12
########

PROJECT_NAME = 'ForecastingAccidentsApp'
CREDENTIALS = 'google_account_service'
MODEL_PATH = 'gs://forecast123/best_model.h5'
import h5py

FS = gcsfs.GCSFileSystem(project=PROJECT_NAME, token=CREDENTIALS)
with FS.open(MODEL_PATH, 'rb') as model_file:
    model_gcs = h5py.File(model_file, 'r')
    model = tf.keras.models.load_model(model_gcs)


app = FastAPI()
#model = load_model('best_model.h5')
df = pd.read_csv('./data/Alko_Insg.csv', index_col='date')

#Pickle scaler for features and value loading
# Open the file in binary mode
with open('./scalers/f_transformer.pkl', 'rb') as file:
    # Call load method to deserialze
    f_transformer = pickle.load(file)

# Open the file in binary mode
with open('./scalers/cnt_transformer.pkl', 'rb') as file:
    # Call load method to deserialze
    cnt_transformer = pickle.load(file)

# prepare sequences
def creat_ds(x, y, time_step=1):
    # Create x series and y series to hold sequences
    xs, ys = [], []

    for i in range(len(x) - time_step):
        # Extract the sequence
        v = x.iloc[i: (i + time_step)].to_numpy()
        # append it into x series
        xs.append(v)

        # Repeat all above for y series
        ys.append(y.iloc[i + time_step])
    return np.array(xs), np.array(ys)

@app.get('/')
def root():
    return {"message": "Forcast Munich Accidents value in category 'Alkoholunfälle' and 'insgesamt' accident type from 2001 to 1st month of 2021"}
00
from typing import Dict

class input_item(BaseModel):
    year: int
    month: int

    @validator('year')
    def year_must_be_2010_or_less(cls, value):
        """"
        Check the value of year and month, if they are outside this range the model can't predict without 12
        previous steps.
        """
        if value not in ([2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009
            ,2008,2007,2006,2005,2004,2003,2002,2001]):

            raise ValueError("We expect the year from 2021 to 2001, but you entered ", value)
        return value

    @validator('month')
    def month_from1_to12(cls, v):
        if v not in [1,2,3,4,5,6,7,8,9,10,11,12]:
            raise ValueError("We expect the month from 1 to 12, but you entered {value}", v)

        else:
            pass

        return v
    '''
    @validator('*')
    def check_sum(cls, v):
        if sum(v) > 2022:
            raise ValueError("There is no historical data this year and this month, sorry we can't forecast")
        return v
    '''

@app.post('/prediction')
def predict(input_feat: input_item):

    # Obtaining data recieved from the user to be used in the prediction
    input_feat = input_feat.dict()

    year = input_feat['year']
    month = input_feat['month']

    #Preparing the sequence for prediction
    input_df = pd.DataFrame()
    #convert the year, month into date to be easier in obtaining sequence (12 timestep)
    input_date = datetime.datetime(int(year), int(month), 1)
    df.index = pd.to_datetime(df.index)
    input_df = df[df.index < input_date].tail(12)
    input_df.loc[len(input_df.index)] = [year, month, 0]
    input_df.rename(index={12: input_date}, inplace=True)

    # Scaling features
    f_columns = ['year', 'month']
    input_df.loc[:, f_columns] = f_transformer.transform(input_df[f_columns].to_numpy())
    input_df['value'] = cnt_transformer.transform(input_df[['value']])

    #converting the dataframe into sequence formate
    xs_inf, ys_inf = creat_ds(input_df, input_df.value, time_step=Time_Step)

    #Predict

    y_predict_inf = model.predict(xs_inf)

    #convert the scaled value again into actual value
    y_pred_inv = cnt_transformer.inverse_transform(y_predict_inf)
    predicted_Value = int(np.ceil(y_pred_inv[0][0]))
    #print('Pridicted Value ', predicted_Value)

    real_value = None
    #check for real value availability
    if df[df.index == input_date].shape[0] != 0:
        real_value = int(df[df.index == input_date].value[0])
        #print('Real value ', real_value)

    #return {'prediction_value': predicted_Value, 'real_value':real_value}
    return {'prediction_value': predicted_Value}


# https://medium.com/analytics-vidhya/how-to-load-keras-h5-model-format-from-google-cloud-bucket-abf9a77d3cb4