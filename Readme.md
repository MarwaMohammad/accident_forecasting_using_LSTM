# Project Name: Munich Accidents' count Forecasting.

## Project Overview:
Forecasting the accidents counts is an important topic to prepare aids, ambulances for victims. Also, it helps to expect the hospitals preparations.

In this project, the datasets from [Munich Open Data Portal] (https://opendata.muenchen.de/dataset/monatszahlen-verkehrsunfaelle/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7) 

During my work on this project, I had the following goals:
- Visualizing data and exploring some insights using the available dataset. 
  I did this through the All_Data_Analysis_Visualization.ipynb notebook using pandas, matplotlib.

- Modeling the data to be able to expect the count of each type and category of accidents. 
  I used LSTM and Keras to achieve this through the Insgesamt_Accidents_Analysis_and_Modeling.ipynb notebook.
  
- Using the model to predict accident counts. This is done through Inference.ipynb notebook.

- Deploying the model using the FastAPI locally to predict the accidents counts through the prediction page. I did this using the uvicorn, fastapi through app.py file.

- Deploying the model on google cloud and using it from the cloud in the prediction. I did this in the gapp.py file.

- Containerizing both app.py and gapp.py

- Documenting all work, ideas, and steps to reproduce the output in the Accident Number Forecasting.pdf

## Installation Instructions:
Please, read the Accident Number Forecasting.pdf to replicate the results.

## Contact info:
- Name: Marwa Matar
- Email: marwa.mohammad.matar@gmail.com
