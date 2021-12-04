import requests
import numpy as np
from DataPreprocessing import Preprocessing

DATA = input("Enter Location of Data\n")

URL = "http://127.0.0.1:5000/predict"

Params = {"Content-Type": "application/json"}

dataframe = Preprocessing(DATA)

#dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)
#print(dataframe)
#print(np.array(dataframe[0:1]).tolist()[0])
data = {"input" : np.array(dataframe).tolist()}

r = requests.get(URL, headers = Params, json = data)

print(r.json())