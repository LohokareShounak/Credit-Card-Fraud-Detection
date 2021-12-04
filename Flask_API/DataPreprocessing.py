import pandas as pd
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt

def Preprocessing(DATA):
    data = pd.read_csv(DATA)
    data = data[0:20]
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['hour'] = data['trans_date_trans_time'].dt.hour

    data['day_of_week'] = data['trans_date_trans_time'].dt.day_name()

    data['year_month'] = data['trans_date_trans_time'].dt.to_period('M')

    data['dob'] = pd.to_datetime(data['dob'])
    data['age'] = np.round((data['trans_date_trans_time'] - data['dob'])/np.timedelta64(1,'Y'))

    def distance(lat1, lat2, lon1, lon2):
        
        lon1 = radians(lon1)
        lon2 = radians(lon2)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        # calculate the result
        return(c * r)

    data['dist'] = data.apply(lambda x: distance(x.merch_lat, x.lat, x.merch_long, x.long), axis=1)

    # Calculating the average of fraud happened with one customer

    Fraud_avg = {}
    for i in data['cc_num'].unique():
        Fraud_avg[i] = len(data[(data['cc_num'] == i) & data['is_fraud'] == 1]) / len(data[data["cc_num"] == i])

    data['Fraud_avg'] = data['cc_num'].map(Fraud_avg)

    #Calculating average amt of fraud and non-fraud cases for specific person

    Amt_Avg_Fraud = {}
    Amt_Avg_Legit = {}

    for i in data['cc_num'].unique():
        Amt_Avg_Fraud[i] = data[(data['cc_num'] == i) & (data['is_fraud'] == 1)]['amt'].mean()
        Amt_Avg_Legit[i] = data[(data['cc_num'] == i) & (data['is_fraud'] == 0)]['amt'].mean()

    data['Amt_Avg_Fraud'] = data['cc_num'].map(Amt_Avg_Fraud)
    data['Amt_Avg_Legit'] = data['cc_num'].map(Amt_Avg_Legit)

    data = data.replace(np.NaN,0)

    #Normalising the amount

    data['amt_normalised'] = data['amt'] / max(data['amt'])


    #Loadind Saved Encodings
    with open('Dicts\Cities.pkl', 'rb') as f:
        cities = pickle.load(f)

    with open('Dicts\States.pkl', 'rb') as f:
        states = pickle.load(f)
        
    with open('Dicts\Merchants.pkl', 'rb') as f:
        merchants = pickle.load(f)

    with open('Dicts\Jobs.pkl', 'rb') as f:
        jobs = pickle.load(f)
        
    with open('Dicts\Days.pkl', 'rb') as f:
        days = pickle.load(f)

    with open('Dicts\Category.pkl', 'rb') as f:
        Category = pickle.load(f)
    data['city'] = data.city.map(cities)
    data['state'] = data.state.map(states)

    data['city'] = data['city'] / len(cities)
    data['state'] = data['state'] / len(states)

    data.merchant = data.merchant.map(merchants) / len(merchants)

    data.day_of_week = data.day_of_week.map(days)

    data.job = data.job.map(jobs)

    data['category'] = data.category.map(Category)

    data['year_month'] = data['year_month'].astype('str')

    data_train_col = ['hour','state','merchant','age','day_of_week','job','category','amt_normalised','dist','Fraud_avg','Amt_Avg_Legit','Amt_Avg_Fraud']

    data = data[data_train_col]

    return data
