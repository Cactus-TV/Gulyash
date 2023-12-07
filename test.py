import geopy.distance
import pandas as pd
import os
from os.path import join
import joblib
import numpy as np
import matplotlib.pyplot as plt


def preprocessing(data, train=False):
    data.loc[data['bedrooms'] == 'two', 'bedrooms'] = 2
    data.loc[data['bedrooms'] == 'one', 'bedrooms'] = 1
    data.loc[data['bedrooms'] == 'three', 'bedrooms'] = 3
    data['bedrooms'] = data['bedrooms'].fillna(0)
    data['bedrooms'] = data['bedrooms'].astype(float).astype(int)

    data['beds'] = data['beds'].fillna(0).astype(int)

    data['host_listings_count'] = data['host_listings_count'].fillna(0).astype(int)
    data['host_total_listings_count'] = data['host_total_listings_count'].fillna(0).astype(int)
    data['host_response_time'] = data['host_response_time'].fillna('0')
    data['host_response_rate'] = data['host_response_rate'].fillna(0)
    data['host_acceptance_rate'] = data['host_acceptance_rate'].fillna(0)
    data['host_is_superhost'] = data['host_is_superhost'].fillna('0')
    data['host_listings_count'] = data['host_listings_count'].fillna(0).astype(int)
    data['host_total_listings_count'] = data['host_total_listings_count'].fillna(0).astype(int)
    data['host_has_profile_pic'] = data['host_has_profile_pic'].fillna('0')
    data['host_identity_verified'] = data['host_identity_verified'].fillna('0')

    data['description'] = data['description'].fillna('0')
    data['neighborhood_overview'] = data['neighborhood_overview'].fillna('0')
    data['host_about'] = data['host_about'].fillna('0')

    for i in [
        'review_scores_rating',
        'review_scores_cleanliness', 
        'review_scores_checkin',
        'review_scores_communication', 
        'review_scores_location', 
        'reviews_per_month' ]:
        data[i] = data[i].fillna(data[i].mean())

    if train:
        data['curr'] = data['price'].apply(lambda x: x[0])
        data['price'] = (
            data['price']
                    .str.replace('$', '', regex=True)
                    .str.replace('€', '', regex=True)
                    .str.replace('₽', '', regex=True)
                    .str.replace(',', '', regex=True)
                    .astype(float)
        )

    data['host_response_rate'] = (
        data['host_response_rate']
                .str.replace('%' , '', regex=True)
                .astype(float)
    )
    data['host_acceptance_rate'] = (
        data['host_acceptance_rate']
                .str.replace('%' , '', regex=True)
                .astype(float)
    )
    return data


def calc_distance(row):
    obj_coords = (row['latitude'], row['longitude'])
    city_coords = (row['city_lat'], row['city_long'])
    d = geopy.distance.distance(city_coords, obj_coords).km
    return(d)


def coder(data, name_list):
    data_encod = {}
    for name in name_list:
        i = 1
        dicti={}
        for val in data[name].unique():
            dicti[val] = i
            i+=1
        data_encod[name] = dicti
    return data_encod


def attr_dist(row):
    list_attract = [
        (41.8902, 12.4923), 
        (41.9022, 12.4539), 
        (40.7460, 14.4989), 
        (43.7696, 11.2568), 
        (45.4375, 12.3335), 
        (45.4382, 12.3358), 
        (45.8760, 9.0661), 
        (45.6041, 10.6355), 
        (40.6340, 14.6034), 
        (43.7229, 10.3966) ] 
    dist = []
    station_coords = (row['latitude'], row['longitude'])
    for i in list_attract:
        d = geopy.distance.distance(i, station_coords).km
        dist.append(d)
    return(min(dist))


def preworking(data_privat): 
    data_privat['host_location'] = data_privat['host_location'].fillna(0).astype(str).apply(lambda x: x.split(',')[0])
    data_privat.loc[data_privat['host_location'] == data_privat['neighbourhood_cleansed'], 'he_is_here'] = 1
    data_privat.loc[data_privat['host_location'] != data_privat['neighbourhood_cleansed'], 'he_is_here'] = 2

    coord_privat = data_privat[['latitude', 'longitude', 'city', 'accommodates']]
    coord_privat.loc[(coord_privat['city'] == 'Bergamo') | (coord_privat['city'] == 'Milano'), 'region'] = 'Lombardia'
    coord_privat.loc[coord_privat['city'] == 'Bologna', 'region'] = 'Emilia-Romagna'
    coord_privat.loc[coord_privat['city'] == 'Firenze', 'region'] = 'Toscana'
    coord_privat.loc[coord_privat['city'] == 'Napoli', 'region'] = 'Campania'
    coord_privat.loc[coord_privat['city'] == 'Puglia', 'region'] = 'Puglia'
    coord_privat.loc[coord_privat['city'] == 'Roma', 'region'] = 'Lazio'
    coord_privat.loc[coord_privat['city'] == 'Sicilia', 'region'] = 'Sicilia'
    coord_privat.loc[coord_privat['city'] == 'Trentino', 'region'] = 'Trentino'
    coord_privat.loc[coord_privat['city'] == 'Venezia', 'region'] = 'Veneto'

    cord_cod={}
    i = 0
    dicti={}
    for val in ['Lombardia', 'Emilia-Romagna','Toscana','Campania','Puglia','Lazio','Sicilia','Trentino','Veneto']:
        dicti[val] = i
        i+=1
    cord_cod['region'] = dicti

    coord_privat = coord_privat.replace(cord_cod)

    name_list = ['neighbourhood_cleansed', 'room_type', 'host_is_superhost', 'has_availability', 'city', 'host_response_time', 'host_identity_verified', 'instant_bookable', 
                'host_has_profile_pic', 'property_type']

    data_encod = coder(data_privat, name_list)

    data_privat = data_privat.replace(data_encod)

    data_privat['amenities'] = (
    data_privat['amenities']
            .replace('}' , '', regex=True)
            .replace('{', '', regex=True)
            .replace('"', '', regex=True)
            .replace('  ', '', regex=True)
            .replace(' ', '', regex=True)
            .replace('   ', '', regex=True)
            .replace(']', '', regex=True)
            .str.lower()
    )

    data_privat['amenities_count'] = data_privat['amenities'].apply(lambda x: len(x.split(',')))

    data_privat['parking'] = data_privat['amenities'].apply(lambda x: 1 if 'parking' in x else 0)
    data_privat['hair_dryer'] = data_privat['amenities'].apply(lambda x: 1 if 'hair' in x else 0)
    data_privat['kitchen'] = data_privat['amenities'].apply(lambda x: 1 if 'kitch' in x else 0)
    data_privat['fireplace'] = data_privat['amenities'].apply(lambda x: 1 if 'fireplace' in x else 0)
    data_privat['refrigerator'] = data_privat['amenities'].apply(lambda x: 1 if 'ref' in x else 0)
    data_privat['microwave'] = data_privat['amenities'].apply(lambda x: 1 if 'microwave' in x else 0)
    data_privat['dishwasher'] = data_privat['amenities'].apply(lambda x: 1 if 'dishwasher' in x else 0)
    data_privat['heating'] = data_privat['amenities'].apply(lambda x: 1 if 'heat' in x else 0)
    data_privat['balcony'] = data_privat['amenities'].apply(lambda x: 1 if 'balcony' in x else 0)
    data_privat['garden'] = data_privat['amenities'].apply(lambda x: 1 if 'garden' in x else 0)
    data_privat['bbq'] = data_privat['amenities'].apply(lambda x: 1 if 'grill'in x else 0)
    data_privat['pool'] = data_privat['amenities'].apply(lambda x: 1 if 'pool' in x else 0)
    data_privat['lake'] = data_privat['amenities'].apply(lambda x: 1 if 'lake' in x else 0)
    data_privat['alarm'] = data_privat['amenities'].apply(lambda x: 1 if 'alarm' in x else 0)

    data_privat['bathrooms'] = data_privat['bathrooms_text'].str.extract('(\d+)').fillna(0).astype(int)

    data_privat = preprocessing(data_privat, train=False)

    data_privat['host_since'] = pd.to_datetime(data_privat['host_since'], format='%Y-%m-%d')
    data_privat['year'] = data_privat['host_since'].dt.year
    data_privat['month'] = data_privat['host_since'].dt.month
    data_privat['day'] = data_privat['host_since'].dt.day

    good_cols = ['neighbourhood_cleansed', 'room_type', 'host_is_superhost', 
                    'has_availability', 'city', 'host_response_time', 
                    'host_identity_verified', 'instant_bookable', 'host_has_profile_pic', 
                    'property_type', 'parking', 'refrigerator', 'dishwasher', 'kitchen', 
                    'heating', 'pool', 'fireplace', 'microwave', 'balcony', 'hair_dryer', 
                    'garden', 'bbq', 'lake', 'alarm', 'review_scores_rating', 
                    'review_scores_accuracy', 'review_scores_cleanliness', 
                    'review_scores_checkin', 'review_scores_communication', 
                    'review_scores_location', 'review_scores_value', 'availability_365', 
                    'number_of_reviews', 'reviews_per_month', 'minimum_nights', 
                    'maximum_nights', 'host_response_rate', 'host_acceptance_rate', 
                    'calculated_host_listings_count', 
                    'calculated_host_listings_count_entire_homes', 
                    'calculated_host_listings_count_private_rooms', 
                    'calculated_host_listings_count_shared_rooms', 'bathrooms', 'bedrooms', 
                    'beds', 'minimum_minimum_nights', 'maximum_minimum_nights', 
                    'minimum_maximum_nights', 'maximum_maximum_nights', 
                    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 
                    'host_listings_count', 'host_total_listings_count', 'availability_30', 
                    'availability_60', 'availability_90', 'number_of_reviews_ltm', 
                    'number_of_reviews_l30d', 'year', 'month', 'day', 'amenities_count', 
                    'he_is_here', 'description', 'neighborhood_overview', 'host_about']

    data_privat = data_privat[good_cols]

    coord_privat['city_lat'] = 0 # столбец с широтой города / стлолицы региона
    coord_privat['city_long'] = 0 # столбец с долготой города / столицы региона

    coord_privat.loc[coord_privat['city'] == 'Bergamo', 'city_lat'] = 45.696
    coord_privat.loc[coord_privat['city'] == 'Bergamo', 'city_long'] = 9.66721
    coord_privat.loc[coord_privat['city'] == 'Venezia', 'city_lat'] = 45.26
    coord_privat.loc[coord_privat['city'] == 'Venezia', 'city_long'] = 12.19
    coord_privat.loc[coord_privat['city'] == 'Bologna', 'city_lat'] = 44.2937
    coord_privat.loc[coord_privat['city'] == 'Bologna', 'city_long'] = 11.2019
    coord_privat.loc[coord_privat['city'] == 'Firenze', 'city_lat'] = 43.4645
    coord_privat.loc[coord_privat['city'] == 'Firenze', 'city_long'] = 11.1446
    coord_privat.loc[coord_privat['city'] == 'Milano', 'city_lat'] = 45.4643 
    coord_privat.loc[coord_privat['city'] == 'Milano', 'city_long'] = 9.18951 
    coord_privat.loc[coord_privat['city'] == 'Napoli', 'city_lat'] = 40.5122
    coord_privat.loc[coord_privat['city'] == 'Napoli', 'city_long'] = 14.1447 
    coord_privat.loc[coord_privat['city'] == 'Puglia', 'city_lat'] = 41.1142
    coord_privat.loc[coord_privat['city'] == 'Puglia', 'city_long'] = 16.8728
    coord_privat.loc[coord_privat['city'] == 'Roma', 'city_lat'] = 41.8919 
    coord_privat.loc[coord_privat['city'] == 'Roma', 'city_long'] = 12.5113 
    coord_privat.loc[coord_privat['city'] == 'Sicilia', 'city_lat'] = 38.07
    coord_privat.loc[coord_privat['city'] == 'Sicilia', 'city_long'] = 13.22
    coord_privat.loc[coord_privat['city'] == 'Trentino', 'city_lat'] = 46.30
    coord_privat.loc[coord_privat['city'] == 'Trentino', 'city_long'] = 11.21

    coord_privat['distance'] = coord_privat.apply(calc_distance, axis=1)
    coord_privat['dist_attr'] = coord_privat.apply(attr_dist, axis=1)

    data_privat['city_dist'] = coord_privat['distance']
    data_privat['attr_dist'] = coord_privat['dist_attr']
    data_privat['accommodates'] = coord_privat['accommodates']
    data_privat['region'] = coord_privat['region']

    list_nint=[]
    for i in data_privat['neighbourhood_cleansed'].unique():
        if isinstance(i, int) == False:
            list_nint.append(i)

    list_nint_2=[]
    for i in data_privat['property_type'].unique():
        if isinstance(i, int) == False:
            list_nint_2.append(i)

    data_encod_1={}
    i = 1000
    dicti={}
    for val in list_nint:
            dicti[val] = i
            i+=1
    data_encod_1['neighbourhood_cleansed'] = dicti

    data_encod_2={}
    i = 1000
    dicti_2={}
    for val in list_nint_2:
            dicti_2[val] = i
            i+=1
    data_encod_2['property_type'] = dicti_2

    data_privat = data_privat.replace(data_encod_1)
    data_privat = data_privat.replace(data_encod_2)
    return data_privat

df = pd.read_csv('/Users/artemmotyakin/Desktop/archive/private_listings_case.csv', index_col='id')
 
data_privat = preworking(df)

# Загрузка обученной модели из файла plk 
model_list = list()
folder_path = 'prediction_app/models'
file_path = join(folder_path, 'Model_reg_B1.pkl')
model_list.append(joblib.load(file_path))
file_path = join(folder_path, 'Model_reg_B2.pkl')
model_list.append(joblib.load(file_path))
file_path = join(folder_path, 'Model_reg_B3.pkl')
model_list.append(joblib.load(file_path))
file_path = join(folder_path, 'Model_reg_B4.pkl')
model_list.append(joblib.load(file_path))
file_path = join(folder_path, 'Model_reg_B5.pkl')
model_list.append(joblib.load(file_path))

print(data_privat)
data_privat.to_csv('privat.csv', index=False)

pred = []
i = 0
for model in model_list:
    print(i)
    predictions = model.predict(data_privat)
    print(predictions)
    pred.append(predictions)
    i+=1

predictions = sum(pred) / len(pred)

predictions_df = pd.DataFrame(predictions, columns=['price'])

# Сохранение предсказаний в виде CSV файла 
predictions_df.to_csv('predictions.csv', index=False)



