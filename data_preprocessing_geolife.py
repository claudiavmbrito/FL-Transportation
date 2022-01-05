import pandas as pd
import datetime as dt
import geopy
import numpy as np
import os
from geopy import distance
from multiprocessing.pool import Pool

#time format conversion
def to_datetime(string):
    return dt.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')


def traject_calculator(long_a, lat_a, long_b, lat_b):
#latitude cannot be <-90 or >90
#longitude cannot be <-180 or >180
  if lat_a == lat_b and long_a == long_b:
    return 0
  if False in np.isfinite([long_a, long_b, lat_a, lat_b]):
        return np.nan
  if lat_a < -90 or lat_a > 90 or lat_b < -90 or lat_b > 90:
    return np.nan
  if long_a < -180 or long_a > 180 or long_b < -180 or long_b > 180:
    return np.nan
  trajectory = distance.distance((lat_a, long_a),(lat_b, long_b)).km
  return trajectory

#velocity equals to the distance between two coordinates / time_variation
#time variation equals to the difference between the time of 2 consecutive points 
def velocity_calculator(distance, time_variation):
  if time_variation.total_seconds() == 0:
      return np.nan 
  else:
    speed = distance / time_variation.total_seconds()
    return speed
#acceleration equals to the difference between the velocity of 2 consecutive points 
#time variation is similar as above
def acceleration_calculator(a_speed, b_speed, time_variation):
  velocity_variation = b_speed - a_speed
  if time_variation.total_seconds() == 0:
      return np.nan 
  else: 
    acceleration = velocity_variation/time_variation.total_seconds()
    return acceleration


def transform_labels(filepath):
  df_label = pd.read_csv(filepath, sep='\t')
  df_label['start_time'] = df_label['Start Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
  df_label['end_time'] = df_label['End Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
  df_label['labels'] = df_label['Transportation Mode']
  df_label = df_label.drop(['End Time', 'Start Time', 'Transportation Mode'], axis=1)
  return df_label


#header for .plt file - new dataframe
header_gps = ['lat','long','null','altitude','timestamp','date','time']

def load_trajectory_df(full_filename):
    subfolder = full_filename.split('/')[-3]
    trajectory_id = full_filename.split('/')[-1].split('.')[0]
    
    df_traject = pd.read_csv(full_filename, skiprows = 6, header = None, names = header_gps)
   
    df_traject['datetime'] = df_traject.apply(lambda z: to_datetime(z.date + ' ' + z.time), axis=1)
    df_traject['datetime_next_position'] = df_traject['datetime'].shift(-1)
    df_traject['timedelta'] = df_traject.apply(lambda z: z.datetime_next_position - z.datetime, axis=1)
    df_traject = df_traject.drop(['datetime_next_position'], axis=1)
    df_traject = df_traject.drop(['null', 'timestamp', 'date', 'time'], axis=1)
    
    
    df_traject['long_next_position'] = df_traject['long'].shift(-1)
    df_traject['lat_next_position'] = df_traject['lat'].shift(-1)
    df_traject['distance'] = df_traject.apply(lambda z: traject_calculator(z.long, z.lat, z.long_next_position, z.lat_next_position), axis=1)
    df_traject = df_traject.drop(['long_next_position', 'lat_next_position'], axis=1)
    
    df_traject['velocity'] = df_traject.apply(lambda z: velocity_calculator(z.distance, z.timedelta), axis=1)
    df_traject['velocity_next_position'] = df_traject['velocity'].shift(-1)
    df_traject['acceleration'] = df_traject.apply(lambda z: acceleration_calculator(z.velocity, z.velocity_next_position, z.timedelta), axis=1)
    df_traject = df_traject.drop(['velocity_next_position'], axis=1)
    
    df_traject['trajectory_id'] = trajectory_id
    df_traject['subfolder'] = subfolder
    df_traject['labels'] = ''
    calculate_agg_features(df_traject)
    return df_traject

#for more accurate results/enriching this data, the average as well as max and min of such values are calculated.
#this data is stored as well in the dataframe and in the metadata of the dataframe
def calculate_agg_features(df):
    v_ave = np.nanmean(df['velocity'].values)
    v_med = np.nanmedian(df['velocity'].values)
    v_max = np.nanmax(df['velocity'].values)
    a_ave = np.nanmean(df['acceleration'].values)
    a_med = np.nanmedian(df['acceleration'].values)
    a_max = np.nanmax(df['acceleration'].values)
   
    df.loc[:, 'v_ave'] = v_ave
    df.loc[:, 'v_med'] = v_med
    df.loc[:, 'v_max'] = v_max
    df.loc[:, 'a_ave'] = a_ave
    df.loc[:, 'a_med'] = a_med
    df.loc[:, 'a_max'] = a_max


'''
df = pd.read_csv('data_geolife/Data/000/Trajectory/20081023025304.plt', skiprows = 6, header = None, names = header_gps)
df['datetime'] = df.apply(lambda z: to_datetime(z.date + ' ' + z.time), axis=1)
df['datetime_next_position'] = df['datetime'].shift(-1)
df['timedelta'] = df.apply(lambda z: z.datetime_next_position - z.datetime, axis=1)
df = df.drop(['datetime_next_position'], axis=1)
df = df.drop(['null', 'timestamp', 'date', 'time'], axis=1)
df.head()
'''

LABELS_FILE = 'labels.txt'
MAIN_FOLDER = 'data_geolife/Data/'
TRAJ_FOLDER = 'Trajectory/'
OUTPUT_FOLDER = 'data_geolife/processed_data/'
POOLSIZE = 10

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
directories = os.listdir(MAIN_FOLDER)

for subfolder in directories:
    list_df_traj = []
    subfolder_ = MAIN_FOLDER + subfolder + '/'
    traj_folder = MAIN_FOLDER + subfolder + '/' + TRAJ_FOLDER
    traj_files = os.listdir(traj_folder)
    
    traj_files_full_path = [traj_folder + traj_file for traj_file in traj_files]
    print(subfolder, len(traj_files_full_path))
    
    #multiprocessing does not work well in the jupyter notebook environment.
    #outside of jupyter you can use multiprocessing to speed up the process
    #pool = Pool(POOLSIZE)
    #for df in pool.imap_unordered(load_trajectory_df, traj_files_full_path):
    #    list_df_traj.append(df)
    
    for file in traj_files_full_path:
        list_df_traj.append(load_trajectory_df(file))
    
    df_traj_all = pd.concat(list_df_traj)
    list_df_traj = []
    
    if LABELS_FILE in os.listdir(subfolder_):
        filename = subfolder_ + LABELS_FILE
        df_labels = transform_labels(filename)
        for idx in df_labels.index.values:
            st = df_labels.loc[idx]['start_time']
            et = df_labels.loc[idx]['end_time']
            labels = df_labels.loc[idx]['labels']
            if labels:
                df_traj_all.loc[(df_traj_all['datetime'] >= st) & 
                                (df_traj_all['datetime'] <= et), 'labels'] = labels

    output_filename = OUTPUT_FOLDER + subfolder + '.csv'
    df_traj_all.to_csv(output_filename)
    del df_traj_all


#need to add main to work for the correct directory