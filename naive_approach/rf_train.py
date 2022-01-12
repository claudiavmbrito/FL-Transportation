import glob
import time

import pandas as pd
import numpy as np
import sklearn
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def clean_label(label):
    return label.lstrip(',').rstrip(',').replace(',,', ',')

def get_dataset():
    INPUT_FOLDER = '../../data_geolife/processed_data/'
    headers_metadf = ['trajectory_id', 'v_ave', 'v_med', 'v_max', 'a_ave', 'a_med', 'a_max', 'labels']

    #Lets load all of the processed data, containing the features of all trajectories into one single dataframe. 
    #The easiest way to do this is to load all of the into a list and concatenate them.
    list_df_metadata = []
    for file in glob.glob(INPUT_FOLDER + "*.csv"):
        df_metadata = pd.read_csv(file, index_col=0)
        list_df_metadata.append(df_metadata)
    df_metadata = pd.concat(list_df_metadata)

    #Get a new dataframe for calculating the vmean and the full distance of the trip
    df_new_values = df_metadata.groupby('trajectory_id').agg({'distance':'sum', 'timedelta':'sum'})
    print(df_new_values.head)
    print(df_new_values.dtypes)

    del df_metadata['subfolder']
    del df_metadata['datetime']
    del df_metadata['timedelta']
    del df_metadata['acceleration']
    del df_metadata['velocity']
    del df_metadata['lat']
    del df_metadata['long']
    del df_metadata['altitude']

    df_labeled = df_metadata.dropna(subset=['v_ave','v_med','v_max', 'a_ave', 'a_med', 'a_max', 'labels'])

    df_labeled.loc[:,'labels'] = df_labeled['labels'].apply(lambda x: clean_label(x))

    all_labels = df_labeled['labels'].unique()

    #We can filter out single modal trajectories by taking the labels which do not contain a comma:
    single_modality_labels = [elem for elem in all_labels if ',' not in elem]

    single_modality_labels.remove('boat')
    single_modality_labels.remove('airplane')
    single_modality_labels.remove('subway')
    single_modality_labels.remove('train')

    df_single_modality = df_labeled[df_labeled['labels'].isin(single_modality_labels)]
    
    to_general_label = {'bike': 'bike', 'run': 'foot', 'walk': 'foot', 'bus': 'passengerduty', 'car': 'vehicle', 'taxi': 'vehicle', 'motorcycle': 'motorcycle'}

    df_single_modality['labels'] = df_single_modality['labels'].apply(lambda x: to_general_label[x])

    mask = np.random.rand(len(df_single_modality)) < 0.7
    df_train = df_single_modality[mask]
    df_test = df_single_modality[~mask]

    #The columns 
    X_colnames = ['v_ave','v_med','v_max',  'a_ave', 'a_med', 'a_max']
    Y_colnames = ['labels']

    X_train = df_train[X_colnames].values
    Y_train = np.ravel(df_train[Y_colnames].values)
    X_test = df_test[X_colnames].values
    Y_test = np.ravel(df_test[Y_colnames].values)

    return X_train, Y_train, X_test, Y_test, df_new_values

#get dataset for training and testing the model
X_train, Y_train, X_test, Y_test, df_new_values = get_dataset()


rf_classifier = RandomForestClassifier(n_estimators = 15)

t_start = time.time()
rf_classifier.fit(X_train, Y_train)
t_end = time.time()
t_diff = t_end - t_start

dump(rf_classifier, 'rf_trained.joblib')

train_score = rf_classifier.score(X_train, Y_train)
test_score = rf_classifier.score(X_test, Y_test)
y_pred_rf= rf_classifier.predict(X_test)
print(y_pred_rf)
print("trained Random Forest in {:.2f} s.\t Score on training / test set: {} / {}".format(t_diff, train_score, test_score))

acc_forest = accuracy_score(Y_test, y_pred_rf)
print(" RF ," + str(acc_forest) + "\n")
