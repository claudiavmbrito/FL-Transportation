import glob
import time
from typing import Tuple, Union, List


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
RFParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def clean_label(label):
    return label.lstrip(',').rstrip(',').replace(',,', ',')

def get_dataset():
    INPUT_FOLDER = '../../../data_geolife/processed_data/'
    headers_metadf = ['trajectory_id', 'v_ave', 'v_med', 'v_max', 'a_ave', 'a_med', 'a_max', 'labels']

    #Lets load all of the processed data, containing the features of all trajectories into one single dataframe. 
    #The easiest way to do this is to load all of the into a list and concatenate them.
    list_df_metadata = []
    for file in glob.glob(INPUT_FOLDER + "*.csv"):
        df_metadata = pd.read_csv(file, index_col=0)
        list_df_metadata.append(df_metadata)
    df_metadata = pd.concat(list_df_metadata)

    del df_metadata['subfolder']
    del df_metadata['datetime']
    del df_metadata['acceleration']
    del df_metadata['velocity']
    del df_metadata['lat']
    del df_metadata['long']
    del df_metadata['altitude']

    #print(df_metadata.dtypes)
    df_metadata['timedelta']=pd.to_timedelta(df_metadata.timedelta)

    df_labeled = df_metadata.dropna(subset=['v_ave','v_med','v_max', 'a_ave', 'a_med', 'a_max', 'labels'])

    df_labeled.loc[:,'labels'] = df_labeled['labels'].apply(lambda x: clean_label(x))

    #Get a new dataframe for calculating the vmean and the full distance of the trip
    df_new_values = df_labeled.groupby('trajectory_id').agg({'distance':'sum', 'timedelta':'sum'})
    df_new_values['timedelta']=pd.to_timedelta(df_new_values.timedelta)

    df_new_values['vmean'] = df_new_values['distance'] / ( df_new_values['timedelta'].dt.total_seconds() / 3600.0 )

    del df_metadata['timedelta']

    df_new_values = df_new_values.merge(df_labeled, on='trajectory_id')
    #df_new_values['labels'] = df_new_values['trajectory_id'].map(df_metadata['labels'])

    del df_new_values['v_ave']
    del df_new_values['v_med']
    del df_new_values['v_max']
    del df_new_values['a_ave']
    del df_new_values['a_med']
    del df_new_values['a_max']
    del df_new_values['distance_y']
    del df_new_values['timedelta_y']
    del df_new_values['timedelta_x']
    del df_new_values['trajectory_id']

    df_new_values['distance_x'] = df_new_values['distance_x'].astype(np.float32)
    df_new_values['vmean'] = df_new_values['vmean'].astype(np.float32)

    df_new_values = df_new_values.dropna(subset=['distance_x','vmean', 'labels'])

    print(df_new_values.head)
    print(df_new_values.dtypes)

    all_labels = df_new_values['labels'].unique()

    #We can filter out single modal trajectories by taking the labels which do not contain a comma:
    single_modality_labels = [elem for elem in all_labels if ',' not in elem]

    single_modality_labels.remove('boat')
    single_modality_labels.remove('airplane')
    single_modality_labels.remove('subway')
    single_modality_labels.remove('train')

    df_single_modality = df_new_values[df_new_values['labels'].isin(single_modality_labels)]
    
    to_general_label = {'bike': 'bike', 'run': 'foot', 'walk': 'foot', 'bus': 'passengerduty', 'car': 'vehicle', 'taxi': 'vehicle', 'motorcycle': 'motorcycle'}

    df_single_modality.loc[:,'labels'] = df_single_modality['labels'].apply(lambda x: to_general_label[x])

    mask = np.random.rand(len(df_single_modality)) < 0.7
    df_train = df_single_modality[mask]
    df_test = df_single_modality[~mask]

    #The columns 
    X_colnames = ['distance_x','vmean']
    Y_colnames = ['labels']

    X_train = df_train[X_colnames].values
    Y_train = np.ravel(df_train[Y_colnames].values)
    X_test = df_test[X_colnames].values
    Y_test = np.ravel(df_test[Y_colnames].values)

    return X_train, Y_train, X_test, Y_test


def get_model_parameters(model: RandomForestClassifier) -> RFParams:
    """Returns the paramters of a sklearn RandomForest model."""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(
    model: RandomForestClassifier, params: RFParams
) -> RandomForestClassifier:
    """Sets the parameters of a sklean RandomForestClassifier model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: RandomForestClassifier):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. 
    """
    n_classes = 6 
    n_features = 2  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )