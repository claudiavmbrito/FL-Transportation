import glob
import time
from itertools import cycle


import pandas as pd
import numpy as np
import sklearn
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder


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

    return X_train, Y_train, X_test, Y_test, df_test

def model_training():

    X_train, Y_train, X_test, Y_test, df_test = get_dataset()
    rf_classifier = RandomForestClassifier(n_estimators = 15)

    t_start = time.time()
    rf_classifier.fit(X_train, Y_train)
    model = rf_classifier.fit(X_train, Y_train)
    t_end = time.time()
    t_diff = t_end - t_start

    dump(rf_classifier, 'rf_trained.joblib')

    train_score = rf_classifier.score(X_train, Y_train)
    test_score = rf_classifier.score(X_test, Y_test)
    y_pred_rf= rf_classifier.predict(X_test)
    #print(y_pred_rf)
    print("trained Random Forest in {:.2f} s.\t Score on training / test set: {} / {}".format(t_diff, train_score, test_score))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap.sample(X_test, 1000))

    print(f'Shape of test dataset: {X_test.shape}')
    print(f'Type of shap_values: {type(shap_values)}. Lenght of the list: {len(shap_values)}')
    print(f'Shape of shap_values: {np.array(shap_values).shape}')
    
    list_features = df_test.columns.tolist()
    B = {'distance_x':"Distance",'vmean':"Velocity", 'labels': "Labels"}
    C = (pd.Series(list_features)).map(B)
    D = list(C)
    print(C)

    #shap.dependence_plot(X_train.labels, shap_values[1], features=X_test)
    I = shap.summary_plot(shap_values, X_test, feature_names=D, plot_type='bar', cmap='gray',show=False)
    plt.legend(frameon=False, loc='lower center', ncol=6)
    #plt.savefig("summary_plot_max.pdf")
    plt.savefig("summary_plot_names.pdf") 
 
    return train_score, test_score, y_pred_rf

train_score, test_score, y_pred_rf = model_training()

print("Score on training / test set: {} / {}".format( train_score, test_score))