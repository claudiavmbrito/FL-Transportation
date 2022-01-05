import glob
import pandas as pd
import numpy as np
import time
from IPython.display import display, HTML

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt


from sklearn import model_selection
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score


def clean_label(label):
    return label.lstrip(',').rstrip(',').replace(',,', ',')

INPUT_FOLDER = 'data_geolife/processed_data/'
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
del df_metadata['timedelta']
del df_metadata['acceleration']
del df_metadata['velocity']
del df_metadata['lat']
del df_metadata['long']
del df_metadata['altitude']


df_labeled = df_metadata.dropna(subset=['v_ave','v_med','v_max', 'a_ave', 'a_med', 'a_max', 'labels'])

df_labeled.loc[:,'labels'] = df_labeled['labels'].apply(lambda x: clean_label(x))

all_labels = df_labeled['labels'].unique()
print("Example of trajectory labels:")
for label in all_labels[0:5]:
    print(label)

#We can filter out single modal trajectories by taking the labels which do not contain a comma:
single_modality_labels = [elem for elem in all_labels if ',' not in elem]

df_single_modality = df_labeled[df_labeled['labels'].isin(single_modality_labels)]

print("\nTotal number of trajectories: {}".format(len(df_metadata)))
print("Total number of labeled trajectories: {}".format(len(df_labeled)))
print("Total number of single modality trajectories: {}".format(len(df_single_modality)))

mask = np.random.rand(len(df_single_modality)) < 0.7
df_train = df_single_modality[mask]
df_test = df_single_modality[~mask]

#print(len(df_train))
print(f'Data shape: {df_train.shape}')
#print(df_single_modality.head())

#The columns 
X_colnames = ['v_ave','v_med','v_max',  'a_ave', 'a_med', 'a_max']
Y_colnames = ['labels']

X_train = df_train[X_colnames].values
Y_train = np.ravel(df_train[Y_colnames].values)
X_test = df_test[X_colnames].values
Y_test = np.ravel(df_test[Y_colnames].values)

rf_classifier = RandomForestClassifier(n_estimators = 15)

t_start = time.time()
rf_classifier.fit(X_train, Y_train)
t_end = time.time()
t_diff = t_end - t_start

train_score = rf_classifier.score(X_train, Y_train)
test_score = rf_classifier.score(X_test, Y_test)
y_pred_rf= rf_classifier.predict(X_test)
print(y_pred_rf)
print("trained Random Forest in {:.2f} s.\t Score on training / test set: {} / {}".format(t_diff, train_score, test_score))

acc_forest = accuracy_score(Y_test, y_pred_rf)
print(" RF ," + str(acc_forest) + "\n")

