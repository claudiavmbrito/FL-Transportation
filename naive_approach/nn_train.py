import glob
import time

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def clean_label(label):
    return label.lstrip(',').rstrip(',').replace(',,', ',')

INPUT_FOLDER = '../data_geolife/processed_data/'
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


train_df, test_df = train_test_split(df_single_modality, test_size=0.2, random_state=233)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=233)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('labels'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('labels'))
test_labels = np.array(test_df.pop('labels'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)


scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


METRICS = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.Accuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')
      #keras.metrics.AUC(name='auc'),
      #keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          1024, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(
          512, activation='relu'),
      keras.layers.Dense(
          256, activation='relu'),
      keras.layers.Dense(
          128, activation='relu'),
      keras.layers.Dense(
          64, activation='relu'),
      keras.layers.Dense(
          32, activation='relu'),
      keras.layers.Dense(
          16, activation='relu'),
      keras.layers.Dense(
          8, activation='relu'),
      keras.layers.Dense(
          4, activation='relu'),
      keras.layers.Dense(6, activation='softmax',
                         bias_initializer=output_bias),
  ])


  model.compile(
      optimizer=keras.optimizers.SGD(learning_rate=1e-2),
      loss=keras.losses.MeanSquaredError(),
      metrics='Accuracy')


  return model

def make_model2():
  # Model and Compile
  model = keras.Sequential()
  activ = 'relu'
  model.add(keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1,train_features.shape[-1],1)))
  model.add(keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ))
  model.add(keras.layers.MaxPooling2D(pool_size=(1, 2)))

  model.add(keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
  model.add(keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
  model.add(keras.layers.MaxPooling2D(pool_size=(1, 2)))

  model.add(keras.layers.Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
  model.add(keras.layers.Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
  model.add(keras.layers.MaxPooling2D(pool_size=(1, 2)))
  model.add(keras.layers.Dropout(.5))

  model.add(keras.layers.Flatten())
  A = model.output_shape
  model.add(keras.layers.Dense(int(A[1] * 1/4.), activation=activ))
  model.add(keras.layers.Dropout(.5))

  model.add(keras.layers.Dense(1, activation='softmax'))

  optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

EPOCHS = 100
BATCH_SIZE = 256

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model2()
model.summary()

train_features = tf.expand_dims(train_features, axis=-2)
test_features = tf.expand_dims(test_features, axis=-2)
val_features = tf.expand_dims(val_features, axis=-2)

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
print("Acc: {:0.4f}".format(results[1]))

naive_training = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=100,
    validation_data=(val_features, val_labels),
    callbacks=[early_stopping])

results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
print("Acc: {:0.4f}".format(results[1]))