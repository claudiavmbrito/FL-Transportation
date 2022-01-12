import pandas as pd
from joblib import load

import emissions
import rf_train

from emissions import EmissionModels
from rf_train import get_dataset

_, _, X_test, Y_test = get_dataset()

print(X_test.head)

velocity = 45
segment = 'Passenger Car'
fuel = 'Petrol'
kilometers = 150
year = 2018

rf_classifier = load('rf_trained.joblib') 
y_pred_rf= rf_classifier.predict(X_test)
print(y_pred_rf)

print('Emissions: \n')
print('CO: ' + str(EmissionModels.full_report(vmean=velocity, fuel=fuel, segment=segment, distance=kilometers, year=year)[0]) + ' g \n')
print('NOx: ' + str(EmissionModels.full_report(vmean=velocity, fuel=fuel, segment=segment, distance=kilometers)[1]) + ' g \n' )
print('PM 2.5: ' + str(EmissionModels.full_report(vmean=velocity, fuel=fuel, segment=segment, distance=kilometers)[2]) + ' g \n')
print('CO2: ' + str(EmissionModels.full_report(vmean=velocity, fuel=fuel, segment=segment, distance=kilometers)[3]) + ' g \n')
print('CH4: ' + str(EmissionModels.full_report(vmean=velocity, fuel=fuel, segment=segment, distance=kilometers)[4]) + ' g \n')
print('VOC: ' + str(EmissionModels.full_report(vmean=velocity, fuel=fuel, segment=segment, distance=kilometers)[5]) + ' g \n')


