import pandas as pd
from joblib import load

import emissions
import rf_train

from emissions import EmissionModels
from rf_train import get_dataset

#_, _, X_test, Y_test, _ = get_dataset()

velocity = 45
category = 'Passenger Cars'
fuel = 'Petrol'
kilometers = 150
year = 2018

#rf_classifier = load('rf_trained.joblib') 
#y_pred_rf= rf_classifier.predict(X_test)
#print(y_pred_rf)

emissions = EmissionModels(velocity, kilometers)

print('Emissions: \n')
print('CO: ' + str(emissions.full_report(vmean=velocity, fuel=fuel, category=category, distance=kilometers, year=year)[0]) + ' g \n')
print('NOx: ' + str(emissions.full_report(vmean=velocity, fuel=fuel, category=category, distance=kilometers, year=year)[1]) + ' g \n' )
print('PM 2.5: ' + str(emissions.full_report(vmean=velocity, fuel=fuel, category=category, distance=kilometers,year=year)[2]) + ' g \n')
print('CO2: ' + str(emissions.full_report(vmean=velocity, fuel=fuel, category=category, distance=kilometers, year=year)[3]) + ' g \n')
print('CH4: ' + str(emissions.full_report(vmean=velocity, fuel=fuel, category=category, distance=kilometers, year=year)[4]) + ' g \n')
print('VOC: ' + str(emissions.full_report(vmean=velocity, fuel=fuel, category=category, distance=kilometers, year=year)[5]) + ' g \n')


