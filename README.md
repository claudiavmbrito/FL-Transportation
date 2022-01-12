# TransportationFL

This repository has the first two stages to train the Federated Learning models locally.

## Naive Approach

In the `naive_approach`folder it is found four scripts. Such folder does not comprise the federated learning approach. 

- The `rf_train.py` python script allows to train a random forest model which obtains nearly 80% of accuracy.
- The `emissions.py` python script is based on the current information of emissions calculations. For the full calculation of emissions it needs the following information: i) `vmean`- the mean velocity of the trip; ii) `fuel`- the type of fuel of the car; iii) `segment`- the segment of the vehicle (i.e., `Passenger`, `PassengerDuty`, `HeavyDuty`, `Motorcycle`); iv) `distance`- the total distance of the trip; v) `year` - the year of the vehicle.
- The `standards.py`python script calculates the Euro Standard of the vehicles based on its year and the segment. This script is called by `emissions.py`.
- The `main.py` python script fetches the trained model and tests it with new data, in the end it outputs the calculated emissions.

### For simple tests run in this order:

1. `download_dataset.sh` - this script downloads the GeoLife data set and moves its contents to the `data_geolife` folder.
2. `data_preprocessing_geolife.py`- this script processes the previous data. It transforms the raw data into data with information of the trip's velocity and acceleration while joining it with its labels. 
3. `rf_train.py` - this script contains information to train a random forest model which obtains nearly 81% of accuracy.

### TODO

- Calculation of the distance and vmean in separate file;
- Get emissions data with less decimal numbers;

## Federated Learning Approach

The `federated` folder comprises both `android` and `ios`versions. 

