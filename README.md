# Federated Learning for User's Transportation Modes Classification

This repository has the first two stages to train the Federated Learning models locally.
This work started with an internship at Ubiwhere. 

**Note:** This work is being updated. 

## Naive Approach

In the `naive_approach`folder four scripts are found. This folder does not comprise the federated learning approach. 

- The `rf_train.py` python script allows training a random forest model, which obtains nearly 80% of accuracy.
- The `emissions.py` python script is based on the current information of emissions calculations. For the full calculation of emissions, it needs the following information: i) `vmean`- the mean velocity of the trip; ii) `fuel`- the type of fuel of the car; iii) `segment`- the segment of the vehicle (i.e., `Passenger`, `PassengerDuty`, `HeavyDuty`, `Motorcycle`); iv) `distance`- the total distance of the trip; v) `year` - the year of the vehicle.
- The `standards.py`python script calculates the Euro Standard of the vehicles based on their year and the segment. This script is called `emissions.py`.
- The `main.py` python script fetches the trained model and tests it with new data. In the end it outputs the calculated emissions.

### For simple tests, run in this order:

1. `download_dataset.sh` - this script downloads the GeoLife data set and moves its contents to the `data_geolife` folder.
2. `data_preprocessing_geolife.py`- this script processes the previous data. It transforms the raw data into data with information on the trip's velocity and acceleration while joining it with its labels. 
3. `rf_train.py` - this script contains information to train a random forest model, which obtains nearly 81% of accuracy.
4. New models `nn_train.py` and `tree_train.py` are being defined similarly to `rf_train.py`.

### TODO

- Calculation of the distance and vmean in a separate file;
- Get emissions data with less decimal numbers;

## Federated Learning Approach

The `federated` folder comprises both `android` and `ios` versions and an `iot` version that can be naively run within servers. 

Go to `federated/iot` to know more.


___
## Contact

Please reach out to `claudia.v.brito@inesctec.pt` with any questions.
