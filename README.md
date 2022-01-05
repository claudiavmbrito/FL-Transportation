# TransportationFL

This repository has the first two stages to train the Federated Learning models locally.

Run in this order:

1. `download_dataset.sh` - this script downloads the GeoLife data set and moves its contents to the `data_geolife` folder.
2. `data_preprocessing_geolife.py`- this script processes the previous data. It transforms the raw data into data with information of the trip's velocity and acceleration while joining it with its labels. 
3. `rf_train.py` - this script contains information to train a random forest model which obtains nearly 80% of accuracy.

