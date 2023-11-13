# masterthesis-lin-tim
This repo contains the random forest model which was built for my masterthesis based on various open source data input. 

### Installation
1. clone the repo: git clone <https://github.com/tolltim/masterthesis-lin-tim.git>
2. navigate to the cloned repo: cd <location of repo>
3. Install the required packages: pip install -r requirements.txt
4. Change the parameters in the get_config* in rf_model.py if necessary
5. Run the xxx_model_xx.py and answer the questions for the model based on the input (hyperparameter tuning, selection of features** and dataset splitting)


*To change the parameters of the rf model, look at the picture in the repo or the train-data/variables.csv and choose numbers to predict the speed data. 
The target features can be selected with the same way

**If using all features for prediction, this model is not a prediction model anymore, because the target variables are predicting

### Models
rf_model_au: Predictive model for Südliche Au with imputing by averaging by weekday

rf_model_both: predictive model for either au or walchenseeplatz by using all features from both areas (not very much improvement)

rf_model_testing: Using only data from before the road closure for predicten after the road closure (no good model)

rf_model_wp: Predictive model for Walchenseeplatz with imputing by averaging by weekday

predict-model-au.py: USing wp_model for predicting it for südliche Au


### Datasets

train-data-au: Training data for rf_model_au.py for Südliche Auu

train-data-wp: training data for rf_model_wp.py for Walchenseeplatz area

train-data-au-new: Trainings data for rf_model_testing.py, which just predicts future values by historical values without imputing

train-data-all: Training data including all variables

### Get data
the folder get-data has all the scripts which were used to get the open source data. 
The API-key needs to be adjusted and some special fixes in each script are described in the script itself.
To run the get-data every hour you can use the windows task planer.
All the data collected needs specialized processing, which is not part of this repo. 

### Disclaimer
No Validation of Correctness: The code, data, and results in this repository are provided as-is. 
While every effort has been made to ensure the accuracy and correctness of the information contained in this repository, 
no guarantee or validation of correctness is provided.

Data Usage: The data included in this repository is meant solely for private and educational purposes. 
Any redistribution, commercial use, or misuse of the data without appropriate permissions is strictly prohibited.

Limitation of Liability: By using the contents of this repository, users acknowledge that they do so at their own discretion 
and will not hold the repository owner or contributors accountable for any errors, omissions, or unexpected outcomes that may arise.

### Master Thesis

topic of thesis: Creating a Predictive Model for the Traffic Impacts of Road Closures: A study in the Area of Munich. 
studies: Transportation Systems Msc. Technical University of Munich 
