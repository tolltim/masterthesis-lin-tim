# masterthesis-lin-tim
This repo contains the random forest model which was built for my masterthesis based on various open source data input. 

### Installation
1. clone the repo: git clone <repo>
2. navigate to the cloned repo: cd <repo>
3. Install the required packages: pip install -r requirements.txt
4. Change the parameters in the get_config* in rf_model.py if necessary
5. Run the rf_model.py and like it

*To change the parameters of the rf model, look at the picture in the repo or the train-data/variables.csv and choose numbers to predict the speed data. 
The target features can be selected with the same way

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
