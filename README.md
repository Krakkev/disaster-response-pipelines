# Disaster Response Pipeline Project

With this project you will train a model based on the data coming from disaster messages and their categories related, 
to predict the categories related to new disaster messages.

Ones that have already created the model, you will run a webb app where you will see 3 different charts
to have a small overview about the data used for training and you will be able to input new messages to predict their 
categories.

This could be very useful in real world situations to prioritize urgent messages.

### Setup your environment
```bash
# Creating virtual env.
virtualenv env -p python3

# Activating virtual env.
source env/bin/activate

# Install libraries
pip install -r requirements.txt

```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


