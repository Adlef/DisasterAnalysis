# DisasterAnalysis
This project builds a machine learning model which, based on a text message, classifies it into different disaster categories (if it is a disaster message).

The application is implemented into the project `MyWebApp (https://github.com/Adlef/MyWepApp)`

- To run ETL pipeline that cleans data and stores in database
    `python process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse.db`

- To run ML pipeline that trains classifier and saves it (make sure if don't have gpu available, in build_model function pipeline remove n_jobs=-1)
    `python train_classifier.py DisasterResponse.db model/model_ada.sav` 