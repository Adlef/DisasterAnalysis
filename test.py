# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:11:26 2020

@author: Adrien
"""

import pickle
from wrangling_scripts.DisasterMessages.train_classifier import *

def get_feature_importance(model, category_names, database_filepath):
    '''collect important features from model and store in database
    Function get the weights of most important (words)
    features, their weights, and the category in database
    'word' table after training
    Args:
      model: name of model
      category_names: list of category name of array Y
      database_filepath: name of database containing data
    Returns:
      None
    ''' 
    # TAKE THE BEST ESTIMATOR FROM MODEL (FROM GRIDSEARCHCV)
    best_pipeline = model.best_estimator_
    #best_pipeline = model
    col_name = []
    imp_value = []
    imp_word = []
    # List vocabulary
    x_name = best_pipeline.named_steps['vect'].get_feature_names()
    # GET FEATURE IMPORTANCES FROM THE LEARNING MODEL AND FOR A SPECIFIC CATEGORY
    for j, col in enumerate(category_names):
        x_imp = best_pipeline.named_steps['clf'].estimators_[j].feature_importances_
        # LIMIT FOR WEIGHT OF FEATURES SET TO MINMUM OF VALUE
        value_max = x_imp.max() / 2.0
        
        # GET FEATURES NOT LESS THAN THE MINIMUM WEIGHT SET PER COLUMN - NO POINT TO DISPLAY ALL FEATURES
        for i,value in enumerate(x_imp):
            if(value > value_max):
                col_name.append(col)
                imp_value.append(value)
                imp_word.append(x_name[i])
                if col == 'cold':
                    print("great")

    # PREPARING DATAFRAME
    col_name = np.array(col_name).reshape(-1, 1)
    imp_value = np.array(imp_value).reshape(-1, 1)
    imp_word = np.array(imp_word).reshape(-1, 1)
    imp_array = np.concatenate((col_name, imp_value, imp_word), axis=1)
    df_imp = pd.DataFrame(imp_array, columns=['category_name', 'importance_value', 'important_word'])  
    
    # IMPORTANCE VALUE SHOULD BE A FLOAT
    df_imp.importance_value = pd.to_numeric(df_imp.importance_value, downcast='float')

    # CREATING SQL ENGINE
    engine = create_engine('sqlite:///' + database_filepath)

    # SAVING DATAFRAME INTO A TABLE
    df_imp.to_sql('Words', engine, if_exists='replace', index=False) 
    df_imp = pd.read_sql("SELECT * FROM Words", engine)
    
    print('Sample feature importance...')
    print(df_imp.head())
    print(len(df_imp))
    
if __name__ == '__main__':
    with open('model/model_ada_v0.sav', 'rb') as f:
        model = pickle.load(f) 
    
    # load data
    engine = create_engine(
        'sqlite:///DisasterResponse.db')

    df_data = pd.read_sql("SELECT * FROM DisasterMessages", engine)
    category_names = list(df_data.columns[4:])
    get_feature_importance(model, category_names, 'DisasterResponse.db')