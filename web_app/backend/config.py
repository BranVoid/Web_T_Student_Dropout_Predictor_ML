import os

class Config:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model/saved_models/xgboost.pkl')
    PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'ml_model/saved_models/preprocessor.pkl')