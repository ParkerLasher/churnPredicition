# scripts/utils.py
import os
import pandas as pd

def load_enhanced_data():
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'enhanced_data.csv')
    data = pd.read_csv(data_path)
    return data