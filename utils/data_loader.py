import pandas as pd
import os

def load_data():
    # Search paths for the dataset
    paths = [
        "gender_pay_gap_india.csv",
        "../gender_pay_gap_india.csv",
        "c:/Users/tejas/Downloads/aiml project/Tejas-Aiml/gender_pay_gap_india.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
            
    # Raise error if not found
    raise FileNotFoundError("Could not locate gender_pay_gap_india.csv in any standard search path.")
